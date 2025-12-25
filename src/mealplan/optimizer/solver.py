"""Core LP/QP solver implementation."""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize

from mealplan.db.queries import NutrientQueries
from mealplan.optimizer.constraints import ConstraintBuilder
from mealplan.optimizer.models import (
    FoodResult,
    NutrientResult,
    OptimizationRequest,
    OptimizationResult,
)


def solve_lp(
    costs: np.ndarray,
    nutrient_matrix: np.ndarray,
    nutrient_mins: np.ndarray,
    nutrient_maxs: np.ndarray,
    food_bounds: list[tuple[float, float]],
) -> dict[str, Any]:
    """Solve linear programming problem using scipy.optimize.linprog with HiGHS.

    Objective: min c'x
    Subject to:
        Ax >= mins (nutrient minimums)
        Ax <= maxs (nutrient maximums)
        0 <= x <= food_maxs (food bounds)

    Args:
        costs: Cost per gram for each food, shape (n_foods,)
        nutrient_matrix: Nutrients per gram, shape (n_foods, n_nutrients)
        nutrient_mins: Minimum nutrient amounts, shape (n_nutrients,)
        nutrient_maxs: Maximum nutrient amounts, shape (n_nutrients,)
        food_bounds: List of (min, max) grams for each food

    Returns:
        Dict with solution info
    """
    n_nutrients = nutrient_matrix.shape[1]
    start_time = time.time()

    # Build inequality constraints: A_ub @ x <= b_ub
    # For min constraints: -Ax <= -min  (i.e., Ax >= min)
    # For max constraints: Ax <= max
    A_ub_rows = []
    b_ub_rows = []

    for j in range(n_nutrients):
        col = nutrient_matrix[:, j]

        if nutrient_mins[j] > -np.inf:
            A_ub_rows.append(-col)
            b_ub_rows.append(-nutrient_mins[j])

        if nutrient_maxs[j] < np.inf:
            A_ub_rows.append(col)
            b_ub_rows.append(nutrient_maxs[j])

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    result = linprog(
        c=costs,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=food_bounds,
        method="highs",
        options={"presolve": True},
    )

    elapsed = time.time() - start_time

    return {
        "success": result.success,
        "x": result.x if result.success else None,
        "fun": result.fun if result.success else None,
        "message": result.message,
        "iterations": getattr(result, "nit", None),
        "elapsed_seconds": elapsed,
    }


def solve_qp(
    costs: np.ndarray,
    nutrient_matrix: np.ndarray,
    nutrient_mins: np.ndarray,
    nutrient_maxs: np.ndarray,
    food_bounds: list[tuple[float, float]],
    typical_consumption: np.ndarray,
    lambda_cost: float = 1.0,
    lambda_deviation: float = 0.001,
) -> dict[str, Any]:
    """Solve quadratic programming problem using scipy.optimize.minimize with SLSQP.

    Objective: min lambda_cost * c'x + lambda_deviation * ||x - x_bar||^2

    The quadratic penalty encourages solutions that don't deviate too far from
    typical consumption patterns, producing more diverse/palatable meal plans.

    Args:
        costs: Cost per gram for each food, shape (n_foods,)
        nutrient_matrix: Nutrients per gram, shape (n_foods, n_nutrients)
        nutrient_mins: Minimum nutrient amounts, shape (n_nutrients,)
        nutrient_maxs: Maximum nutrient amounts, shape (n_nutrients,)
        food_bounds: List of (min, max) grams for each food
        typical_consumption: Typical daily consumption in grams, shape (n_foods,)
        lambda_cost: Weight for cost term (default 1.0)
        lambda_deviation: Weight for deviation penalty (default 0.001)

    Returns:
        Dict with solution info
    """
    start_time = time.time()

    def objective(x: np.ndarray) -> float:
        cost_term = lambda_cost * np.dot(costs, x)
        deviation_term = lambda_deviation * np.sum((x - typical_consumption) ** 2)
        return cost_term + deviation_term

    def objective_grad(x: np.ndarray) -> np.ndarray:
        return lambda_cost * costs + 2 * lambda_deviation * (x - typical_consumption)

    # Build constraints as list of dicts
    constraints = []

    for j in range(nutrient_matrix.shape[1]):
        col = nutrient_matrix[:, j].copy()

        if nutrient_mins[j] > -np.inf:
            min_val = nutrient_mins[j]
            # Constraint: Ax >= min  =>  Ax - min >= 0
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, c=col, m=min_val: np.dot(c, x) - m,
                    "jac": lambda x, c=col: c,
                }
            )

        if nutrient_maxs[j] < np.inf:
            max_val = nutrient_maxs[j]
            # Constraint: Ax <= max  =>  max - Ax >= 0
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, c=col, m=max_val: m - np.dot(c, x),
                    "jac": lambda x, c=col: -c,
                }
            )

    # Initial guess: midpoint of bounds, clipped toward typical
    x0 = np.array([(b[0] + b[1]) / 2 for b in food_bounds])
    x0 = np.minimum(x0, typical_consumption * 1.5)
    x0 = np.maximum(x0, 0.1)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=objective_grad,
        constraints=constraints,
        bounds=food_bounds,
        options={"maxiter": 1000, "ftol": 1e-9, "disp": False},
    )

    elapsed = time.time() - start_time

    return {
        "success": result.success,
        "x": result.x if result.success else None,
        "fun": result.fun if result.success else None,
        "message": result.message,
        "iterations": getattr(result, "nit", None),
        "elapsed_seconds": elapsed,
    }


def solve_diet_problem(
    request: OptimizationRequest,
    conn: sqlite3.Connection,
) -> OptimizationResult:
    """Main entry point for solving the diet optimization problem.

    Args:
        request: The optimization request specification
        conn: Database connection

    Returns:
        OptimizationResult with solution or error info
    """
    # Build constraint matrices
    builder = ConstraintBuilder(conn, request)
    constraint_data = builder.build()

    # Check if we have any foods to work with
    if not constraint_data["food_ids"]:
        return OptimizationResult(
            success=False,
            status="error",
            message="No eligible foods found. Check that foods have prices and pass tag filters.",
            foods=[],
            total_cost=None,
            nutrients=[],
            solver_info={},
        )

    # Get nutrient names for results
    nutrient_names = NutrientQueries.get_nutrient_names(
        conn, constraint_data["nutrient_ids"]
    )

    # Choose solver based on request
    if request.use_quadratic_penalty:
        # Build typical consumption vector (default 100g per food)
        n_foods = len(constraint_data["food_ids"])
        typical = np.full(n_foods, 100.0)

        result = solve_qp(
            costs=constraint_data["costs"],
            nutrient_matrix=constraint_data["nutrient_matrix"],
            nutrient_mins=constraint_data["nutrient_mins"],
            nutrient_maxs=constraint_data["nutrient_maxs"],
            food_bounds=constraint_data["food_bounds"],
            typical_consumption=typical,
            lambda_cost=request.lambda_cost,
            lambda_deviation=request.lambda_deviation,
        )
    else:
        result = solve_lp(
            costs=constraint_data["costs"],
            nutrient_matrix=constraint_data["nutrient_matrix"],
            nutrient_mins=constraint_data["nutrient_mins"],
            nutrient_maxs=constraint_data["nutrient_maxs"],
            food_bounds=constraint_data["food_bounds"],
        )

    # Handle failed optimization
    if not result["success"]:
        return OptimizationResult(
            success=False,
            status="infeasible",
            message=f"Optimization failed: {result['message']}",
            foods=[],
            total_cost=None,
            nutrients=[],
            solver_info={"elapsed_seconds": result["elapsed_seconds"]},
        )

    # Build food results (only include foods with > 0.1g)
    x = result["x"]
    foods = []
    for i, (fid, desc, grams) in enumerate(
        zip(
            constraint_data["food_ids"],
            constraint_data["food_descriptions"],
            x,
        )
    ):
        if grams > 0.1:
            cost = constraint_data["costs"][i] * grams
            nutrient_amounts = {
                nid: constraint_data["nutrient_matrix"][i, j] * grams
                for j, nid in enumerate(constraint_data["nutrient_ids"])
            }
            foods.append(
                FoodResult(
                    fdc_id=fid,
                    description=desc,
                    grams=grams,
                    cost=cost,
                    nutrients=nutrient_amounts,
                )
            )

    # Sort foods by grams descending
    foods.sort(key=lambda f: -f.grams)

    # Calculate total nutrients
    total_nutrients = constraint_data["nutrient_matrix"].T @ x
    nutrient_results = []
    for j, nid in enumerate(constraint_data["nutrient_ids"]):
        name, unit = nutrient_names.get(nid, (f"Nutrient {nid}", "?"))
        min_c = constraint_data["nutrient_mins"][j]
        max_c = constraint_data["nutrient_maxs"][j]
        amount = total_nutrients[j]

        # Check if constraint is satisfied (with small tolerance)
        satisfied = True
        if min_c > -np.inf and amount < min_c - 0.01:
            satisfied = False
        if max_c < np.inf and amount > max_c + 0.01:
            satisfied = False

        nutrient_results.append(
            NutrientResult(
                nutrient_id=nid,
                name=name,
                unit=unit,
                amount=amount,
                min_constraint=min_c if min_c > -np.inf else None,
                max_constraint=max_c if max_c < np.inf else None,
                satisfied=satisfied,
            )
        )

    total_cost = float(np.dot(constraint_data["costs"], x))

    return OptimizationResult(
        success=True,
        status="optimal",
        message="Optimization successful",
        foods=foods,
        total_cost=total_cost,
        nutrients=nutrient_results,
        solver_info={
            "iterations": result["iterations"],
            "elapsed_seconds": result["elapsed_seconds"],
            "solver": "qp_slsqp" if request.use_quadratic_penalty else "lp_highs",
        },
    )
