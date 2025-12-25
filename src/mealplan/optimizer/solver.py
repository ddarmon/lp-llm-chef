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
    ConstraintKKT,
    FoodResult,
    KKTAnalysis,
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
    return_kkt: bool = False,
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
        return_kkt: If True, include KKT data in the result

    Returns:
        Dict with solution info (and KKT data if return_kkt=True)
    """
    n_nutrients = nutrient_matrix.shape[1]
    start_time = time.time()

    # Build inequality constraints: A_ub @ x <= b_ub
    # For min constraints: -Ax <= -min  (i.e., Ax >= min)
    # For max constraints: Ax <= max
    A_ub_rows = []
    b_ub_rows = []
    # Track which constraint each row corresponds to for KKT analysis
    constraint_map = []  # List of (nutrient_index, "min"|"max")

    for j in range(n_nutrients):
        col = nutrient_matrix[:, j]

        if nutrient_mins[j] > -np.inf:
            A_ub_rows.append(-col)
            b_ub_rows.append(-nutrient_mins[j])
            constraint_map.append((j, "min"))

        if nutrient_maxs[j] < np.inf:
            A_ub_rows.append(col)
            b_ub_rows.append(nutrient_maxs[j])
            constraint_map.append((j, "max"))

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

    ret = {
        "success": result.success,
        "x": result.x if result.success else None,
        "fun": result.fun if result.success else None,
        "message": result.message,
        "iterations": getattr(result, "nit", None),
        "elapsed_seconds": elapsed,
    }

    if return_kkt and result.success:
        ret["kkt"] = {
            "slack": getattr(result, "slack", None),
            "ineqlin_marginals": (
                result.ineqlin.marginals if hasattr(result, "ineqlin") else None
            ),
            "lower_marginals": (
                result.lower.marginals if hasattr(result, "lower") else None
            ),
            "upper_marginals": (
                result.upper.marginals if hasattr(result, "upper") else None
            ),
            "constraint_map": constraint_map,
        }

    return ret


def solve_qp(
    costs: np.ndarray,
    nutrient_matrix: np.ndarray,
    nutrient_mins: np.ndarray,
    nutrient_maxs: np.ndarray,
    food_bounds: list[tuple[float, float]],
    typical_consumption: np.ndarray,
    lambda_cost: float = 1.0,
    lambda_deviation: float = 0.001,
    return_kkt: bool = False,
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
        return_kkt: If True, include KKT data in the result

    Returns:
        Dict with solution info (and KKT data if return_kkt=True)
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
    # Track constraint mapping for KKT analysis
    constraint_map = []  # List of (nutrient_index, "min"|"max")

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
            constraint_map.append((j, "min"))

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
            constraint_map.append((j, "max"))

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

    ret = {
        "success": result.success,
        "x": result.x if result.success else None,
        "fun": result.fun if result.success else None,
        "message": result.message,
        "iterations": getattr(result, "nit", None),
        "elapsed_seconds": elapsed,
    }

    if return_kkt and result.success:
        x = result.x
        # scipy 1.16+ returns KKT multipliers in result.multipliers
        # Format: m[:meq] for equality, m[meq:] for inequality constraints
        # We have no equality constraints, so all are inequality
        multipliers = getattr(result, "multipliers", None)

        # Compute constraint values at solution for slack calculation
        constraint_values = []
        multiplier_idx = 0
        for j in range(nutrient_matrix.shape[1]):
            col = nutrient_matrix[:, j]
            nutrient_value = float(np.dot(col, x))

            if nutrient_mins[j] > -np.inf:
                mult = None
                if multipliers is not None and multiplier_idx < len(multipliers):
                    mult = float(multipliers[multiplier_idx])
                    multiplier_idx += 1
                constraint_values.append({
                    "type": "min",
                    "nutrient_index": j,
                    "bound": float(nutrient_mins[j]),
                    "value": nutrient_value,
                    "slack": nutrient_value - nutrient_mins[j],
                    "multiplier": mult,
                })
            if nutrient_maxs[j] < np.inf:
                mult = None
                if multipliers is not None and multiplier_idx < len(multipliers):
                    mult = float(multipliers[multiplier_idx])
                    multiplier_idx += 1
                constraint_values.append({
                    "type": "max",
                    "nutrient_index": j,
                    "bound": float(nutrient_maxs[j]),
                    "value": nutrient_value,
                    "slack": nutrient_maxs[j] - nutrient_value,
                    "multiplier": mult,
                })

        ret["kkt"] = {
            "multipliers": multipliers,
            "constraint_values": constraint_values,
            "constraint_map": constraint_map,
            "objective_grad": objective_grad(x),
        }

    return ret


def build_kkt_analysis(
    solver_result: dict[str, Any],
    solver_type: str,
    constraint_data: dict[str, Any],
    nutrient_names: dict[int, tuple[str, str]],
    x: np.ndarray,
    food_bounds: list[tuple[float, float]],
    tolerance: float = 1e-6,
) -> KKTAnalysis:
    """Build KKT analysis from solver results.

    Args:
        solver_result: Result dict from solve_lp or solve_qp with kkt data
        solver_type: "lp_highs", "qp_slsqp", or "qp_feasibility"
        constraint_data: The constraint_data dict from ConstraintBuilder
        nutrient_names: Mapping of nutrient_id -> (name, unit)
        x: Solution vector
        food_bounds: List of (min, max) grams for each food
        tolerance: Threshold for considering a constraint binding

    Returns:
        KKTAnalysis with categorized constraints and KKT verification
    """
    kkt_data = solver_result.get("kkt", {})
    nutrient_ids = constraint_data["nutrient_ids"]
    nutrient_matrix = constraint_data["nutrient_matrix"]
    nutrient_mins = constraint_data["nutrient_mins"]
    nutrient_maxs = constraint_data["nutrient_maxs"]
    food_descriptions = constraint_data["food_descriptions"]

    nutrient_constraints: list[ConstraintKKT] = []
    binding_food_bounds: list[ConstraintKKT] = []

    # For LP, we have direct access to slack and multipliers
    if solver_type == "lp_highs":
        slack = kkt_data.get("slack")
        marginals = kkt_data.get("ineqlin_marginals")
        constraint_map = kkt_data.get("constraint_map", [])
        lower_marginals = kkt_data.get("lower_marginals")
        upper_marginals = kkt_data.get("upper_marginals")

        # Process nutrient constraints
        for i, (nutrient_idx, constraint_type) in enumerate(constraint_map):
            nid = nutrient_ids[nutrient_idx]
            name, unit = nutrient_names.get(nid, (f"Nutrient {nid}", "?"))
            col = nutrient_matrix[:, nutrient_idx]
            nutrient_value = float(np.dot(col, x))

            if constraint_type == "min":
                bound = nutrient_mins[nutrient_idx]
                slack_val = nutrient_value - bound
                display_name = f"{name} (min)"
            else:
                bound = nutrient_maxs[nutrient_idx]
                slack_val = bound - nutrient_value
                display_name = f"{name} (max)"

            multiplier = float(marginals[i]) if marginals is not None else None
            is_binding = abs(slack_val) < tolerance

            nutrient_constraints.append(
                ConstraintKKT(
                    name=display_name,
                    constraint_type=f"nutrient_{constraint_type}",
                    bound=float(bound),
                    value=nutrient_value,
                    slack=slack_val,
                    multiplier=multiplier,
                    is_binding=is_binding,
                )
            )

        # Process food bounds
        if lower_marginals is not None and upper_marginals is not None:
            for i, (lb, ub) in enumerate(food_bounds):
                food_val = float(x[i])
                # Check lower bound
                if food_val - lb < tolerance:
                    binding_food_bounds.append(
                        ConstraintKKT(
                            name=food_descriptions[i][:50],
                            constraint_type="food_lower",
                            bound=lb,
                            value=food_val,
                            slack=food_val - lb,
                            multiplier=float(lower_marginals[i]),
                            is_binding=True,
                        )
                    )
                # Check upper bound
                if ub - food_val < tolerance:
                    binding_food_bounds.append(
                        ConstraintKKT(
                            name=food_descriptions[i][:50],
                            constraint_type="food_upper",
                            bound=ub,
                            value=food_val,
                            slack=ub - food_val,
                            multiplier=float(upper_marginals[i]),
                            is_binding=True,
                        )
                    )

    else:
        # QP solver - use scipy 1.16+ native multipliers
        constraint_values = kkt_data.get("constraint_values", [])

        for cv in constraint_values:
            nutrient_idx = cv["nutrient_index"]
            nid = nutrient_ids[nutrient_idx]
            name, unit = nutrient_names.get(nid, (f"Nutrient {nid}", "?"))
            constraint_type = cv["type"]

            if constraint_type == "min":
                display_name = f"{name} (min)"
            else:
                display_name = f"{name} (max)"

            is_binding = abs(cv["slack"]) < tolerance

            # Use scipy's native multiplier
            multiplier = cv.get("multiplier")

            nutrient_constraints.append(
                ConstraintKKT(
                    name=display_name,
                    constraint_type=f"nutrient_{constraint_type}",
                    bound=cv["bound"],
                    value=cv["value"],
                    slack=cv["slack"],
                    multiplier=multiplier,
                    is_binding=is_binding,
                )
            )

        # Check food bounds for QP (bound multipliers not returned by scipy)
        for i, (lb, ub) in enumerate(food_bounds):
            food_val = float(x[i])
            if food_val - lb < tolerance:
                binding_food_bounds.append(
                    ConstraintKKT(
                        name=food_descriptions[i][:50],
                        constraint_type="food_lower",
                        bound=lb,
                        value=food_val,
                        slack=food_val - lb,
                        multiplier=None,
                        is_binding=True,
                    )
                )
            if ub - food_val < tolerance:
                binding_food_bounds.append(
                    ConstraintKKT(
                        name=food_descriptions[i][:50],
                        constraint_type="food_upper",
                        bound=ub,
                        value=food_val,
                        slack=ub - food_val,
                        multiplier=None,
                        is_binding=True,
                    )
                )

    # Check KKT conditions
    # 1. Primal feasibility: all slacks >= 0 (within tolerance)
    primal_feasible = all(c.slack >= -tolerance for c in nutrient_constraints)

    # 2. Dual feasibility: all multipliers >= 0 for inequality constraints
    # Note: scipy returns multipliers with sign convention matching the constraint formulation
    dual_feasible = True
    for c in nutrient_constraints:
        if c.multiplier is not None and c.multiplier < -tolerance:
            dual_feasible = False
            break

    # 3. Complementary slackness: slack * multiplier ≈ 0
    complementary_slackness_satisfied = True
    for c in nutrient_constraints:
        if c.multiplier is not None:
            product = abs(c.slack * c.multiplier)
            if product > tolerance * 100:  # Use looser tolerance for products
                complementary_slackness_satisfied = False
                break

    # 4. Stationarity: compute ||∇L(x*)||
    # For interior variables (not at bounds): ∇L_i = 0
    # For variables at bounds: ∇L_i has appropriate sign (handled by bound multipliers)
    stationarity_residual = 0.0
    if solver_type == "lp_highs":
        # Stationarity is automatically satisfied by LP solver
        stationarity_residual = 0.0
    else:
        # For QP, compute ∇L = ∇f + Σ λᵢ∇gᵢ using scipy's multipliers
        obj_grad = kkt_data.get("objective_grad")
        constraint_values = kkt_data.get("constraint_values", [])
        if obj_grad is not None:
            lagrangian_grad = obj_grad.copy()
            # Add contribution from each constraint
            for cv in constraint_values:
                mult = cv.get("multiplier")
                if mult is not None and mult != 0:
                    nutrient_idx = cv["nutrient_index"]
                    col = nutrient_matrix[:, nutrient_idx]
                    if cv["type"] == "min":
                        # ∇g = A for (Ax - min >= 0)
                        lagrangian_grad += mult * col
                    else:
                        # ∇g = -A for (max - Ax >= 0)
                        lagrangian_grad += mult * (-col)

            # Only check stationarity for interior variables (not at bounds)
            # Variables at bounds have non-zero bound multipliers that balance the gradient
            interior_mask = np.ones(len(x), dtype=bool)
            for i, (lb, ub) in enumerate(food_bounds):
                if x[i] - lb < tolerance or ub - x[i] < tolerance:
                    interior_mask[i] = False

            n_interior = np.sum(interior_mask)
            if n_interior > 0:
                # Compute RMS (root mean square) of gradient for interior variables
                # This gives a per-variable measure that's comparable across problem sizes
                interior_grad = lagrangian_grad[interior_mask]
                stationarity_residual = float(np.sqrt(np.mean(interior_grad**2)))
            else:
                # All variables at bounds - stationarity is satisfied via bound multipliers
                stationarity_residual = 0.0

    return KKTAnalysis(
        solver_type=solver_type,
        primal_feasible=primal_feasible,
        dual_feasible=dual_feasible,
        complementary_slackness_satisfied=complementary_slackness_satisfied,
        stationarity_residual=stationarity_residual,
        nutrient_constraints=nutrient_constraints,
        binding_food_bounds=binding_food_bounds,
    )


def solve_diet_problem(
    request: OptimizationRequest,
    conn: sqlite3.Connection,
    verbose: bool = False,
) -> OptimizationResult:
    """Main entry point for solving the diet optimization problem.

    Args:
        request: The optimization request specification
        conn: Database connection
        verbose: If True, include KKT analysis in the result

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

    # Choose solver based on request mode
    n_foods = len(constraint_data["food_ids"])
    typical = np.full(n_foods, 100.0)

    if request.mode == "feasibility":
        # Feasibility mode: QP with λ_cost=0, just minimize deviation from typical
        result = solve_qp(
            costs=constraint_data["costs"],
            nutrient_matrix=constraint_data["nutrient_matrix"],
            nutrient_mins=constraint_data["nutrient_mins"],
            nutrient_maxs=constraint_data["nutrient_maxs"],
            food_bounds=constraint_data["food_bounds"],
            typical_consumption=typical,
            lambda_cost=0.0,
            lambda_deviation=request.lambda_deviation,
            return_kkt=verbose,
        )
        solver_type = "qp_feasibility"
    elif request.use_quadratic_penalty:
        # Cost minimization with diversity penalty
        result = solve_qp(
            costs=constraint_data["costs"],
            nutrient_matrix=constraint_data["nutrient_matrix"],
            nutrient_mins=constraint_data["nutrient_mins"],
            nutrient_maxs=constraint_data["nutrient_maxs"],
            food_bounds=constraint_data["food_bounds"],
            typical_consumption=typical,
            lambda_cost=request.lambda_cost,
            lambda_deviation=request.lambda_deviation,
            return_kkt=verbose,
        )
        solver_type = "qp_slsqp"
    else:
        # Pure LP cost minimization
        result = solve_lp(
            costs=constraint_data["costs"],
            nutrient_matrix=constraint_data["nutrient_matrix"],
            nutrient_mins=constraint_data["nutrient_mins"],
            nutrient_maxs=constraint_data["nutrient_maxs"],
            food_bounds=constraint_data["food_bounds"],
            return_kkt=verbose,
        )
        solver_type = "lp_highs"

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

    # Build KKT analysis if verbose mode is enabled
    kkt_analysis = None
    if verbose and "kkt" in result:
        kkt_analysis = build_kkt_analysis(
            solver_result=result,
            solver_type=solver_type,
            constraint_data=constraint_data,
            nutrient_names=nutrient_names,
            x=x,
            food_bounds=constraint_data["food_bounds"],
        )

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
            "solver": solver_type,
        },
        kkt_analysis=kkt_analysis,
    )
