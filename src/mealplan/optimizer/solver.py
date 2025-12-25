"""Core LP/QP solver implementation."""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import numpy as np
from qpsolvers import Problem, solve_problem
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, diags

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
    """Solve quadratic programming problem using qpsolvers with Clarabel.

    Objective: min lambda_cost * c'x + lambda_deviation * ||x - x_bar||^2

    The quadratic penalty encourages solutions that don't deviate too far from
    typical consumption patterns, producing more diverse/palatable meal plans.

    Uses Clarabel interior-point solver for high precision and proper KKT multipliers.

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
    n_foods = len(costs)
    n_nutrients = nutrient_matrix.shape[1]

    # Build QP in standard form: min 0.5 * x'Px + q'x
    # Our objective: lambda_cost * c'x + lambda_deviation * ||x - x_bar||^2
    #              = lambda_cost * c'x + lambda_deviation * (x'x - 2*x_bar'x + x_bar'x_bar)
    #              = lambda_deviation * x'Ix + (lambda_cost * c - 2*lambda_deviation*x_bar)'x + const
    # Standard form: 0.5 * x'Px + q'x
    # So: P = 2 * lambda_deviation * I, q = lambda_cost * c - 2 * lambda_deviation * x_bar

    # Use sparse diagonal matrix for P (Clarabel expects sparse CSC format)
    P = diags([2 * lambda_deviation] * n_foods, format="csc")
    q = lambda_cost * costs - 2 * lambda_deviation * typical_consumption

    # Build inequality constraints: G @ x <= h
    # Min constraint: Ax >= min  =>  -Ax <= -min
    # Max constraint: Ax <= max  =>   Ax <= max
    G_rows = []
    h_rows = []
    constraint_map = []  # Track (nutrient_index, "min"|"max") for KKT analysis

    for j in range(n_nutrients):
        col = nutrient_matrix[:, j]

        if nutrient_mins[j] > -np.inf:
            G_rows.append(-col)  # -A @ x <= -min
            h_rows.append(-nutrient_mins[j])
            constraint_map.append((j, "min"))

        if nutrient_maxs[j] < np.inf:
            G_rows.append(col)  # A @ x <= max
            h_rows.append(nutrient_maxs[j])
            constraint_map.append((j, "max"))

    # Convert G to sparse CSC format for Clarabel
    G = csc_matrix(np.array(G_rows)) if G_rows else None
    h = np.array(h_rows) if h_rows else None

    # Extract bounds
    lb = np.array([b[0] for b in food_bounds])
    ub = np.array([b[1] for b in food_bounds])

    # Solve with Clarabel interior-point solver
    problem = Problem(P, q, G, h, lb=lb, ub=ub)
    solution = solve_problem(problem, solver="clarabel")

    elapsed = time.time() - start_time

    success = solution.found if solution is not None else False
    x = solution.x if success else None

    # Compute objective value
    fun = None
    if success and x is not None:
        fun = float(0.5 * x @ P @ x + q @ x)

    ret = {
        "success": success,
        "x": x,
        "fun": fun,
        "message": "Optimization successful" if success else "Optimization failed",
        "iterations": None,  # Clarabel doesn't report iterations in the same way
        "elapsed_seconds": elapsed,
    }

    if return_kkt and success:
        # qpsolvers returns:
        # - z: dual multipliers for inequality constraints (G @ x <= h)
        # - z_box: dual multipliers for box constraints (lb <= x <= ub)
        z = solution.z if solution.z is not None else np.zeros(len(h_rows))
        z_box = solution.z_box if solution.z_box is not None else np.zeros(n_foods)

        # Build constraint values for KKT display
        constraint_values = []
        for i, (nutrient_idx, constraint_type) in enumerate(constraint_map):
            col = nutrient_matrix[:, nutrient_idx]
            nutrient_value = float(np.dot(col, x))

            if constraint_type == "min":
                bound = float(nutrient_mins[nutrient_idx])
                slack = nutrient_value - bound
                # For -Ax <= -min, the multiplier z[i] >= 0 when active
                # This corresponds to the min constraint being tight
                mult = float(z[i]) if z is not None else None
            else:  # "max"
                bound = float(nutrient_maxs[nutrient_idx])
                slack = bound - nutrient_value
                # For Ax <= max, the multiplier z[i] >= 0 when active
                mult = float(z[i]) if z is not None else None

            constraint_values.append({
                "type": constraint_type,
                "nutrient_index": nutrient_idx,
                "bound": bound,
                "value": nutrient_value,
                "slack": slack,
                "multiplier": mult,
            })

        # Compute objective gradient for stationarity check
        # ∇f = ∂/∂x [0.5 x'Px + q'x] = Px + q = 2*lambda_deviation*x + q
        objective_grad = P @ x + q

        ret["kkt"] = {
            "z": z,                      # Inequality constraint multipliers
            "z_box": z_box,              # Bound multipliers (key improvement!)
            "constraint_values": constraint_values,
            "constraint_map": constraint_map,
            "objective_grad": objective_grad,
            "G": G.toarray() if G is not None else None,  # Dense for KKT computation
            "primal_residual": solution.primal_residual() if hasattr(solution, 'primal_residual') else None,
            "dual_residual": solution.dual_residual() if hasattr(solution, 'dual_residual') else None,
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
        solver_type: "lp_highs", "qp_clarabel", or "qp_feasibility"
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
        # QP solver - use qpsolvers with Clarabel (provides z and z_box multipliers)
        constraint_values = kkt_data.get("constraint_values", [])
        z_box = kkt_data.get("z_box")  # Bound multipliers from Clarabel

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

            # Use qpsolvers multiplier (z values for inequality constraints)
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

        # Check food bounds for QP - now we have z_box from Clarabel!
        # z_box[i] < 0: at lower bound (multiplier for lb constraint)
        # z_box[i] = 0: interior (no active bound)
        # z_box[i] > 0: at upper bound (multiplier for ub constraint)
        for i, (lb, ub) in enumerate(food_bounds):
            food_val = float(x[i])
            bound_mult = float(z_box[i]) if z_box is not None else None

            if food_val - lb < tolerance:
                # At lower bound - z_box should be negative (pushing up)
                binding_food_bounds.append(
                    ConstraintKKT(
                        name=food_descriptions[i][:50],
                        constraint_type="food_lower",
                        bound=lb,
                        value=food_val,
                        slack=food_val - lb,
                        multiplier=bound_mult,
                        is_binding=True,
                    )
                )
            if ub - food_val < tolerance:
                # At upper bound - z_box should be positive (pushing down)
                binding_food_bounds.append(
                    ConstraintKKT(
                        name=food_descriptions[i][:50],
                        constraint_type="food_upper",
                        bound=ub,
                        value=food_val,
                        slack=ub - food_val,
                        multiplier=bound_mult,
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
    # Full KKT stationarity: ∇f + G'z + z_box = 0
    stationarity_residual = 0.0
    if solver_type == "lp_highs":
        # Stationarity is automatically satisfied by LP solver
        stationarity_residual = 0.0
    else:
        # For QP with Clarabel, compute full stationarity using all multipliers
        # ∇L = ∇f + G'z + z_box (should equal 0 at optimum)
        obj_grad = kkt_data.get("objective_grad")
        G = kkt_data.get("G")
        z = kkt_data.get("z")
        z_box = kkt_data.get("z_box")

        if obj_grad is not None:
            lagrangian_grad = obj_grad.copy()

            # Add contribution from inequality constraints: G'z
            # Note: G is formulated as G @ x <= h, so for the Lagrangian:
            # L = f(x) + z'(Gx - h), and ∇L_x = ∇f + G'z
            if G is not None and z is not None:
                lagrangian_grad += G.T @ z

            # Add contribution from box constraints: z_box
            # For lb <= x <= ub, the Lagrangian includes bound multipliers
            if z_box is not None:
                lagrangian_grad += z_box

            # Now we can compute stationarity for ALL variables (not just interior)
            # because we have the full multiplier information
            stationarity_residual = float(np.linalg.norm(lagrangian_grad))

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
        solver_type = "qp_clarabel"
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
