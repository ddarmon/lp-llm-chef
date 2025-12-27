"""Sparse solver for limiting the number of foods in solution.

Implements a two-phase heuristic:
1. Phase 1: Run full QP optimization with all eligible foods
2. Phase 2: Select top N foods by contribution, re-optimize with only those

This produces meal-prep friendly solutions with fewer distinct foods
at meaningful portion sizes.
"""

from __future__ import annotations

import sqlite3
from copy import deepcopy

from mealplan.optimizer.models import OptimizationRequest, OptimizationResult
from mealplan.optimizer.solver import solve_diet_problem


def solve_sparse_diet(
    request: OptimizationRequest,
    conn: sqlite3.Connection,
    max_foods: int,
    min_grams_if_included: float = 50.0,
    verbose: bool = False,
) -> OptimizationResult:
    """Two-phase solver for sparse solutions with limited food count.

    Args:
        request: Base optimization request
        conn: Database connection
        max_foods: Maximum number of distinct foods in solution
        min_grams_if_included: Minimum grams for any included food
        verbose: Include KKT analysis in result

    Returns:
        OptimizationResult with at most max_foods foods
    """
    # Phase 1: Full solve to get initial solution
    full_result = solve_diet_problem(request, conn, verbose=False)

    if not full_result.success:
        # If full solve fails, no point in continuing
        return full_result

    # If we already have <= max_foods, just return the result
    if len(full_result.foods) <= max_foods:
        # Re-run with verbose if requested
        if verbose:
            return solve_diet_problem(request, conn, verbose=True)
        return full_result

    # Phase 2: Select top N foods by grams and re-optimize
    # Sort foods by amount (descending) and take top N
    sorted_foods = sorted(full_result.foods, key=lambda f: -f.grams)
    top_foods = sorted_foods[:max_foods]
    top_food_ids = [f.fdc_id for f in top_foods]

    # Create modified request with only these foods
    sparse_request = OptimizationRequest(
        mode=request.mode,
        objective=request.objective,
        calorie_range=request.calorie_range,
        nutrient_constraints=list(request.nutrient_constraints),
        food_constraints=list(request.food_constraints),
        exclude_tags=list(request.exclude_tags),
        include_tags=[],  # Clear include tags since we're using explicit IDs
        max_grams_per_food=request.max_grams_per_food,
        max_foods=max_foods,
        planning_days=request.planning_days,
        use_quadratic_penalty=request.use_quadratic_penalty,
        lambda_cost=request.lambda_cost,
        lambda_deviation=request.lambda_deviation,
        # Use explicit food IDs
        explicit_food_ids=top_food_ids,
    )

    # Run phase 2 optimization
    sparse_result = solve_diet_problem(sparse_request, conn, verbose=verbose)

    if sparse_result.success:
        # Update message to indicate sparse solve
        sparse_result.message = (
            f"Sparse solution with {len(sparse_result.foods)} foods "
            f"(reduced from {len(full_result.foods)} in full solve)"
        )

        # Update solver info
        sparse_result.solver_info["sparse_solver"] = True
        sparse_result.solver_info["full_solve_food_count"] = len(full_result.foods)
        sparse_result.solver_info["max_foods_limit"] = max_foods

    return sparse_result


def solve_iterative_sparse(
    request: OptimizationRequest,
    conn: sqlite3.Connection,
    max_foods: int,
    min_grams_if_included: float = 50.0,
    max_iterations: int = 3,
    verbose: bool = False,
) -> OptimizationResult:
    """Iterative sparse solver that refines the food selection.

    Runs multiple iterations, each time:
    1. Solve with current food pool
    2. Remove foods with < min_grams
    3. Re-solve with remaining foods

    This can produce better solutions than single-pass sparse solve.

    Args:
        request: Base optimization request
        conn: Database connection
        max_foods: Maximum number of distinct foods in solution
        min_grams_if_included: Minimum grams for any included food
        max_iterations: Maximum refinement iterations
        verbose: Include KKT analysis in result

    Returns:
        OptimizationResult with at most max_foods foods
    """
    current_request = deepcopy(request)
    last_result: OptimizationResult | None = None

    for iteration in range(max_iterations):
        # Solve current iteration
        result = solve_diet_problem(current_request, conn, verbose=False)

        if not result.success:
            # If this iteration fails but we have a previous result, use that
            if last_result is not None:
                return last_result
            return result

        # Check if we're done (all foods have >= min_grams and count <= max_foods)
        small_foods = [f for f in result.foods if f.grams < min_grams_if_included]
        if len(result.foods) <= max_foods and not small_foods:
            # Perfect solution
            if verbose:
                return solve_diet_problem(current_request, conn, verbose=True)
            return result

        last_result = result

        # Filter out small foods and excess foods
        valid_foods = [f for f in result.foods if f.grams >= min_grams_if_included]
        valid_foods.sort(key=lambda f: -f.grams)
        valid_foods = valid_foods[:max_foods]

        if not valid_foods:
            # All foods were too small, use the full result
            break

        # Update request for next iteration
        current_request.explicit_food_ids = [f.fdc_id for f in valid_foods]
        current_request.include_tags = []  # Clear since we're using explicit IDs

    # Final solve with verbose if requested
    if verbose and current_request.explicit_food_ids:
        return solve_diet_problem(current_request, conn, verbose=True)

    return last_result if last_result else result


def select_diverse_foods(
    request: OptimizationRequest,
    conn: sqlite3.Connection,
    n_foods: int,
) -> list[int]:
    """Select a diverse subset of foods for sparse optimization.

    Uses a greedy approach to select foods that provide good nutrient coverage.

    Args:
        request: Optimization request (used for constraints)
        conn: Database connection
        n_foods: Number of foods to select

    Returns:
        List of fdc_ids for selected foods
    """
    # Run full optimization first
    result = solve_diet_problem(request, conn, verbose=False)

    if not result.success:
        return []

    # Score foods by their contribution to nutrient targets
    food_scores: dict[int, float] = {}

    for food in result.foods:
        # Base score from grams
        score = food.grams

        # Bonus for foods that contribute to constrained nutrients
        for nutrient_id, amount in food.nutrients.items():
            # Nutrients in the food contribute to the score
            score += amount * 0.1

        food_scores[food.fdc_id] = score

    # Sort by score and take top N
    sorted_foods = sorted(food_scores.items(), key=lambda x: -x[1])
    return [fdc_id for fdc_id, _ in sorted_foods[:n_foods]]
