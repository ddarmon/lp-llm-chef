"""Quantity optimization for template-based meal composition.

This module handles the continuous optimization phase: given selected foods
for each slot, find the optimal quantities that satisfy nutrient constraints.

The key difference from Stigler-style optimization is that we optimize a
MUCH smaller problem (16-20 variables vs 300+) with slot-specific targets.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

from mealplan.optimizer.multiperiod_models import MealType
from mealplan.templates.definitions import get_template_for_patterns
from mealplan.templates.models import (
    DietTemplate,
    OptimizedFood,
    OptimizedMeal,
    SelectedMeal,
    SelectionStrategy,
    TemplateOptimizationRequest,
    TemplateResult,
)
from mealplan.templates.selector import select_foods_for_template

# Nutrient IDs
ENERGY_ID = 1008
PROTEIN_ID = 1003
CARB_ID = 1005
FAT_ID = 1004


def optimize_template(
    conn: sqlite3.Connection,
    request: TemplateOptimizationRequest,
) -> TemplateResult:
    """Run template-based optimization with automatic retry on infeasibility.

    Args:
        conn: Database connection
        request: Optimization request with template and constraints

    Returns:
        TemplateResult with optimized meals.
    """
    template = request.template

    for attempt in range(request.max_retries):
        # Phase 1: Select foods (or use provided selections)
        if request.selections is not None and attempt == 0:
            selections = request.selections
        else:
            selections = select_foods_for_template(
                conn=conn,
                template=template,
                strategy=request.selection_strategy,
                seed=request.seed + attempt if request.seed else None,
                excluded_foods=request.excluded_foods,
                preferred_foods=request.preferred_foods,
            )

        # Phase 2: Optimize quantities
        result = _optimize_quantities(conn, selections, request)

        if result.success:
            result.selection_attempts = attempt + 1
            return result

    # All retries failed
    return TemplateResult(
        success=False,
        meals=[],
        template_name=template.name,
        selection_attempts=request.max_retries,
        message=f"Failed to find feasible solution after {request.max_retries} attempts",
    )


def _optimize_quantities(
    conn: sqlite3.Connection,
    selections: list[SelectedMeal],
    request: TemplateOptimizationRequest,
) -> TemplateResult:
    """Optimize quantities for selected foods.

    This runs a small QP:
        min ||x - target_grams||²
        s.t. daily_cal_min <= sum(calories) <= daily_cal_max
             daily_prot_min <= sum(protein) <= daily_prot_max
             slot_min <= x_i <= slot_max for each food

    Args:
        conn: Database connection
        selections: Selected foods for each meal
        request: Optimization request

    Returns:
        TemplateResult with optimized quantities.
    """
    template = request.template

    # Flatten selections into a list of (meal_idx, selection) pairs
    food_list: list[tuple[int, int, SelectedMeal, any]] = []  # (var_idx, meal_idx, meal, selection)
    var_idx = 0
    for meal_idx, meal in enumerate(selections):
        for selection in meal.selections:
            food_list.append((var_idx, meal_idx, meal, selection))
            var_idx += 1

    n_vars = len(food_list)
    if n_vars == 0:
        return TemplateResult(
            success=False,
            meals=[],
            template_name=template.name,
            message="No foods selected",
        )

    # Get nutrient data for all selected foods
    fdc_ids = [item[3].fdc_id for item in food_list]
    nutrient_data = _get_nutrient_data(conn, fdc_ids)

    # Build QP matrices
    # Objective: min 0.5 * x'Px + q'x where we want to minimize ||x - target||²
    # This expands to: x'x - 2*target'x + target'target
    # So P = 2*I, q = -2*target

    P = lil_matrix((n_vars, n_vars))
    q = np.zeros(n_vars)
    lb = np.zeros(n_vars)  # Lower bounds
    ub = np.zeros(n_vars)  # Upper bounds

    for i, (_, _, _, selection) in enumerate(food_list):
        target = selection.target_grams
        P[i, i] = 2.0  # Coefficient for x²
        q[i] = -2.0 * target  # Coefficient for x
        lb[i] = selection.slot.min_grams
        ub[i] = selection.slot.max_grams

    P_sparse = P.tocsc()

    # Build inequality constraints: G @ x <= h
    # We need: cal_min <= sum(cal) <= cal_max
    #          prot_min <= sum(prot) <= prot_max
    # Rewrite as: -sum(cal) <= -cal_min, sum(cal) <= cal_max, etc.

    G_rows = []
    h_vals = []

    # Daily calorie constraints
    cal_min, cal_max = request.daily_calories
    cal_row = np.array([nutrient_data[fid].get(ENERGY_ID, 0) / 100 for fid in fdc_ids])

    # -sum(cal) <= -cal_min  (i.e., sum(cal) >= cal_min)
    G_rows.append(-cal_row)
    h_vals.append(-cal_min)

    # sum(cal) <= cal_max
    G_rows.append(cal_row)
    h_vals.append(cal_max)

    # Daily protein constraints
    prot_min, prot_max = request.daily_protein
    prot_row = np.array([nutrient_data[fid].get(PROTEIN_ID, 0) / 100 for fid in fdc_ids])

    # -sum(prot) <= -prot_min
    G_rows.append(-prot_row)
    h_vals.append(-prot_min)

    # sum(prot) <= prot_max
    G_rows.append(prot_row)
    h_vals.append(prot_max)

    # Per-meal calorie constraints (if specified in template)
    for meal_idx, meal_template in enumerate(template.meals):
        if meal_template.calorie_range:
            meal_cal_min, meal_cal_max = meal_template.calorie_range
            meal_mask = np.array([1.0 if item[1] == meal_idx else 0.0 for item in food_list])
            meal_cal_row = cal_row * meal_mask

            # -sum(meal_cal) <= -meal_cal_min
            G_rows.append(-meal_cal_row)
            h_vals.append(-meal_cal_min)

            # sum(meal_cal) <= meal_cal_max
            G_rows.append(meal_cal_row)
            h_vals.append(meal_cal_max)

        if meal_template.protein_min:
            meal_mask = np.array([1.0 if item[1] == meal_idx else 0.0 for item in food_list])
            meal_prot_row = prot_row * meal_mask

            # -sum(meal_prot) <= -meal_prot_min
            G_rows.append(-meal_prot_row)
            h_vals.append(-meal_template.protein_min)

    G = np.array(G_rows) if G_rows else np.zeros((0, n_vars))
    h = np.array(h_vals) if h_vals else np.zeros(0)

    # Solve QP
    try:
        import qpsolvers

        solution = qpsolvers.solve_qp(
            P=P_sparse,
            q=q,
            G=csc_matrix(G) if G.size > 0 else None,
            h=h if h.size > 0 else None,
            lb=lb,
            ub=ub,
            solver="clarabel",
        )

        if solution is None:
            return TemplateResult(
                success=False,
                meals=[],
                template_name=template.name,
                message="QP solver returned no solution (infeasible)",
            )

    except Exception as e:
        return TemplateResult(
            success=False,
            meals=[],
            template_name=template.name,
            message=f"QP solver error: {e}",
        )

    # Build result from solution
    meals: list[OptimizedMeal] = []

    for meal_idx, selected_meal in enumerate(selections):
        foods: list[OptimizedFood] = []

        for i, (var_i, m_idx, _, selection) in enumerate(food_list):
            if m_idx != meal_idx:
                continue

            grams = max(0, solution[var_i])  # Ensure non-negative
            if grams < 1:  # Skip very small quantities
                continue

            fdc_id = selection.fdc_id
            nutrients = nutrient_data.get(fdc_id, {})

            foods.append(
                OptimizedFood(
                    fdc_id=fdc_id,
                    description=selection.description,
                    grams=round(grams, 1),
                    calories=round(nutrients.get(ENERGY_ID, 0) * grams / 100, 1),
                    protein=round(nutrients.get(PROTEIN_ID, 0) * grams / 100, 1),
                    carbs=round(nutrients.get(CARB_ID, 0) * grams / 100, 1),
                    fat=round(nutrients.get(FAT_ID, 0) * grams / 100, 1),
                    slot_name=selection.slot.name,
                )
            )

        meals.append(OptimizedMeal(meal_type=selected_meal.meal_type, foods=foods))

    return TemplateResult(
        success=True,
        meals=meals,
        template_name=template.name,
        message="Optimization successful",
    )


def _get_nutrient_data(
    conn: sqlite3.Connection,
    fdc_ids: list[int],
) -> dict[int, dict[int, float]]:
    """Get nutrient data for foods.

    Args:
        conn: Database connection
        fdc_ids: List of FDC IDs

    Returns:
        Dict mapping fdc_id -> {nutrient_id -> amount_per_100g}
    """
    if not fdc_ids:
        return {}

    placeholders = ",".join("?" * len(fdc_ids))
    query = f"""
        SELECT fdc_id, nutrient_id, amount
        FROM food_nutrients
        WHERE fdc_id IN ({placeholders})
        AND nutrient_id IN (?, ?, ?, ?)
    """

    params = fdc_ids + [ENERGY_ID, PROTEIN_ID, CARB_ID, FAT_ID]
    cursor = conn.execute(query, params)

    result: dict[int, dict[int, float]] = {fid: {} for fid in fdc_ids}
    for row in cursor.fetchall():
        fdc_id, nutrient_id, amount = row
        result[fdc_id][nutrient_id] = amount or 0

    return result


# =============================================================================
# High-Level API
# =============================================================================

def run_template_optimization(
    conn: sqlite3.Connection,
    patterns: list[str],
    daily_calories: tuple[float, float] = (1800, 2200),
    daily_protein: tuple[float, float] = (150, 200),
    strategy: SelectionStrategy = SelectionStrategy.RANDOM,
    seed: Optional[int] = None,
    excluded_foods: Optional[set[int]] = None,
    max_retries: int = 5,
) -> TemplateResult:
    """Run template-based optimization for given dietary patterns.

    This is the main entry point for template-based meal planning.

    Args:
        conn: Database connection
        patterns: List of dietary patterns (e.g., ["pescatarian", "slow_carb"])
        daily_calories: (min, max) daily calorie target
        daily_protein: (min, max) daily protein target
        strategy: Food selection strategy
        seed: Random seed for reproducibility
        excluded_foods: FDC IDs to exclude
        max_retries: Max re-selections on infeasibility

    Returns:
        TemplateResult with optimized meals.

    Example:
        >>> result = run_template_optimization(
        ...     conn,
        ...     patterns=["pescatarian", "slow_carb"],
        ...     daily_calories=(1800, 2200),
        ...     daily_protein=(150, 185),
        ... )
        >>> if result.success:
        ...     for meal in result.meals:
        ...         print(f"{meal.meal_type.value}: {meal.total_calories:.0f} kcal")
    """
    # Get template for patterns
    template = get_template_for_patterns(patterns)
    if template is None:
        return TemplateResult(
            success=False,
            meals=[],
            template_name="unknown",
            message=f"No template available for patterns: {patterns}",
        )

    # Build request
    request = TemplateOptimizationRequest(
        template=template,
        daily_calories=daily_calories,
        daily_protein=daily_protein,
        selection_strategy=strategy,
        seed=seed,
        excluded_foods=excluded_foods or set(),
        max_retries=max_retries,
    )

    return optimize_template(conn, request)
