"""Multi-period (per-meal) diet optimization solver.

This module provides the main entry point for solving diet optimization
with per-meal constraints. It builds on the existing QP infrastructure
but with expanded variable structure.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional

import numpy as np
from qpsolvers import Problem, solve_problem

from llmn.db.queries import NutrientQueries
from llmn.optimizer.multiperiod_constraints import MultiPeriodConstraintBuilder
from llmn.optimizer.multiperiod_diagnosis import diagnose_multiperiod_infeasibility
from llmn.optimizer.multiperiod_models import (
    InfeasibilityDiagnosis,
    MealFoodResult,
    MealResult,
    MultiPeriodRequest,
    MultiPeriodResult,
)


# Standard nutrient IDs
ENERGY_ID = 1008  # Calories
PROTEIN_ID = 1003
CARB_ID = 1005
FAT_ID = 1004


def solve_multiperiod_diet(
    request: MultiPeriodRequest,
    conn: sqlite3.Connection,
    verbose: bool = False,
) -> MultiPeriodResult:
    """Solve multi-period (per-meal) diet optimization.

    This function optimizes food allocation across multiple meals (breakfast,
    lunch, dinner, snack) with per-meal calorie/nutrient constraints, daily
    linking constraints, and optional equi-calorie constraints.

    The problem is formulated as a QP:
        min  λ_cost * c'x + λ_dev * ||x - x̄||²
        s.t. Per-meal calorie bounds
             Per-meal nutrient bounds
             Daily nutrient bounds (linking constraints)
             Equi-calorie constraints (linearized)
             Food-meal affinity (via upper bounds)
             0 ≤ x ≤ max_grams

    Args:
        request: Multi-period optimization request
        conn: Database connection
        verbose: If True, include detailed solver info and KKT analysis

    Returns:
        MultiPeriodResult with per-meal food allocations and totals
    """
    start_time = time.time()

    # Build constraint matrices
    builder = MultiPeriodConstraintBuilder(conn, request)
    data = builder.build()

    # Check if we have any foods to work with
    if data["n_foods"] == 0:
        return MultiPeriodResult(
            success=False,
            status="error",
            message="No eligible foods found. Check that foods pass tag filters.",
            meals=[],
            daily_totals={},
            solver_info={"elapsed_seconds": time.time() - start_time},
            infeasibility_diagnosis=None,
        )

    # Solve QP using Clarabel interior-point solver
    problem = Problem(
        P=data["P"],
        q=data["q"],
        G=data["G"],
        h=data["h"],
        lb=data["lb"],
        ub=data["ub"],
    )

    solution = solve_problem(problem, solver="clarabel")
    elapsed = time.time() - start_time

    # Handle infeasibility
    if not solution.found:
        # Run full IIS infeasibility diagnosis
        diagnosis = diagnose_multiperiod_infeasibility(request, conn)

        return MultiPeriodResult(
            success=False,
            status="infeasible",
            message="No feasible solution found for multi-period constraints",
            meals=[],
            daily_totals={},
            solver_info={
                "solver": "qp_clarabel_multiperiod",
                "elapsed_seconds": elapsed,
                "n_foods": data["n_foods"],
                "n_meals": data["n_meals"],
                "n_vars": data["n_vars"],
                "n_constraints": len(data["h"]) if len(data["h"]) else 0,
            },
            infeasibility_diagnosis=diagnosis,
        )

    # Parse solution vector back to meal structure
    x = solution.x
    meals = _parse_solution_to_meals(x, data, builder, conn)

    # Calculate daily totals
    daily_totals = _calculate_daily_totals(meals)

    # Build solver info
    solver_info = {
        "solver": "qp_clarabel_multiperiod",
        "elapsed_seconds": elapsed,
        "n_foods": data["n_foods"],
        "n_meals": data["n_meals"],
        "n_vars": data["n_vars"],
        "n_constraints": len(data["h"]) if len(data["h"]) else 0,
        "total_eligible_foods": data["total_eligible_foods"],
        "was_sampled": data["was_sampled"],
    }

    if verbose:
        solver_info["z"] = solution.z.tolist() if solution.z is not None else None
        solver_info["z_box"] = (
            solution.z_box.tolist() if solution.z_box is not None else None
        )
        solver_info["constraint_info"] = data["constraint_info"]

    # Check for data quality warnings
    if data["data_quality_warnings"]:
        solver_info["data_quality_warnings"] = data["data_quality_warnings"]

    return MultiPeriodResult(
        success=True,
        status="optimal",
        message="Multi-period optimization successful",
        meals=meals,
        daily_totals=daily_totals,
        solver_info=solver_info,
        infeasibility_diagnosis=None,
    )


def _parse_solution_to_meals(
    x: np.ndarray,
    data: dict,
    builder: MultiPeriodConstraintBuilder,
    conn: sqlite3.Connection,
) -> list[MealResult]:
    """Parse the flat solution vector back to MealResult objects.

    Args:
        x: Solution vector of shape (n_vars,)
        data: Constraint data dict from builder
        builder: The constraint builder (for var_index mapping)
        conn: Database connection

    Returns:
        List of MealResult objects, one per meal
    """
    n_foods = data["n_foods"]
    n_meals = data["n_meals"]
    nutrient_ids = data["nutrient_ids"]
    nutrient_matrix = data["nutrient_matrix"]
    food_ids = data["food_ids"]
    food_descriptions = data["food_descriptions"]
    costs = data["costs"]
    meal_configs = data["meal_configs"]

    # Get indices for key nutrients
    cal_idx = nutrient_ids.index(ENERGY_ID) if ENERGY_ID in nutrient_ids else None
    prot_idx = nutrient_ids.index(PROTEIN_ID) if PROTEIN_ID in nutrient_ids else None
    carb_idx = nutrient_ids.index(CARB_ID) if CARB_ID in nutrient_ids else None
    fat_idx = nutrient_ids.index(FAT_ID) if FAT_ID in nutrient_ids else None

    meals = []
    for m, meal_config in enumerate(meal_configs):
        meal_foods = []
        total_cal = 0.0
        total_prot = 0.0
        total_carb = 0.0
        total_fat = 0.0
        total_cost = 0.0

        for i in range(n_foods):
            var_idx = builder.var_index(i, m)
            grams = x[var_idx]

            if grams > 0.1:  # Threshold for inclusion
                # Calculate nutrients for this amount
                nutrients = {}
                for j, nid in enumerate(nutrient_ids):
                    amount = nutrient_matrix[i, j] * grams
                    nutrients[nid] = amount

                cost = costs[i] * grams

                meal_foods.append(
                    MealFoodResult(
                        fdc_id=food_ids[i],
                        description=food_descriptions[i],
                        grams=grams,
                        cost=cost,
                        nutrients=nutrients,
                    )
                )

                # Accumulate totals
                if cal_idx is not None:
                    total_cal += nutrient_matrix[i, cal_idx] * grams
                if prot_idx is not None:
                    total_prot += nutrient_matrix[i, prot_idx] * grams
                if carb_idx is not None:
                    total_carb += nutrient_matrix[i, carb_idx] * grams
                if fat_idx is not None:
                    total_fat += nutrient_matrix[i, fat_idx] * grams
                total_cost += cost

        # Sort foods by grams descending
        meal_foods.sort(key=lambda f: -f.grams)

        meals.append(
            MealResult(
                meal_type=meal_config.meal_type,
                foods=meal_foods,
                total_calories=total_cal,
                total_protein=total_prot,
                total_carbs=total_carb,
                total_fat=total_fat,
                total_cost=total_cost,
            )
        )

    return meals


def _calculate_daily_totals(meals: list[MealResult]) -> dict[str, float]:
    """Calculate daily totals from meal results.

    Args:
        meals: List of MealResult objects

    Returns:
        Dict with calories, protein, carbs, fat, cost
    """
    return {
        "calories": sum(m.total_calories for m in meals),
        "protein": sum(m.total_protein for m in meals),
        "carbs": sum(m.total_carbs for m in meals),
        "fat": sum(m.total_fat for m in meals),
        "cost": sum(m.total_cost for m in meals),
    }
