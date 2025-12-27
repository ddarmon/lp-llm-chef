"""Infeasibility diagnosis for multi-period optimization.

This module provides IIS-like (Irreducible Infeasible Subset) analysis
to identify the minimal set of conflicting constraints when multi-period
optimization fails.

The algorithm uses a constraint deletion approach:
1. Start with the full constraint set
2. Iteratively remove constraints and re-solve
3. Track which constraints are necessary for infeasibility
4. Generate human-readable suggestions for relaxation
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from qpsolvers import Problem, solve_problem
from scipy.sparse import csc_matrix

from mealplan.optimizer.multiperiod_constraints import MultiPeriodConstraintBuilder
from mealplan.optimizer.multiperiod_models import (
    InfeasibilityDiagnosis,
    MealType,
    MultiPeriodRequest,
)


@dataclass
class ConstraintGroup:
    """A logical group of constraints for diagnosis purposes."""

    name: str  # Human-readable name
    row_indices: list[int]  # Indices into G matrix
    constraint_type: str  # "per_meal_cal", "per_meal_nutrient", "daily", "equicalorie"
    meal_type: Optional[MealType] = None
    nutrient_id: Optional[int] = None
    is_min: bool = True  # True for min constraint, False for max


def diagnose_multiperiod_infeasibility(
    request: MultiPeriodRequest,
    conn: sqlite3.Connection,
    max_iterations: int = 50,
) -> InfeasibilityDiagnosis:
    """Diagnose why multi-period optimization failed.

    Uses a constraint deletion algorithm to identify the minimal set of
    constraints that conflict. This is more computationally expensive than
    the basic heuristic but provides much more specific guidance.

    Args:
        request: The original multi-period request
        conn: Database connection
        max_iterations: Maximum iterations for IIS search

    Returns:
        InfeasibilityDiagnosis with detailed conflict analysis
    """
    # Build constraint matrices
    builder = MultiPeriodConstraintBuilder(conn, request)
    data = builder.build()

    if data["n_foods"] == 0:
        return InfeasibilityDiagnosis(
            infeasible=True,
            conflicting_constraints=["no_eligible_foods"],
            suggested_relaxations=[
                "No eligible foods found. Relax tag filters or add more staple foods."
            ],
            analysis_method="food_check",
        )

    # Group constraints by type
    constraint_groups = _group_constraints(data["constraint_info"], request)

    # Run IIS-like analysis
    iis_constraints, iterations = _find_iis(
        P=data["P"],
        q=data["q"],
        G=data["G"],
        h=data["h"],
        lb=data["lb"],
        ub=data["ub"],
        constraint_groups=constraint_groups,
        max_iterations=max_iterations,
    )

    # Generate human-readable suggestions
    suggestions = _generate_suggestions(iis_constraints, request, data)

    return InfeasibilityDiagnosis(
        infeasible=True,
        conflicting_constraints=[c.name for c in iis_constraints],
        suggested_relaxations=suggestions,
        analysis_method=f"iis_deletion_{iterations}_iterations",
    )


def _group_constraints(
    constraint_info: list[tuple[str, float]],
    request: MultiPeriodRequest,
) -> list[ConstraintGroup]:
    """Group constraint rows by logical type for easier analysis.

    Args:
        constraint_info: List of (name, bound) for each constraint row
        request: The original request

    Returns:
        List of ConstraintGroup objects
    """
    groups: list[ConstraintGroup] = []

    for i, (name, bound) in enumerate(constraint_info):
        # Parse constraint name to determine type
        if name.startswith("daily_cal_"):
            is_min = name.endswith("_min")
            groups.append(
                ConstraintGroup(
                    name=f"Daily calories {'minimum' if is_min else 'maximum'}",
                    row_indices=[i],
                    constraint_type="daily_cal",
                    is_min=is_min,
                )
            )
        elif name.startswith("daily_n"):
            # Format: daily_n{nutrient_id}_{min|max}
            parts = name.split("_")
            nutrient_id = int(parts[1][1:])  # Remove 'n' prefix
            is_min = parts[2] == "min"
            groups.append(
                ConstraintGroup(
                    name=f"Daily nutrient {nutrient_id} {'minimum' if is_min else 'maximum'}",
                    row_indices=[i],
                    constraint_type="daily_nutrient",
                    nutrient_id=nutrient_id,
                    is_min=is_min,
                )
            )
        elif "_cal_" in name and not name.startswith("daily"):
            # Format: {meal}_cal_{min|max}
            parts = name.split("_")
            meal_name = parts[0]
            is_min = parts[2] == "min"
            try:
                meal_type = MealType(meal_name)
            except ValueError:
                meal_type = None
            groups.append(
                ConstraintGroup(
                    name=f"{meal_name.title()} calories {'minimum' if is_min else 'maximum'}",
                    row_indices=[i],
                    constraint_type="per_meal_cal",
                    meal_type=meal_type,
                    is_min=is_min,
                )
            )
        elif "_n" in name and not name.startswith("daily"):
            # Format: {meal}_n{nutrient_id}_{min|max}
            parts = name.split("_")
            meal_name = parts[0]
            nutrient_id = int(parts[1][1:])
            is_min = parts[2] == "min"
            try:
                meal_type = MealType(meal_name)
            except ValueError:
                meal_type = None
            groups.append(
                ConstraintGroup(
                    name=f"{meal_name.title()} nutrient {nutrient_id} {'minimum' if is_min else 'maximum'}",
                    row_indices=[i],
                    constraint_type="per_meal_nutrient",
                    meal_type=meal_type,
                    nutrient_id=nutrient_id,
                    is_min=is_min,
                )
            )
        elif name.startswith("equical_"):
            # Format: equical_{meal_a}_{meal_b}_{pos|neg}
            parts = name.split("_")
            meal_a = parts[1]
            meal_b = parts[2]
            direction = parts[3]
            groups.append(
                ConstraintGroup(
                    name=f"Equi-calorie {meal_a}/{meal_b} ({direction})",
                    row_indices=[i],
                    constraint_type="equicalorie",
                )
            )

    return groups


def _find_iis(
    P: csc_matrix,
    q: np.ndarray,
    G: csc_matrix,
    h: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    constraint_groups: list[ConstraintGroup],
    max_iterations: int,
) -> tuple[list[ConstraintGroup], int]:
    """Find an Irreducible Infeasible Subset using constraint deletion.

    Algorithm:
    1. Verify problem is infeasible with all constraints
    2. For each constraint group:
       a. Remove the group's rows from G and h
       b. Re-solve
       c. If now feasible, the removed constraint is necessary for infeasibility
       d. If still infeasible, the constraint might be redundant

    Args:
        P, q, G, h, lb, ub: QP problem data
        constraint_groups: Logical groupings of constraints
        max_iterations: Maximum iterations

    Returns:
        Tuple of (list of constraint groups in IIS, number of iterations)
    """
    # First verify the full problem is infeasible
    if _is_feasible(P, q, G, h, lb, ub):
        # Problem is actually feasible - shouldn't happen
        return [], 0

    # Track which constraints are in the IIS
    iis_candidates = list(constraint_groups)
    iis_confirmed: list[ConstraintGroup] = []

    iterations = 0
    for group in constraint_groups:
        if iterations >= max_iterations:
            break
        iterations += 1

        # Remove this constraint group and test feasibility
        remaining_groups = [g for g in iis_candidates if g != group]
        remaining_indices = []
        for g in remaining_groups:
            remaining_indices.extend(g.row_indices)

        if not remaining_indices:
            # No constraints left - this constraint is necessary
            iis_confirmed.append(group)
            continue

        # Build reduced problem
        G_reduced = G[remaining_indices, :]
        h_reduced = h[remaining_indices]

        if _is_feasible(P, q, G_reduced, h_reduced, lb, ub):
            # Removing this constraint makes problem feasible
            # So this constraint is necessary for infeasibility
            iis_confirmed.append(group)
        # else: constraint is not necessary for infeasibility, skip it

    return iis_confirmed, iterations


def _is_feasible(
    P: csc_matrix,
    q: np.ndarray,
    G: Optional[csc_matrix],
    h: Optional[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
) -> bool:
    """Test if a QP problem is feasible.

    Args:
        P, q, G, h, lb, ub: QP problem data

    Returns:
        True if a feasible solution exists
    """
    # Handle empty constraint matrix
    if G is None or G.shape[0] == 0:
        G_use = None
        h_use = None
    else:
        G_use = G if isinstance(G, csc_matrix) else csc_matrix(G)
        h_use = h

    problem = Problem(P=P, q=q, G=G_use, h=h_use, lb=lb, ub=ub)

    try:
        solution = solve_problem(problem, solver="clarabel")
        return solution.found
    except Exception:
        return False


def _generate_suggestions(
    iis_constraints: list[ConstraintGroup],
    request: MultiPeriodRequest,
    data: dict[str, Any],
) -> list[str]:
    """Generate human-readable relaxation suggestions.

    Args:
        iis_constraints: Constraint groups in the IIS
        request: Original request
        data: Constraint data from builder

    Returns:
        List of actionable suggestions
    """
    suggestions: list[str] = []

    # Analyze by constraint type
    per_meal_cal = [c for c in iis_constraints if c.constraint_type == "per_meal_cal"]
    daily_cal = [c for c in iis_constraints if c.constraint_type == "daily_cal"]
    per_meal_nutrient = [
        c for c in iis_constraints if c.constraint_type == "per_meal_nutrient"
    ]
    daily_nutrient = [
        c for c in iis_constraints if c.constraint_type == "daily_nutrient"
    ]
    equicalorie = [c for c in iis_constraints if c.constraint_type == "equicalorie"]

    # Per-meal calorie conflicts
    if per_meal_cal:
        meal_names = [
            c.meal_type.value if c.meal_type else "unknown" for c in per_meal_cal
        ]
        unique_meals = list(set(meal_names))

        # Check if minimums are too high
        min_constraints = [c for c in per_meal_cal if c.is_min]
        if min_constraints:
            suggestions.append(
                f"Per-meal calorie minimums for {', '.join(unique_meals)} may be too high. "
                f"Try reducing them by 10-20%."
            )

        # Check if maximums are too low
        max_constraints = [c for c in per_meal_cal if not c.is_min]
        if max_constraints:
            suggestions.append(
                f"Per-meal calorie maximums for {', '.join(unique_meals)} may be too restrictive. "
                f"Try increasing them by 10-20%."
            )

    # Daily calorie conflicts
    if daily_cal:
        daily_min, daily_max = request.daily_calorie_range
        if daily_max - daily_min < 200:
            suggestions.append(
                f"Daily calorie range ({daily_min}-{daily_max}) is very tight. "
                f"Consider widening to at least 200 kcal range."
            )

        # Check if per-meal sums conflict with daily
        if per_meal_cal:
            total_meal_min = sum(
                m.calorie_target.min_calories
                for m in request.meals
                if m.calorie_target
            )
            if total_meal_min > daily_max:
                suggestions.append(
                    f"Sum of per-meal minimums ({total_meal_min:.0f} kcal) exceeds "
                    f"daily maximum ({daily_max:.0f} kcal). Reduce meal minimums."
                )

    # Per-meal nutrient conflicts
    if per_meal_nutrient:
        nutrient_ids = set(c.nutrient_id for c in per_meal_nutrient if c.nutrient_id)
        for nid in nutrient_ids:
            suggestions.append(
                f"Per-meal constraints for nutrient {nid} may be too strict. "
                f"Consider relaxing the minimum or maximum bounds."
            )

    # Daily nutrient conflicts
    if daily_nutrient:
        nutrient_ids = set(c.nutrient_id for c in daily_nutrient if c.nutrient_id)
        for nid in nutrient_ids:
            nc = next(
                (n for n in request.daily_nutrient_constraints if n.nutrient_id == nid),
                None,
            )
            if nc:
                suggestions.append(
                    f"Daily constraint for nutrient {nid} "
                    f"(min={nc.min_value}, max={nc.max_value}) may be infeasible "
                    f"with current food pool. Try relaxing by 10-20%."
                )

    # Equi-calorie conflicts
    if equicalorie:
        for eq_const in request.equicalorie_constraints:
            suggestions.append(
                f"Equi-calorie constraint between {eq_const.meal_a.value} and "
                f"{eq_const.meal_b.value} (tolerance={eq_const.tolerance} kcal) "
                f"may be too strict. Try increasing tolerance to {eq_const.tolerance * 2:.0f} kcal."
            )

    # Check food pool size
    if data["n_foods"] < 20:
        suggestions.append(
            f"Only {data['n_foods']} foods in the pool. "
            f"Consider adding more staple foods or relaxing tag filters."
        )

    # Fallback suggestion
    if not suggestions:
        suggestions.append(
            "The constraint combination appears infeasible. "
            "Try relaxing per-meal calorie/nutrient constraints or widening daily ranges."
        )

    return suggestions
