"""Food selection logic for template-based meal composition.

This module handles the discrete food selection phase: picking one food
from each slot for each meal, while enforcing diversity rules.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from typing import Optional

from mealplan.data.pattern_staples import get_foods_for_sources
from mealplan.templates.models import (
    DietTemplate,
    DiversityRule,
    MealTemplate,
    SelectedFood,
    SelectedMeal,
    SelectionStrategy,
    SlotDefinition,
)


@dataclass
class FoodCandidate:
    """A candidate food for slot selection."""

    fdc_id: int
    description: str
    protein_per_100g: float
    calories_per_100g: float
    fiber_per_100g: float


def get_candidates_for_slot(
    conn: sqlite3.Connection,
    slot: SlotDefinition,
    exclude_ids: set[int] | None = None,
) -> list[FoodCandidate]:
    """Get candidate foods for a template slot.

    Args:
        conn: Database connection
        slot: Slot definition with source names
        exclude_ids: FDC IDs to exclude (for diversity)

    Returns:
        List of FoodCandidate objects.
    """
    exclude_ids = exclude_ids or set()

    # Get all FDC IDs from the slot's sources
    source_ids = get_foods_for_sources(slot.sources)
    if not source_ids:
        return []

    # Filter out excluded foods
    available_ids = [fid for fid in source_ids if fid not in exclude_ids]
    if not available_ids:
        return []

    # Query database for food details
    placeholders = ",".join("?" * len(available_ids))
    query = f"""
        SELECT
            f.fdc_id,
            f.description,
            COALESCE(prot.amount, 0) as protein,
            COALESCE(cal.amount, 0) as calories,
            COALESCE(fib.amount, 0) as fiber
        FROM foods f
        LEFT JOIN food_nutrients prot ON f.fdc_id = prot.fdc_id AND prot.nutrient_id = 1003
        LEFT JOIN food_nutrients cal ON f.fdc_id = cal.fdc_id AND cal.nutrient_id = 1008
        LEFT JOIN food_nutrients fib ON f.fdc_id = fib.fdc_id AND fib.nutrient_id = 1079
        WHERE f.fdc_id IN ({placeholders})
        AND f.is_active = 1
    """

    cursor = conn.execute(query, available_ids)
    candidates = []
    for row in cursor.fetchall():
        candidates.append(
            FoodCandidate(
                fdc_id=row[0],
                description=row[1],
                protein_per_100g=row[2] or 0,
                calories_per_100g=row[3] or 0,
                fiber_per_100g=row[4] or 0,
            )
        )

    return candidates


def select_food_for_slot(
    candidates: list[FoodCandidate],
    strategy: SelectionStrategy,
    rng: random.Random,
) -> FoodCandidate | None:
    """Select one food from candidates using the given strategy.

    Args:
        candidates: Available food candidates
        strategy: Selection strategy to use
        rng: Random number generator (for reproducibility)

    Returns:
        Selected FoodCandidate or None if no candidates.
    """
    if not candidates:
        return None

    if strategy == SelectionStrategy.RANDOM:
        return rng.choice(candidates)

    elif strategy == SelectionStrategy.HIGHEST_PROTEIN:
        return max(candidates, key=lambda c: c.protein_per_100g)

    elif strategy == SelectionStrategy.HIGHEST_FIBER:
        return max(candidates, key=lambda c: c.fiber_per_100g)

    elif strategy == SelectionStrategy.LOWEST_CALORIE:
        # Filter out zero-calorie foods (likely data errors)
        valid = [c for c in candidates if c.calories_per_100g > 0]
        if not valid:
            return rng.choice(candidates)
        return min(valid, key=lambda c: c.calories_per_100g)

    else:
        return rng.choice(candidates)


def select_foods_for_meal(
    conn: sqlite3.Connection,
    meal_template: MealTemplate,
    strategy: SelectionStrategy,
    rng: random.Random,
    used_foods: set[int],
    diversity_rule: DiversityRule,
) -> SelectedMeal:
    """Select foods for all slots in a meal.

    Args:
        conn: Database connection
        meal_template: Template defining meal structure
        strategy: Selection strategy
        rng: Random number generator
        used_foods: Set of already-used FDC IDs (modified in place)
        diversity_rule: How to enforce diversity

    Returns:
        SelectedMeal with selections for each slot.
    """
    selections: list[SelectedFood] = []

    for slot in meal_template.slots:
        # Select `count` foods for this slot
        for _ in range(slot.count):
            candidates = get_candidates_for_slot(conn, slot, exclude_ids=used_foods)

            if not candidates:
                # No candidates available - skip this slot instance
                continue

            selected = select_food_for_slot(candidates, strategy, rng)
            if selected is None:
                continue

            selections.append(
                SelectedFood(
                    slot=slot,
                    fdc_id=selected.fdc_id,
                    description=selected.description,
                    target_grams=slot.target_grams,
                )
            )

            # Update used foods based on diversity rule
            if diversity_rule == DiversityRule.NO_REPEAT:
                used_foods.add(selected.fdc_id)
            elif diversity_rule == DiversityRule.NO_REPEAT_ADJACENT:
                # Only prevent immediate reuse - handled at meal level
                used_foods.add(selected.fdc_id)

    return SelectedMeal(meal_type=meal_template.meal_type, selections=selections)


def select_foods_for_template(
    conn: sqlite3.Connection,
    template: DietTemplate,
    strategy: SelectionStrategy = SelectionStrategy.RANDOM,
    seed: Optional[int] = None,
    excluded_foods: Optional[set[int]] = None,
    preferred_foods: Optional[dict[str, int]] = None,
) -> list[SelectedMeal]:
    """Select foods for all meals in a template.

    Args:
        conn: Database connection
        template: Diet template to fill
        strategy: Selection strategy
        seed: Random seed for reproducibility
        excluded_foods: FDC IDs to never select
        preferred_foods: Dict mapping slot names to preferred FDC IDs

    Returns:
        List of SelectedMeal objects, one per meal.
    """
    rng = random.Random(seed)
    used_foods: set[int] = set(excluded_foods or set())
    preferred = preferred_foods or {}

    # Handle preferred foods first - validate they exist and add to selections
    # This is handled in the meal selection loop below

    selections: list[SelectedMeal] = []
    previous_meal_foods: set[int] = set()

    for meal_template in template.meals:
        # For NO_REPEAT_ADJACENT, only exclude previous meal's foods
        if template.diversity_rule == DiversityRule.NO_REPEAT_ADJACENT:
            exclude_for_meal = used_foods | previous_meal_foods
        else:
            exclude_for_meal = used_foods

        meal_selection = select_foods_for_meal(
            conn=conn,
            meal_template=meal_template,
            strategy=strategy,
            rng=rng,
            used_foods=exclude_for_meal,
            diversity_rule=template.diversity_rule,
        )

        # Apply preferred foods (override selections)
        meal_selection = _apply_preferred_foods(
            conn, meal_selection, preferred, exclude_for_meal
        )

        selections.append(meal_selection)

        # Track for adjacent diversity rule
        previous_meal_foods = {s.fdc_id for s in meal_selection.selections}

        # Update global used set for NO_REPEAT rule
        if template.diversity_rule == DiversityRule.NO_REPEAT:
            used_foods.update(previous_meal_foods)

    return selections


def _apply_preferred_foods(
    conn: sqlite3.Connection,
    meal: SelectedMeal,
    preferred: dict[str, int],
    exclude_ids: set[int],
) -> SelectedMeal:
    """Apply user-preferred foods to a meal selection.

    If a preferred food is specified for a slot, replace the random selection
    with the preferred food (if it's valid for that slot).

    Args:
        conn: Database connection
        meal: Current meal selection
        preferred: Dict mapping slot names to preferred FDC IDs
        exclude_ids: Foods that shouldn't be used

    Returns:
        Modified SelectedMeal with preferences applied.
    """
    if not preferred:
        return meal

    new_selections: list[SelectedFood] = []

    for selection in meal.selections:
        slot_name = selection.slot.name
        preferred_id = preferred.get(slot_name)

        if preferred_id is not None and preferred_id not in exclude_ids:
            # Check if preferred food is valid for this slot
            source_ids = get_foods_for_sources(selection.slot.sources)
            if preferred_id in source_ids:
                # Get food description from database
                cursor = conn.execute(
                    "SELECT description FROM foods WHERE fdc_id = ?",
                    (preferred_id,)
                )
                row = cursor.fetchone()
                if row:
                    new_selections.append(
                        SelectedFood(
                            slot=selection.slot,
                            fdc_id=preferred_id,
                            description=row[0],
                            target_grams=selection.target_grams,
                        )
                    )
                    continue

        # Keep original selection
        new_selections.append(selection)

    return SelectedMeal(meal_type=meal.meal_type, selections=new_selections)


def validate_selections(selections: list[SelectedMeal], template: DietTemplate) -> list[str]:
    """Validate that selections meet basic template requirements.

    Args:
        selections: Food selections to validate
        template: Template that was used

    Returns:
        List of warning messages (empty if valid).
    """
    warnings: list[str] = []

    # Check that we have selections for each meal
    meal_types_needed = {m.meal_type for m in template.meals}
    meal_types_have = {s.meal_type for s in selections}
    missing = meal_types_needed - meal_types_have
    if missing:
        warnings.append(f"Missing selections for meals: {[m.value for m in missing]}")

    # Check that each meal has at least one selection
    for meal in selections:
        if not meal.selections:
            warnings.append(f"{meal.meal_type.value}: No foods selected")

    # Check for duplicate foods (if NO_REPEAT rule)
    if template.diversity_rule == DiversityRule.NO_REPEAT:
        all_ids = [s.fdc_id for meal in selections for s in meal.selections]
        if len(all_ids) != len(set(all_ids)):
            warnings.append("Duplicate foods selected despite NO_REPEAT rule")

    return warnings
