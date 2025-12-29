"""Consolidate optimizer output into practical meal portions.

The optimizer produces many tiny portions (40+ foods at 10-30g). This module
consolidates them into practical servings suitable for actual meal prep.

Rules:
- Maximum 4-6 foods per meal
- Minimum portion: 50g (25g for snacks)
- Round to nearest 25g
- Merge similar foods (raw/cooked variants, same base ingredient)
- Group by food type within each meal
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from llmn.optimizer.multiperiod_models import MealResult, MultiPeriodResult


@dataclass
class ConsolidatedFood:
    """A consolidated food item in a meal."""

    name: str
    grams: float
    calories: float
    protein: float
    carbs: float
    fat: float
    original_foods: list[str] = field(default_factory=list)  # Source food descriptions
    fdc_ids: list[int] = field(default_factory=list)


@dataclass
class ConsolidatedMeal:
    """A meal with consolidated, practical portions."""

    meal_name: str
    foods: list[ConsolidatedFood]
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float


@dataclass
class ConsolidatedResult:
    """Complete consolidated meal plan."""

    meals: list[ConsolidatedMeal]
    daily_totals: dict[str, float]
    consolidation_notes: list[str]  # What was merged/removed


# Patterns for identifying base ingredients (for merging similar foods)
BASE_INGREDIENT_PATTERNS = [
    # Remove prep states
    (r",?\s*(raw|cooked|boiled|steamed|baked|fried|roasted|grilled)", ""),
    (r",?\s*(canned|frozen|fresh|dried)", ""),
    (r",?\s*(with salt|without salt|no salt added)", ""),
    (r",?\s*(drained solids|solids and liquids)", ""),
    (r",?\s*(chopped|sliced|diced|whole|pieces)", ""),
    # Remove brand/variety info
    (r"\([^)]+\)", ""),  # Remove parenthetical info
    (r",\s*NFS$", ""),  # Not Further Specified
    (r",\s*NS.*$", ""),  # Not Specified
]


def get_base_ingredient(description: str) -> str:
    """Extract base ingredient name for grouping similar foods.

    Examples:
        "Spinach, raw" -> "spinach"
        "Spinach, cooked, boiled, drained" -> "spinach"
        "Eggs, scrambled" -> "eggs"
    """
    result = description.lower().strip()

    # Apply cleanup patterns
    for pattern, replacement in BASE_INGREDIENT_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Take first comma-separated part as base
    if "," in result:
        result = result.split(",")[0].strip()

    return result.strip()


def consolidate_meal(
    meal: MealResult,
    max_foods: int = 6,
    min_portion: float = 50.0,
    round_to: float = 25.0,
    is_snack: bool = False,
) -> ConsolidatedMeal:
    """Consolidate a single meal's foods into practical portions.

    Args:
        meal: Original meal result from optimizer
        max_foods: Maximum distinct foods to keep (default 6)
        min_portion: Minimum portion size in grams (default 50g, 25g for snacks)
        round_to: Round portions to this increment (default 25g)
        is_snack: If True, use smaller minimum portion

    Returns:
        ConsolidatedMeal with practical portions
    """
    if is_snack:
        min_portion = 25.0

    # Calculate TRUE meal totals from original meal (before any filtering)
    # This ensures we report accurate nutrition even when displaying fewer foods
    true_calories = meal.total_calories
    true_protein = meal.total_protein
    true_carbs = meal.total_carbs
    true_fat = meal.total_fat

    # Group foods by base ingredient
    groups: dict[str, list[tuple]] = {}  # base -> [(food, grams, cals, prot, carbs, fat)]

    for food in meal.foods:
        base = get_base_ingredient(food.description)

        # Get nutrient values
        cals = food.nutrients.get(1008, 0)
        prot = food.nutrients.get(1003, 0)
        carbs = food.nutrients.get(1005, 0)
        fat = food.nutrients.get(1004, 0)

        if base not in groups:
            groups[base] = []
        groups[base].append((food, food.grams, cals, prot, carbs, fat))

    # Merge each group into a single ConsolidatedFood
    merged: list[ConsolidatedFood] = []

    for base, items in groups.items():
        total_grams = sum(item[1] for item in items)
        total_cals = sum(item[2] for item in items)
        total_prot = sum(item[3] for item in items)
        total_carbs = sum(item[4] for item in items)
        total_fat = sum(item[5] for item in items)

        original_foods = [item[0].description for item in items]
        fdc_ids = [item[0].fdc_id for item in items]

        # Round to nearest increment
        rounded_grams = round(total_grams / round_to) * round_to

        # Skip if below minimum (after rounding)
        if rounded_grams < min_portion:
            continue

        # Create cleaned name (capitalize first letter)
        name = base.title()

        merged.append(ConsolidatedFood(
            name=name,
            grams=rounded_grams,
            calories=total_cals,
            protein=total_prot,
            carbs=total_carbs,
            fat=total_fat,
            original_foods=original_foods,
            fdc_ids=fdc_ids,
        ))

    # Sort by calories (highest first) and take top max_foods
    merged.sort(key=lambda x: x.calories, reverse=True)
    final_foods = merged[:max_foods]

    # Use TRUE totals from original meal (not from filtered/merged foods)
    # This ensures accurate nutrition reporting even when we display fewer foods
    total_calories = true_calories
    total_protein = true_protein
    total_carbs = true_carbs
    total_fat = true_fat

    return ConsolidatedMeal(
        meal_name=meal.meal_type.value,
        foods=final_foods,
        total_calories=total_calories,
        total_protein=total_protein,
        total_carbs=total_carbs,
        total_fat=total_fat,
    )


def consolidate_result(
    result: MultiPeriodResult,
    max_foods_per_meal: int = 6,
    min_portion: float = 50.0,
) -> ConsolidatedResult:
    """Consolidate a full multi-period result into practical meals.

    Args:
        result: Original multi-period optimization result
        max_foods_per_meal: Maximum foods per meal (default 6)
        min_portion: Minimum portion in grams (default 50g)

    Returns:
        ConsolidatedResult with practical meal plan
    """
    if not result.success:
        return ConsolidatedResult(
            meals=[],
            daily_totals={},
            consolidation_notes=["Optimization was not successful"],
        )

    consolidated_meals = []
    notes = []

    for meal in result.meals:
        is_snack = meal.meal_type.value == "snack"
        original_food_count = len(meal.foods)

        consolidated = consolidate_meal(
            meal,
            max_foods=max_foods_per_meal,
            min_portion=min_portion,
            is_snack=is_snack,
        )

        consolidated_meals.append(consolidated)

        # Track what was consolidated
        final_count = len(consolidated.foods)
        if original_food_count > final_count:
            notes.append(
                f"{meal.meal_type.value}: {original_food_count} foods -> {final_count} foods"
            )

    # Calculate daily totals
    daily_totals = {
        "calories": sum(m.total_calories for m in consolidated_meals),
        "protein": sum(m.total_protein for m in consolidated_meals),
        "carbs": sum(m.total_carbs for m in consolidated_meals),
        "fat": sum(m.total_fat for m in consolidated_meals),
    }

    return ConsolidatedResult(
        meals=consolidated_meals,
        daily_totals=daily_totals,
        consolidation_notes=notes,
    )


def format_consolidated_result(result: ConsolidatedResult) -> str:
    """Format consolidated result as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("CONSOLIDATED MEAL PLAN")
    lines.append("=" * 60)
    lines.append("")

    for meal in result.meals:
        lines.append(f"## {meal.meal_name.upper()}")
        lines.append(f"   ({meal.total_calories:.0f} kcal, {meal.total_protein:.0f}g protein)")
        lines.append("")

        for food in meal.foods:
            lines.append(f"   - {food.name}: {food.grams:.0f}g")
            lines.append(f"     ({food.calories:.0f} kcal, {food.protein:.0f}g P)")

        lines.append("")

    lines.append("-" * 60)
    lines.append("DAILY TOTALS")
    lines.append(f"  Calories: {result.daily_totals['calories']:.0f} kcal")
    lines.append(f"  Protein:  {result.daily_totals['protein']:.0f}g")
    lines.append(f"  Carbs:    {result.daily_totals['carbs']:.0f}g")
    lines.append(f"  Fat:      {result.daily_totals['fat']:.0f}g")
    lines.append("-" * 60)

    if result.consolidation_notes:
        lines.append("")
        lines.append("Consolidation notes:")
        for note in result.consolidation_notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def consolidated_result_to_dict(result: ConsolidatedResult) -> dict:
    """Convert consolidated result to dict for JSON output."""
    return {
        "meals": {
            meal.meal_name: {
                "foods": [
                    {
                        "name": food.name,
                        "grams": food.grams,
                        "calories": round(food.calories, 0),
                        "protein": round(food.protein, 1),
                        "carbs": round(food.carbs, 1),
                        "fat": round(food.fat, 1),
                        "original_foods": food.original_foods,
                    }
                    for food in meal.foods
                ],
                "totals": {
                    "calories": round(meal.total_calories, 0),
                    "protein": round(meal.total_protein, 1),
                    "carbs": round(meal.total_carbs, 1),
                    "fat": round(meal.total_fat, 1),
                },
            }
            for meal in result.meals
        },
        "daily_totals": {
            k: round(v, 1) for k, v in result.daily_totals.items()
        },
        "consolidation_notes": result.consolidation_notes,
    }
