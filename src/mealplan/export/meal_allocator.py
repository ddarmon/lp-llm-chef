"""Meal slot allocation for distributing foods into meals.

Uses heuristics based on food keywords and categories to distribute
optimization results into breakfast, lunch, dinner, and snack slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mealplan.optimizer.models import FoodResult


@dataclass
class MealSlot:
    """A meal slot with allocated foods."""

    name: str  # "breakfast", "lunch", "dinner", "snack"
    target_calories: float
    foods: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_calories(self) -> float:
        """Calculate total calories in this meal."""
        return sum(f.get("calories", 0) for f in self.foods)

    @property
    def total_protein(self) -> float:
        """Calculate total protein in this meal."""
        return sum(f.get("protein", 0) for f in self.foods)


# Meal affinity based on food keywords
MEAL_AFFINITY: dict[str, list[str]] = {
    # Breakfast foods
    "egg": ["breakfast"],
    "bacon": ["breakfast"],
    "sausage": ["breakfast"],
    "oatmeal": ["breakfast"],
    "cereal": ["breakfast"],
    "yogurt": ["breakfast", "snack"],
    "cottage cheese": ["breakfast", "snack"],

    # Lunch/dinner foods
    "chicken": ["lunch", "dinner"],
    "fish": ["lunch", "dinner"],
    "salmon": ["lunch", "dinner"],
    "cod": ["lunch", "dinner"],
    "tuna": ["lunch", "dinner"],
    "sardine": ["lunch", "dinner"],
    "shrimp": ["lunch", "dinner"],
    "beef": ["lunch", "dinner"],
    "pork": ["lunch", "dinner"],

    # Legumes for lunch/dinner
    "bean": ["lunch", "dinner"],
    "lentil": ["lunch", "dinner"],
    "chickpea": ["lunch", "dinner"],
    "hummus": ["lunch", "dinner"],

    # Vegetables for lunch/dinner
    "broccoli": ["lunch", "dinner"],
    "spinach": ["lunch", "dinner"],
    "kale": ["lunch", "dinner"],
    "asparagus": ["lunch", "dinner"],
    "cauliflower": ["lunch", "dinner"],
    "zucchini": ["lunch", "dinner"],
    "tomato": ["lunch", "dinner"],
    "pepper": ["lunch", "dinner"],
    "cabbage": ["lunch", "dinner"],

    # Snack foods
    "nut": ["snack"],
    "almond": ["snack"],
    "walnut": ["snack"],
    "cashew": ["snack"],
}

# Default meal structure (percentage of daily calories)
DEFAULT_MEAL_STRUCTURE: dict[str, float] = {
    "breakfast": 0.25,
    "lunch": 0.35,
    "dinner": 0.35,
    "snack": 0.05,
}


def get_meal_affinity(description: str) -> list[str]:
    """Determine which meals a food is suited for.

    Args:
        description: Food description

    Returns:
        List of meal names (e.g., ["lunch", "dinner"])
    """
    desc_lower = description.lower()

    for keyword, meals in MEAL_AFFINITY.items():
        if keyword in desc_lower:
            return meals

    # Default: can go in any meal
    return ["breakfast", "lunch", "dinner"]


def allocate_to_meals(
    foods: list[FoodResult],
    total_calories: float,
    meal_structure: dict[str, float] | None = None,
) -> list[MealSlot]:
    """Distribute foods into meal slots using heuristics.

    Algorithm:
    1. Calculate target calories per meal
    2. For each food, determine meal affinity
    3. Assign foods to meals based on affinity and calorie targets
    4. Balance by moving foods if a meal is over/under target

    Args:
        foods: List of FoodResult from optimization
        total_calories: Total daily calories
        meal_structure: Optional custom meal structure (defaults to 25/35/35/5)

    Returns:
        List of MealSlot objects with allocated foods
    """
    if meal_structure is None:
        meal_structure = DEFAULT_MEAL_STRUCTURE

    # Create meal slots
    meals = {
        name: MealSlot(
            name=name,
            target_calories=total_calories * fraction,
        )
        for name, fraction in meal_structure.items()
    }

    # Build food info with affinity
    food_info = []
    for f in foods:
        # Get calories from nutrients (nutrient_id 1008)
        calories = f.nutrients.get(1008, 0) * f.grams / 100 if f.nutrients else 0
        protein = f.nutrients.get(1003, 0) * f.grams / 100 if f.nutrients else 0

        food_info.append({
            "fdc_id": f.fdc_id,
            "description": f.description,
            "grams": round(f.grams, 1),
            "calories": round(calories, 0),
            "protein": round(protein, 1),
            "affinity": get_meal_affinity(f.description),
        })

    # Sort by calories (larger items first for better packing)
    food_info.sort(key=lambda x: -x["calories"])

    # Greedy assignment: assign each food to its preferred meal that is furthest from target
    for food in food_info:
        # Find best meal for this food
        best_meal = None
        best_gap = float("-inf")

        for meal_name in food["affinity"]:
            if meal_name in meals:
                meal = meals[meal_name]
                gap = meal.target_calories - meal.total_calories
                if gap > best_gap:
                    best_gap = gap
                    best_meal = meal

        # If no preferred meal available, use any meal with most room
        if best_meal is None:
            for meal in meals.values():
                gap = meal.target_calories - meal.total_calories
                if gap > best_gap:
                    best_gap = gap
                    best_meal = meal

        if best_meal:
            best_meal.foods.append(food)

    return list(meals.values())


def format_meal_allocation(
    meals: list[MealSlot],
) -> dict[str, Any]:
    """Format meal allocation for JSON output.

    Args:
        meals: List of MealSlot objects

    Returns:
        Dict with meal allocation data
    """
    return {
        "meals": [
            {
                "name": meal.name,
                "target_calories": round(meal.target_calories, 0),
                "actual_calories": round(meal.total_calories, 0),
                "protein": round(meal.total_protein, 1),
                "foods": [
                    {
                        "description": f["description"],
                        "grams": f["grams"],
                        "calories": f["calories"],
                    }
                    for f in meal.foods
                ],
            }
            for meal in meals
        ],
    }


def format_meal_allocation_text(meals: list[MealSlot]) -> str:
    """Format meal allocation for text/markdown output.

    Args:
        meals: List of MealSlot objects

    Returns:
        Markdown-formatted string
    """
    lines = ["## Meal Allocation", ""]

    for meal in meals:
        if not meal.foods:
            continue

        lines.append(f"### {meal.name.title()} ({round(meal.total_calories):.0f} kcal)")

        for food in meal.foods:
            lines.append(f"  - {food['description']}: {food['grams']:.0f}g")

        lines.append("")

    return "\n".join(lines)
