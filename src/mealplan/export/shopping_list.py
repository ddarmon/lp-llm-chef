"""Shopping list generator from optimization results.

Aggregates foods across all meals and days, groups by category,
and produces a practical shopping list with weekly totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mealplan.data.food_categories import FoodCategory, classify_food
from mealplan.export.meal_consolidator import ConsolidatedResult


@dataclass
class ShoppingItem:
    """A single item on the shopping list."""

    name: str
    category: str
    total_grams: float
    daily_grams: float
    weekly_grams: float
    estimated_cost: Optional[float] = None
    unit_suggestion: Optional[str] = None  # e.g., "2 cans" or "1 lb"
    notes: list[str] = field(default_factory=list)


@dataclass
class ShoppingList:
    """Complete shopping list grouped by category."""

    items_by_category: dict[str, list[ShoppingItem]]
    weekly_totals: dict[str, float]  # calories, protein, cost
    days: int
    notes: list[str] = field(default_factory=list)


# Common unit conversions for practical shopping
UNIT_CONVERSIONS = {
    # Proteins
    "eggs": {"unit": "eggs", "grams_per_unit": 50, "format": "{n} eggs"},
    "salmon": {"unit": "lb", "grams_per_unit": 454, "format": "{n:.1f} lb"},
    "tuna": {"unit": "cans", "grams_per_unit": 140, "format": "{n} cans (5oz)"},
    "chicken": {"unit": "lb", "grams_per_unit": 454, "format": "{n:.1f} lb"},
    "fish": {"unit": "lb", "grams_per_unit": 454, "format": "{n:.1f} lb"},

    # Legumes
    "lentil": {"unit": "lb", "grams_per_unit": 454, "format": "{n:.1f} lb dried"},
    "bean": {"unit": "cans", "grams_per_unit": 425, "format": "{n} cans (15oz)"},
    "chickpea": {"unit": "cans", "grams_per_unit": 425, "format": "{n} cans (15oz)"},

    # Vegetables
    "spinach": {"unit": "bags", "grams_per_unit": 280, "format": "{n} bags (10oz)"},
    "broccoli": {"unit": "heads", "grams_per_unit": 500, "format": "{n} heads"},
    "kale": {"unit": "bunches", "grams_per_unit": 200, "format": "{n} bunches"},

    # Default
    "default": {"unit": "g", "grams_per_unit": 100, "format": "{n:.0f}g"},
}


def suggest_shopping_unit(name: str, total_grams: float) -> str:
    """Suggest practical shopping units for a food item.

    Args:
        name: Food name (lowercase)
        total_grams: Total grams needed

    Returns:
        Human-friendly unit string (e.g., "2 cans" or "1.5 lb")
    """
    name_lower = name.lower()

    # Find matching conversion
    conversion = UNIT_CONVERSIONS["default"]
    for keyword, conv in UNIT_CONVERSIONS.items():
        if keyword in name_lower:
            conversion = conv
            break

    # Calculate units needed (round up)
    import math
    units_needed = math.ceil(total_grams / conversion["grams_per_unit"])

    return conversion["format"].format(n=units_needed)


def categorize_for_shopping(name: str) -> str:
    """Categorize a food for shopping list grouping.

    Uses simple keyword matching for shopping categories.
    """
    name_lower = name.lower()

    # Check keywords in order of specificity
    if any(kw in name_lower for kw in ["salmon", "tuna", "fish", "cod", "tilapia", "shrimp"]):
        return "Proteins - Seafood"
    if any(kw in name_lower for kw in ["egg", "chicken", "turkey"]):
        return "Proteins - Poultry & Eggs"
    if any(kw in name_lower for kw in ["beef", "pork", "lamb"]):
        return "Proteins - Meat"
    if any(kw in name_lower for kw in ["bean", "lentil", "chickpea", "tofu", "tempeh"]):
        return "Legumes & Plant Proteins"
    if any(kw in name_lower for kw in ["spinach", "kale", "lettuce", "broccoli", "cauliflower", "pepper", "tomato", "onion", "garlic", "cabbage", "zucchini", "squash", "mushroom", "carrot", "celery"]):
        return "Vegetables"
    if any(kw in name_lower for kw in ["apple", "banana", "berry", "orange", "grape"]):
        return "Fruits"
    if any(kw in name_lower for kw in ["rice", "oat", "bread", "pasta", "quinoa"]):
        return "Grains"
    if any(kw in name_lower for kw in ["almond", "walnut", "cashew", "peanut", "nut"]):
        return "Nuts & Seeds"
    if any(kw in name_lower for kw in ["oil", "butter"]):
        return "Oils & Fats"
    if any(kw in name_lower for kw in ["milk", "cheese", "yogurt"]):
        return "Dairy"

    return "Other"


def generate_shopping_list(
    result: ConsolidatedResult,
    days: int = 7,
) -> ShoppingList:
    """Generate a shopping list from consolidated meal plan.

    Args:
        result: Consolidated meal plan result
        days: Number of days to shop for (default 7)

    Returns:
        ShoppingList with items grouped by category
    """
    # Aggregate foods across all meals
    food_totals: dict[str, dict] = {}  # name -> {grams, calories, protein}

    for meal in result.meals:
        for food in meal.foods:
            name = food.name.lower()
            if name not in food_totals:
                food_totals[name] = {
                    "grams": 0,
                    "calories": 0,
                    "protein": 0,
                    "display_name": food.name,
                }
            food_totals[name]["grams"] += food.grams
            food_totals[name]["calories"] += food.calories
            food_totals[name]["protein"] += food.protein

    # Create shopping items grouped by category
    items_by_category: dict[str, list[ShoppingItem]] = {}

    for name, totals in food_totals.items():
        daily_grams = totals["grams"]
        weekly_grams = daily_grams * days

        category = categorize_for_shopping(name)
        unit_suggestion = suggest_shopping_unit(name, weekly_grams)

        item = ShoppingItem(
            name=totals["display_name"],
            category=category,
            total_grams=daily_grams,
            daily_grams=daily_grams,
            weekly_grams=weekly_grams,
            unit_suggestion=unit_suggestion,
        )

        if category not in items_by_category:
            items_by_category[category] = []
        items_by_category[category].append(item)

    # Sort items within each category by weight
    for category in items_by_category:
        items_by_category[category].sort(key=lambda x: x.weekly_grams, reverse=True)

    # Calculate weekly totals
    weekly_totals = {
        "calories": result.daily_totals.get("calories", 0) * days,
        "protein": result.daily_totals.get("protein", 0) * days,
        "total_grams": sum(t["grams"] for t in food_totals.values()) * days,
    }

    return ShoppingList(
        items_by_category=items_by_category,
        weekly_totals=weekly_totals,
        days=days,
    )


def format_shopping_list(shopping_list: ShoppingList) -> str:
    """Format shopping list as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"SHOPPING LIST ({shopping_list.days} days)")
    lines.append("=" * 60)
    lines.append("")

    # Define category order
    category_order = [
        "Proteins - Seafood",
        "Proteins - Poultry & Eggs",
        "Proteins - Meat",
        "Legumes & Plant Proteins",
        "Vegetables",
        "Fruits",
        "Grains",
        "Nuts & Seeds",
        "Oils & Fats",
        "Dairy",
        "Other",
    ]

    for category in category_order:
        if category not in shopping_list.items_by_category:
            continue

        items = shopping_list.items_by_category[category]
        lines.append(f"## {category}")

        for item in items:
            lines.append(
                f"   [ ] {item.name}: {item.unit_suggestion} "
                f"({item.weekly_grams:.0f}g total)"
            )

        lines.append("")

    lines.append("-" * 60)
    lines.append("WEEKLY TOTALS")
    lines.append(f"  Total weight: {shopping_list.weekly_totals['total_grams']:.0f}g")
    lines.append(f"  Calories: {shopping_list.weekly_totals['calories']:.0f} kcal")
    lines.append(f"  Protein: {shopping_list.weekly_totals['protein']:.0f}g")
    lines.append("-" * 60)

    return "\n".join(lines)


def shopping_list_to_dict(shopping_list: ShoppingList) -> dict:
    """Convert shopping list to dict for JSON output."""
    return {
        "days": shopping_list.days,
        "categories": {
            category: [
                {
                    "name": item.name,
                    "weekly_amount": item.unit_suggestion,
                    "weekly_grams": round(item.weekly_grams, 0),
                    "daily_grams": round(item.daily_grams, 0),
                }
                for item in items
            ]
            for category, items in shopping_list.items_by_category.items()
        },
        "weekly_totals": {
            k: round(v, 0) for k, v in shopping_list.weekly_totals.items()
        },
    }
