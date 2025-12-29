"""Food category classification based on macronutrient dominance.

Classifies foods into categories (protein, fat, carb, legume, vegetable, etc.)
based on their macronutrient ratios. Used for pool generation and category-based
filtering in the LLM meta-optimization workflow.
"""

from __future__ import annotations

import sqlite3
from enum import Enum
from typing import Optional


class FoodCategory(Enum):
    """Food categories based on macronutrient dominance."""

    PROTEIN = "protein"
    FAT = "fat"
    CARB = "carb"
    LEGUME = "legume"
    VEGETABLE = "vegetable"
    FRUIT = "fruit"
    MIXED = "mixed"


# Keywords to help identify specific food types
LEGUME_KEYWORDS = [
    "bean", "lentil", "chickpea", "pea,", "hummus", "pinto", "black bean",
    "kidney", "navy bean", "lima", "edamame", "soybean",
]

FRUIT_KEYWORDS = [
    "apple", "banana", "berry", "orange", "grape", "mango", "pear",
    "peach", "plum", "cherry", "melon", "watermelon", "cantaloupe",
    "strawberr", "blueberr", "raspberr", "blackberr", "kiwi", "pineapple",
]

VEGETABLE_KEYWORDS = [
    "spinach", "kale", "broccoli", "cauliflower", "lettuce", "cabbage",
    "carrot", "celery", "cucumber", "pepper", "tomato", "onion", "garlic",
    "asparagus", "zucchini", "squash", "eggplant", "mushroom", "artichoke",
    "beet", "radish", "turnip", "chard", "collard", "arugula",
]

# USDA nutrient IDs
ENERGY_ID = 1008
PROTEIN_ID = 1003
FAT_ID = 1004
CARB_ID = 1005
FIBER_ID = 1079


def classify_food(
    protein_per_100g: float,
    carbs_per_100g: float,
    fat_per_100g: float,
    fiber_per_100g: float,
    energy_per_100g: float,
    description: str,
) -> FoodCategory:
    """Classify a food based on macronutrient dominance.

    Args:
        protein_per_100g: Protein in grams per 100g
        carbs_per_100g: Carbohydrates in grams per 100g
        fat_per_100g: Fat in grams per 100g
        fiber_per_100g: Fiber in grams per 100g
        energy_per_100g: Energy in kcal per 100g
        description: Food description (for keyword matching)

    Returns:
        FoodCategory enum value
    """
    # Calculate calorie percentages from macros
    protein_cals = protein_per_100g * 4
    carb_cals = carbs_per_100g * 4
    fat_cals = fat_per_100g * 9
    total_cals = protein_cals + carb_cals + fat_cals

    if total_cals == 0:
        return FoodCategory.MIXED

    protein_pct = protein_cals / total_cals
    carb_pct = carb_cals / total_cals
    fat_pct = fat_cals / total_cals

    desc_lower = description.lower()

    # Check for keyword matches first for specific categories
    has_legume_keyword = any(kw in desc_lower for kw in LEGUME_KEYWORDS)
    has_fruit_keyword = any(kw in desc_lower for kw in FRUIT_KEYWORDS)
    has_vegetable_keyword = any(kw in desc_lower for kw in VEGETABLE_KEYWORDS)

    # Classification logic
    if protein_pct > 0.40:
        return FoodCategory.PROTEIN
    elif fat_pct > 0.60:
        return FoodCategory.FAT
    elif carb_pct > 0.50:
        if fiber_per_100g > 5 and has_legume_keyword:
            return FoodCategory.LEGUME
        if has_fruit_keyword:
            return FoodCategory.FRUIT
        if fiber_per_100g > 2 and energy_per_100g < 50 and has_vegetable_keyword:
            return FoodCategory.VEGETABLE
        return FoodCategory.CARB
    elif has_vegetable_keyword and fiber_per_100g > 1 and energy_per_100g < 50:
        return FoodCategory.VEGETABLE
    elif has_legume_keyword and fiber_per_100g > 4:
        return FoodCategory.LEGUME
    elif has_fruit_keyword:
        return FoodCategory.FRUIT
    else:
        return FoodCategory.MIXED


def classify_foods_in_db(
    conn: sqlite3.Connection,
    fdc_ids: list[int],
) -> dict[int, FoodCategory]:
    """Classify multiple foods from the database.

    Args:
        conn: Database connection
        fdc_ids: List of food IDs to classify

    Returns:
        Dict mapping fdc_id to FoodCategory
    """
    if not fdc_ids:
        return {}

    results: dict[int, FoodCategory] = {}

    # Query nutrient data for all foods
    placeholders = ",".join("?" * len(fdc_ids))
    nutrient_ids = [ENERGY_ID, PROTEIN_ID, FAT_ID, CARB_ID, FIBER_ID]
    nutrient_placeholders = ",".join("?" * len(nutrient_ids))

    query = f"""
        SELECT f.fdc_id, f.description, fn.nutrient_id, fn.amount
        FROM foods f
        LEFT JOIN food_nutrients fn ON f.fdc_id = fn.fdc_id
            AND fn.nutrient_id IN ({nutrient_placeholders})
        WHERE f.fdc_id IN ({placeholders})
    """
    params = nutrient_ids + fdc_ids

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    # Build per-food nutrient data
    food_nutrients: dict[int, dict[int, float]] = {}
    food_descriptions: dict[int, str] = {}

    for fdc_id, description, nutrient_id, amount in rows:
        if fdc_id not in food_nutrients:
            food_nutrients[fdc_id] = {}
            food_descriptions[fdc_id] = description or f"Food {fdc_id}"
        if nutrient_id is not None:
            food_nutrients[fdc_id][nutrient_id] = amount or 0.0

    # Classify each food
    for fdc_id in fdc_ids:
        nutrients = food_nutrients.get(fdc_id, {})
        description = food_descriptions.get(fdc_id, f"Food {fdc_id}")

        energy = nutrients.get(ENERGY_ID, 0.0)
        protein = nutrients.get(PROTEIN_ID, 0.0)
        fat = nutrients.get(FAT_ID, 0.0)
        carbs = nutrients.get(CARB_ID, 0.0)
        fiber = nutrients.get(FIBER_ID, 0.0)

        results[fdc_id] = classify_food(
            protein_per_100g=protein,
            carbs_per_100g=carbs,
            fat_per_100g=fat,
            fiber_per_100g=fiber,
            energy_per_100g=energy,
            description=description,
        )

    return results


def get_foods_by_category(
    conn: sqlite3.Connection,
    category: FoodCategory,
    tag: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Get foods that belong to a specific category.

    Args:
        conn: Database connection
        category: Category to filter by
        tag: Optional tag to filter by
        limit: Maximum number of foods to return

    Returns:
        List of food dicts with fdc_id, description, category
    """
    # Build query for foods with the specified tag
    if tag:
        query = """
            SELECT DISTINCT f.fdc_id, f.description
            FROM foods f
            INNER JOIN food_tags ft ON f.fdc_id = ft.fdc_id
            WHERE ft.tag = ? AND f.is_active = TRUE
            LIMIT ?
        """
        cursor = conn.execute(query, (tag.lower(), limit * 3))  # Get more, filter later
    else:
        query = """
            SELECT fdc_id, description
            FROM foods
            WHERE is_active = TRUE
            LIMIT ?
        """
        cursor = conn.execute(query, (limit * 3,))

    foods = cursor.fetchall()
    fdc_ids = [f[0] for f in foods]
    descriptions = {f[0]: f[1] for f in foods}

    # Classify all foods
    categories = classify_foods_in_db(conn, fdc_ids)

    # Filter to requested category
    results = []
    for fdc_id in fdc_ids:
        if categories.get(fdc_id) == category:
            results.append({
                "fdc_id": fdc_id,
                "description": descriptions[fdc_id],
                "category": category.value,
            })
            if len(results) >= limit:
                break

    return results


# Price estimation tiers (per 100g) - rough estimates
PRICE_TIERS: dict[str, float] = {
    "canned_fish": 0.40,
    "fresh_fish": 0.80,
    "eggs": 0.20,
    "legumes_canned": 0.15,
    "legumes_dried": 0.08,
    "vegetables_fresh": 0.25,
    "vegetables_frozen": 0.15,
    "olive_oil": 0.50,
    "nuts": 0.80,
    "chicken": 0.50,
    "beef": 0.70,
    "pork": 0.45,
    "dairy": 0.30,
    "grains": 0.10,
    "default": 0.30,
}

# Keywords for price tier matching
PRICE_TIER_KEYWORDS: dict[str, list[str]] = {
    "canned_fish": ["canned", "sardine", "tuna, canned", "salmon, canned"],
    "fresh_fish": ["salmon", "cod", "tilapia", "shrimp", "fish"],
    "eggs": ["egg"],
    "legumes_canned": ["canned bean", "canned lentil", "hummus"],
    "legumes_dried": ["dried bean", "dried lentil", "dried pea"],
    "vegetables_frozen": ["frozen"],
    "olive_oil": ["olive oil"],
    "nuts": ["almond", "walnut", "cashew", "peanut", "nut"],
    "chicken": ["chicken"],
    "beef": ["beef", "steak"],
    "pork": ["pork", "bacon", "ham"],
    "dairy": ["cheese", "yogurt", "milk", "cottage"],
    "grains": ["rice", "oat", "wheat", "bread", "pasta"],
}


def estimate_price(description: str, category: FoodCategory) -> float:
    """Estimate price per 100g based on food description and category.

    Args:
        description: Food description
        category: Food category

    Returns:
        Estimated price per 100g
    """
    desc_lower = description.lower()

    # Check keyword matches first
    for tier, keywords in PRICE_TIER_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return PRICE_TIERS.get(tier, PRICE_TIERS["default"])

    # Fall back to category-based estimate
    category_tier_map = {
        FoodCategory.PROTEIN: "chicken",  # Default protein price
        FoodCategory.FAT: "olive_oil",
        FoodCategory.CARB: "grains",
        FoodCategory.LEGUME: "legumes_canned",
        FoodCategory.VEGETABLE: "vegetables_fresh",
        FoodCategory.FRUIT: "vegetables_fresh",  # Similar price point
        FoodCategory.MIXED: "default",
    }

    tier = category_tier_map.get(category, "default")
    return PRICE_TIERS.get(tier, PRICE_TIERS["default"])
