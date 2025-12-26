"""Schema export for LLM agents.

This module provides machine-readable schema documentation that LLMs
can use to understand how to construct valid optimization requests.
"""

from __future__ import annotations

from sqlite3 import Connection
from typing import Any, Optional

from mealplan.data.nutrient_ids import (
    MACRO_NUTRIENT_IDS,
    MINERAL_NUTRIENT_IDS,
    NUTRIENT_DISPLAY_NAMES,
    NUTRIENT_IDS,
    NUTRIENT_UNITS,
    VITAMIN_NUTRIENT_IDS,
)
from mealplan.db.queries import TagQueries


# Typical recommended ranges for nutrients (for LLM guidance)
NUTRIENT_TYPICAL_RANGES: dict[str, tuple[float, float]] = {
    "energy": (1400, 3500),
    "protein": (50, 200),
    "total_fat": (40, 150),
    "carbohydrate": (100, 400),
    "fiber": (20, 50),
    "sugar": (0, 50),
    "saturated_fat": (0, 25),
    "cholesterol": (0, 300),
    "calcium": (800, 1500),
    "iron": (8, 25),
    "magnesium": (300, 500),
    "phosphorus": (700, 1500),
    "potassium": (2500, 4700),
    "sodium": (500, 2300),
    "zinc": (8, 15),
    "vitamin_a": (700, 1500),
    "vitamin_c": (60, 200),
    "vitamin_d": (15, 50),
    "vitamin_e": (12, 30),
    "vitamin_k": (80, 200),
    "vitamin_b6": (1.2, 3),
    "vitamin_b12": (2, 10),
    "folate": (300, 600),
}


def get_constraint_schema() -> dict[str, Any]:
    """Get the full constraint schema for LLM reference.

    Returns:
        Schema dictionary describing all valid constraint options
    """
    return {
        "description": "Schema for meal plan optimization constraints",
        "format": {
            "calories": {
                "type": "range",
                "fields": ["min", "max"],
                "unit": "kcal",
                "description": "Daily calorie target range",
                "example": {"min": 1800, "max": 2200},
            },
            "nutrients": {
                "type": "dict",
                "description": "Per-nutrient min/max constraints",
                "key_options": list(NUTRIENT_IDS.keys()),
                "value_format": {
                    "min": {"type": "number", "optional": True},
                    "max": {"type": "number", "optional": True},
                },
                "example": {
                    "protein": {"min": 150},
                    "sodium": {"max": 2300},
                    "fiber": {"min": 30, "max": 50},
                },
            },
            "include_tags": {
                "type": "list[str]",
                "description": "Only use foods with at least one of these tags",
                "example": ["staple"],
            },
            "exclude_tags": {
                "type": "list[str]",
                "description": "Never use foods with any of these tags",
                "example": ["junk_food", "exclude"],
            },
            "options": {
                "type": "dict",
                "fields": {
                    "mode": {
                        "type": "enum",
                        "values": ["feasibility", "minimize_cost"],
                        "default": "feasibility",
                        "description": "feasibility = diverse foods, minimize_cost = cheapest solution",
                    },
                    "max_foods": {
                        "type": "int",
                        "default": 300,
                        "description": "Max foods to consider (randomly sampled if exceeded)",
                    },
                    "max_grams_per_food": {
                        "type": "float",
                        "default": 500,
                        "description": "Maximum grams of any single food",
                    },
                    "lambda_deviation": {
                        "type": "float",
                        "default": 0.001,
                        "description": "Weight for deviation penalty in QP mode",
                    },
                },
            },
        },
        "example_profile": {
            "calories": {"min": 1800, "max": 2200},
            "nutrients": {
                "protein": {"min": 150},
                "fiber": {"min": 30},
                "sodium": {"max": 2300},
            },
            "include_tags": ["staple"],
            "exclude_tags": ["junk_food"],
            "options": {
                "mode": "feasibility",
                "max_grams_per_food": 500,
            },
        },
    }


def get_nutrient_list() -> list[dict[str, Any]]:
    """Get list of all available nutrients with metadata.

    Returns:
        List of nutrient info dictionaries
    """
    nutrients = []
    for name, nutrient_id in sorted(NUTRIENT_IDS.items()):
        category = "macro"
        if nutrient_id in MINERAL_NUTRIENT_IDS:
            category = "mineral"
        elif nutrient_id in VITAMIN_NUTRIENT_IDS:
            category = "vitamin"
        elif nutrient_id not in MACRO_NUTRIENT_IDS:
            category = "fat"  # saturated, mono, poly, cholesterol

        nutrient_info: dict[str, Any] = {
            "name": name,
            "id": nutrient_id,
            "display_name": NUTRIENT_DISPLAY_NAMES.get(nutrient_id, name),
            "unit": NUTRIENT_UNITS.get(nutrient_id, "?"),
            "category": category,
        }

        if name in NUTRIENT_TYPICAL_RANGES:
            nutrient_info["typical_range"] = NUTRIENT_TYPICAL_RANGES[name]

        nutrients.append(nutrient_info)

    return nutrients


def get_tag_list(conn: Connection) -> list[dict[str, Any]]:
    """Get list of all tags in the database with counts.

    Args:
        conn: Database connection

    Returns:
        List of tag info dictionaries
    """
    all_tags = TagQueries.get_all_tags(conn)
    return [{"tag": tag, "count": count} for tag, count in all_tags]


def get_food_filter_schema() -> dict[str, Any]:
    """Get schema for food filtering/exploration options.

    Returns:
        Schema dictionary for food queries
    """
    return {
        "description": "Options for filtering and exploring foods",
        "filters": {
            "query": {
                "type": "str",
                "description": "Text search on food description",
            },
            "min_protein": {
                "type": "float",
                "unit": "g per 100g",
                "description": "Minimum protein content",
            },
            "max_protein": {
                "type": "float",
                "unit": "g per 100g",
                "description": "Maximum protein content",
            },
            "min_calories": {
                "type": "float",
                "unit": "kcal per 100g",
                "description": "Minimum calorie content",
            },
            "max_calories": {
                "type": "float",
                "unit": "kcal per 100g",
                "description": "Maximum calorie content",
            },
            "min_cost": {
                "type": "float",
                "unit": "$ per 100g",
                "description": "Minimum cost",
            },
            "max_cost": {
                "type": "float",
                "unit": "$ per 100g",
                "description": "Maximum cost",
            },
            "has_tag": {
                "type": "str",
                "description": "Filter to foods with this tag",
            },
            "has_price": {
                "type": "bool",
                "description": "Only foods with prices set",
            },
        },
        "example": {
            "query": "chicken",
            "min_protein": 25,
            "max_cost": 0.50,
            "has_tag": "staple",
        },
    }


def get_full_schema(conn: Optional[Connection] = None) -> dict[str, Any]:
    """Get complete schema documentation for LLM agents.

    Args:
        conn: Optional database connection (for tag list)

    Returns:
        Complete schema dictionary
    """
    schema: dict[str, Any] = {
        "version": "1.0",
        "description": "Meal planning optimization tool schema",
        "constraints": get_constraint_schema(),
        "nutrients": get_nutrient_list(),
        "food_filters": get_food_filter_schema(),
    }

    if conn:
        schema["tags"] = get_tag_list(conn)

    return schema
