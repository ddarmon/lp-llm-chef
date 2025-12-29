"""USDA Nutrient ID constants and lookup utilities.

These IDs correspond to the USDA FoodData Central nutrient database.
All nutrient values are stored per 100g of food.
"""

from __future__ import annotations

# Mapping from friendly names to USDA nutrient IDs
NUTRIENT_IDS: dict[str, int] = {
    # Macros
    "energy": 1008,  # kcal
    "protein": 1003,  # g
    "total_fat": 1004,  # g
    "carbohydrate": 1005,  # g
    "fiber": 1079,  # g
    "sugar": 2000,  # g
    # Fats breakdown
    "saturated_fat": 1258,  # g
    "monounsaturated": 1292,  # g
    "polyunsaturated": 1293,  # g
    "cholesterol": 1253,  # mg
    # Minerals
    "calcium": 1087,  # mg
    "iron": 1089,  # mg
    "magnesium": 1090,  # mg
    "phosphorus": 1091,  # mg
    "potassium": 1092,  # mg
    "sodium": 1093,  # mg
    "zinc": 1095,  # mg
    "copper": 1098,  # mg
    "manganese": 1101,  # mg
    "selenium": 1103,  # mcg
    # Vitamins
    "vitamin_a": 1106,  # mcg RAE
    "vitamin_c": 1162,  # mg
    "vitamin_d": 1114,  # mcg
    "vitamin_e": 1109,  # mg
    "vitamin_k": 1185,  # mcg
    "vitamin_b6": 1175,  # mg
    "vitamin_b12": 1178,  # mcg
    "folate": 1177,  # mcg
    "thiamin": 1165,  # mg (B1)
    "riboflavin": 1166,  # mg (B2)
    "niacin": 1167,  # mg (B3)
    "pantothenic_acid": 1170,  # mg (B5)
    "choline": 1180,  # mg
}

# Reverse mapping: nutrient ID to name
NUTRIENT_NAMES: dict[int, str] = {v: k for k, v in NUTRIENT_IDS.items()}

# Nutrient units
NUTRIENT_UNITS: dict[int, str] = {
    # Macros
    1008: "kcal",
    1003: "g",
    1004: "g",
    1005: "g",
    1079: "g",
    2000: "g",
    # Fats
    1258: "g",
    1292: "g",
    1293: "g",
    1253: "mg",
    # Minerals
    1087: "mg",
    1089: "mg",
    1090: "mg",
    1091: "mg",
    1092: "mg",
    1093: "mg",
    1095: "mg",
    1098: "mg",
    1101: "mg",
    1103: "mcg",
    # Vitamins
    1106: "mcg",
    1162: "mg",
    1114: "mcg",
    1109: "mg",
    1185: "mcg",
    1175: "mg",
    1178: "mcg",
    1177: "mcg",
    1165: "mg",
    1166: "mg",
    1167: "mg",
    1170: "mg",
    1180: "mg",
}

# Display names for nutrients (more user-friendly)
NUTRIENT_DISPLAY_NAMES: dict[int, str] = {
    1008: "Calories",
    1003: "Protein",
    1004: "Total Fat",
    1005: "Carbohydrates",
    1079: "Fiber",
    2000: "Sugar",
    1258: "Saturated Fat",
    1292: "Monounsaturated Fat",
    1293: "Polyunsaturated Fat",
    1253: "Cholesterol",
    1087: "Calcium",
    1089: "Iron",
    1090: "Magnesium",
    1091: "Phosphorus",
    1092: "Potassium",
    1093: "Sodium",
    1095: "Zinc",
    1098: "Copper",
    1101: "Manganese",
    1103: "Selenium",
    1106: "Vitamin A",
    1162: "Vitamin C",
    1114: "Vitamin D",
    1109: "Vitamin E",
    1185: "Vitamin K",
    1175: "Vitamin B6",
    1178: "Vitamin B12",
    1177: "Folate",
    1165: "Thiamin (B1)",
    1166: "Riboflavin (B2)",
    1167: "Niacin (B3)",
    1170: "Pantothenic Acid (B5)",
    1180: "Choline",
}

# Set of macro nutrient IDs
MACRO_NUTRIENT_IDS: set[int] = {1008, 1003, 1004, 1005, 1079, 2000}

# Set of mineral IDs
MINERAL_NUTRIENT_IDS: set[int] = {1087, 1089, 1090, 1091, 1092, 1093, 1095, 1098, 1101, 1103}

# Set of vitamin IDs
VITAMIN_NUTRIENT_IDS: set[int] = {
    1106,
    1162,
    1114,
    1109,
    1185,
    1175,
    1178,
    1177,
    1165,
    1166,
    1167,
    1170,
    1180,
}


def get_nutrient_id(name: str) -> int:
    """Look up a nutrient ID by name.

    Args:
        name: Nutrient name (e.g., 'protein', 'vitamin_c')

    Returns:
        USDA nutrient ID

    Raises:
        KeyError: If nutrient name not found
    """
    name_lower = name.lower().replace(" ", "_").replace("-", "_")
    if name_lower not in NUTRIENT_IDS:
        raise KeyError(
            f"Unknown nutrient: {name}. "
            f"Available nutrients: {', '.join(sorted(NUTRIENT_IDS.keys()))}"
        )
    return NUTRIENT_IDS[name_lower]


def get_nutrient_name(nutrient_id: int) -> str:
    """Look up a nutrient name by ID.

    Args:
        nutrient_id: USDA nutrient ID

    Returns:
        Nutrient name

    Raises:
        KeyError: If nutrient ID not found
    """
    if nutrient_id not in NUTRIENT_NAMES:
        raise KeyError(f"Unknown nutrient ID: {nutrient_id}")
    return NUTRIENT_NAMES[nutrient_id]


def get_nutrient_unit(nutrient_id: int) -> str:
    """Get the unit for a nutrient.

    Args:
        nutrient_id: USDA nutrient ID

    Returns:
        Unit string (e.g., 'g', 'mg', 'mcg', 'kcal')
    """
    return NUTRIENT_UNITS.get(nutrient_id, "?")


def get_nutrient_display_name(nutrient_id: int) -> str:
    """Get a user-friendly display name for a nutrient.

    Args:
        nutrient_id: USDA nutrient ID

    Returns:
        Display name string
    """
    return NUTRIENT_DISPLAY_NAMES.get(nutrient_id, f"Nutrient {nutrient_id}")
