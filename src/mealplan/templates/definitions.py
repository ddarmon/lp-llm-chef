"""Built-in meal templates for each dietary pattern.

Templates define the structure of meals: which food categories (slots)
should be filled and with what portion sizes. The template system
then selects specific foods to fill each slot and optimizes quantities.
"""

from __future__ import annotations

from mealplan.optimizer.multiperiod_models import MealType
from mealplan.templates.models import (
    DietTemplate,
    DiversityRule,
    MealTemplate,
    SlotDefinition,
)


# =============================================================================
# Slot Definitions (reusable building blocks)
# =============================================================================

def _protein_slot(sources: list[str], target: float = 150) -> SlotDefinition:
    """Create a protein slot with given sources."""
    return SlotDefinition(
        name="protein",
        sources=sources,
        target_grams=target,
        min_grams=50,
        max_grams=300,
    )


def _legume_slot(target: float = 125) -> SlotDefinition:
    """Create a legume slot."""
    return SlotDefinition(
        name="legume",
        sources=["legumes"],
        target_grams=target,
        min_grams=50,
        max_grams=250,
    )


def _vegetable_slot(sources: list[str] | None = None, target: float = 125, count: int = 1) -> SlotDefinition:
    """Create a vegetable slot."""
    return SlotDefinition(
        name="vegetable",
        sources=sources or ["leafy_greens", "cruciferous", "other_vegetables"],
        target_grams=target,
        min_grams=50,
        max_grams=300,
        count=count,
    )


def _fat_slot(sources: list[str] | None = None, target: float = 30) -> SlotDefinition:
    """Create a fat/nut slot."""
    return SlotDefinition(
        name="fat",
        sources=sources or ["nuts"],
        target_grams=target,
        min_grams=15,
        max_grams=75,
    )


# =============================================================================
# Pescatarian + Slow Carb Template
# =============================================================================

PESCATARIAN_SLOWCARB_TEMPLATE = DietTemplate(
    name="pescatarian_slowcarb",
    description="Fish and eggs with legumes, vegetables. No grains, dairy, or fruit.",
    meals=[
        MealTemplate(
            meal_type=MealType.BREAKFAST,
            slots=[
                _protein_slot(["eggs"], target=150),
                _legume_slot(target=100),
                _vegetable_slot(["leafy_greens", "cruciferous"], target=100),
            ],
            calorie_range=(400, 600),
            protein_min=30,
        ),
        MealTemplate(
            meal_type=MealType.LUNCH,
            slots=[
                _protein_slot(["fish"], target=175),
                _legume_slot(target=125),
                _vegetable_slot(target=150, count=2),
            ],
            calorie_range=(550, 750),
            protein_min=40,
        ),
        MealTemplate(
            meal_type=MealType.DINNER,
            slots=[
                _protein_slot(["fish"], target=175),
                _legume_slot(target=125),
                _vegetable_slot(target=150, count=2),
            ],
            calorie_range=(550, 750),
            protein_min=40,
        ),
        MealTemplate(
            meal_type=MealType.SNACK,
            slots=[
                _fat_slot(["nuts"], target=30),
            ],
            calorie_range=(100, 200),
            protein_min=5,
        ),
    ],
    diversity_rule=DiversityRule.NO_REPEAT,
    daily_calorie_range=(1800, 2200),
    daily_protein_min=150,
)


# =============================================================================
# Pescatarian Template (with grains and dairy)
# =============================================================================

PESCATARIAN_TEMPLATE = DietTemplate(
    name="pescatarian",
    description="Fish and eggs with legumes, grains, dairy, vegetables, and fruit.",
    meals=[
        MealTemplate(
            meal_type=MealType.BREAKFAST,
            slots=[
                _protein_slot(["eggs", "yogurt"], target=150),
                SlotDefinition(
                    name="carb",
                    sources=["grains", "fruits"],
                    target_grams=100,
                    min_grams=50,
                    max_grams=200,
                ),
                _vegetable_slot(["leafy_greens"], target=75, count=1),
            ],
            calorie_range=(400, 600),
            protein_min=25,
        ),
        MealTemplate(
            meal_type=MealType.LUNCH,
            slots=[
                _protein_slot(["fish"], target=175),
                _legume_slot(target=125),
                _vegetable_slot(target=150, count=2),
            ],
            calorie_range=(550, 750),
            protein_min=40,
        ),
        MealTemplate(
            meal_type=MealType.DINNER,
            slots=[
                _protein_slot(["fish", "eggs"], target=175),
                SlotDefinition(
                    name="carb",
                    sources=["grains", "legumes"],
                    target_grams=125,
                    min_grams=75,
                    max_grams=200,
                ),
                _vegetable_slot(target=150, count=2),
            ],
            calorie_range=(550, 750),
            protein_min=35,
        ),
        MealTemplate(
            meal_type=MealType.SNACK,
            slots=[
                _fat_slot(["nuts", "fruits"], target=50),
            ],
            calorie_range=(100, 200),
            protein_min=5,
        ),
    ],
    diversity_rule=DiversityRule.NO_REPEAT,
    daily_calorie_range=(1800, 2400),
    daily_protein_min=130,
)


# =============================================================================
# Keto Template (high fat, very low carb)
# =============================================================================

KETO_TEMPLATE = DietTemplate(
    name="keto",
    description="High fat, moderate protein, very low carb. No grains, legumes, or fruit.",
    meals=[
        MealTemplate(
            meal_type=MealType.BREAKFAST,
            slots=[
                _protein_slot(["eggs"], target=150),
                SlotDefinition(
                    name="fat",
                    sources=["avocado", "cheese"],
                    target_grams=75,
                    min_grams=30,
                    max_grams=150,
                ),
                _vegetable_slot(["low_carb_vegetables"], target=100),
            ],
            calorie_range=(400, 600),
            protein_min=25,
        ),
        MealTemplate(
            meal_type=MealType.LUNCH,
            slots=[
                _protein_slot(["fish", "poultry"], target=200),
                SlotDefinition(
                    name="fat",
                    sources=["oils", "nuts", "cheese"],
                    target_grams=50,
                    min_grams=20,
                    max_grams=100,
                ),
                _vegetable_slot(["low_carb_vegetables"], target=150),
            ],
            calorie_range=(600, 800),
            protein_min=40,
        ),
        MealTemplate(
            meal_type=MealType.DINNER,
            slots=[
                _protein_slot(["fish", "red_meat", "poultry"], target=200),
                SlotDefinition(
                    name="fat",
                    sources=["avocado", "oils", "cheese"],
                    target_grams=50,
                    min_grams=20,
                    max_grams=100,
                ),
                _vegetable_slot(["low_carb_vegetables"], target=150),
            ],
            calorie_range=(600, 800),
            protein_min=40,
        ),
        MealTemplate(
            meal_type=MealType.SNACK,
            slots=[
                _fat_slot(["nuts", "cheese"], target=40),
            ],
            calorie_range=(150, 300),
            protein_min=5,
        ),
    ],
    diversity_rule=DiversityRule.NO_REPEAT,
    daily_calorie_range=(1800, 2400),
    daily_protein_min=120,
)


# =============================================================================
# Vegan Template
# =============================================================================

VEGAN_TEMPLATE = DietTemplate(
    name="vegan",
    description="Plant-based only. Legumes, tofu, vegetables, grains, nuts, fruits.",
    meals=[
        MealTemplate(
            meal_type=MealType.BREAKFAST,
            slots=[
                _protein_slot(["plant_protein"], target=150),
                SlotDefinition(
                    name="carb",
                    sources=["grains", "fruits"],
                    target_grams=100,
                    min_grams=50,
                    max_grams=200,
                ),
                _vegetable_slot(["leafy_greens"], target=100),
            ],
            calorie_range=(400, 600),
            protein_min=20,
        ),
        MealTemplate(
            meal_type=MealType.LUNCH,
            slots=[
                _protein_slot(["plant_protein", "legumes"], target=175),
                _legume_slot(target=150),
                _vegetable_slot(target=150, count=2),
            ],
            calorie_range=(550, 750),
            protein_min=30,
        ),
        MealTemplate(
            meal_type=MealType.DINNER,
            slots=[
                _protein_slot(["plant_protein", "legumes"], target=175),
                SlotDefinition(
                    name="carb",
                    sources=["grains", "starchy"],
                    target_grams=150,
                    min_grams=75,
                    max_grams=250,
                ),
                _vegetable_slot(target=150, count=2),
            ],
            calorie_range=(550, 750),
            protein_min=30,
        ),
        MealTemplate(
            meal_type=MealType.SNACK,
            slots=[
                _fat_slot(["nuts", "fruits"], target=50),
            ],
            calorie_range=(100, 200),
            protein_min=5,
        ),
    ],
    diversity_rule=DiversityRule.NO_REPEAT,
    daily_calorie_range=(1800, 2400),
    daily_protein_min=100,
)


# =============================================================================
# Vegetarian Template
# =============================================================================

VEGETARIAN_TEMPLATE = DietTemplate(
    name="vegetarian",
    description="No meat or fish. Eggs, dairy, legumes, grains, vegetables.",
    meals=[
        MealTemplate(
            meal_type=MealType.BREAKFAST,
            slots=[
                _protein_slot(["eggs", "yogurt"], target=150),
                SlotDefinition(
                    name="carb",
                    sources=["grains", "fruits"],
                    target_grams=100,
                    min_grams=50,
                    max_grams=200,
                ),
                _vegetable_slot(["leafy_greens"], target=75),
            ],
            calorie_range=(400, 600),
            protein_min=25,
        ),
        MealTemplate(
            meal_type=MealType.LUNCH,
            slots=[
                _protein_slot(["eggs", "plant_protein", "cheese"], target=150),
                _legume_slot(target=150),
                _vegetable_slot(target=150, count=2),
            ],
            calorie_range=(550, 750),
            protein_min=30,
        ),
        MealTemplate(
            meal_type=MealType.DINNER,
            slots=[
                _protein_slot(["plant_protein", "eggs", "cheese"], target=150),
                SlotDefinition(
                    name="carb",
                    sources=["grains", "legumes"],
                    target_grams=150,
                    min_grams=75,
                    max_grams=250,
                ),
                _vegetable_slot(target=150, count=2),
            ],
            calorie_range=(550, 750),
            protein_min=30,
        ),
        MealTemplate(
            meal_type=MealType.SNACK,
            slots=[
                _fat_slot(["nuts", "yogurt", "fruits"], target=50),
            ],
            calorie_range=(100, 200),
            protein_min=5,
        ),
    ],
    diversity_rule=DiversityRule.NO_REPEAT,
    daily_calorie_range=(1800, 2400),
    daily_protein_min=100,
)


# =============================================================================
# Mediterranean Template
# =============================================================================

MEDITERRANEAN_TEMPLATE = DietTemplate(
    name="mediterranean",
    description="Fish, olive oil, legumes, whole grains, vegetables, fruits, nuts.",
    meals=[
        MealTemplate(
            meal_type=MealType.BREAKFAST,
            slots=[
                _protein_slot(["eggs", "yogurt"], target=125),
                SlotDefinition(
                    name="carb",
                    sources=["grains", "fruits"],
                    target_grams=100,
                    min_grams=50,
                    max_grams=200,
                ),
                _vegetable_slot(["leafy_greens", "other_vegetables"], target=75),
            ],
            calorie_range=(350, 550),
            protein_min=20,
        ),
        MealTemplate(
            meal_type=MealType.LUNCH,
            slots=[
                _protein_slot(["fish"], target=150),
                _legume_slot(target=125),
                _vegetable_slot(target=150, count=2),
                SlotDefinition(
                    name="fat",
                    sources=["oils"],
                    target_grams=15,
                    min_grams=10,
                    max_grams=30,
                ),
            ],
            calorie_range=(550, 750),
            protein_min=35,
        ),
        MealTemplate(
            meal_type=MealType.DINNER,
            slots=[
                _protein_slot(["fish", "poultry"], target=150),
                SlotDefinition(
                    name="carb",
                    sources=["grains", "legumes"],
                    target_grams=150,
                    min_grams=75,
                    max_grams=250,
                ),
                _vegetable_slot(target=150, count=2),
                SlotDefinition(
                    name="fat",
                    sources=["oils"],
                    target_grams=15,
                    min_grams=10,
                    max_grams=30,
                ),
            ],
            calorie_range=(550, 750),
            protein_min=35,
        ),
        MealTemplate(
            meal_type=MealType.SNACK,
            slots=[
                _fat_slot(["nuts", "fruits"], target=50),
            ],
            calorie_range=(100, 200),
            protein_min=5,
        ),
    ],
    diversity_rule=DiversityRule.NO_REPEAT,
    daily_calorie_range=(1800, 2400),
    daily_protein_min=100,
)


# =============================================================================
# Paleo Template
# =============================================================================

PALEO_TEMPLATE = DietTemplate(
    name="paleo",
    description="Meat, fish, eggs, vegetables, fruits, nuts. No grains, legumes, or dairy.",
    meals=[
        MealTemplate(
            meal_type=MealType.BREAKFAST,
            slots=[
                _protein_slot(["eggs"], target=150),
                SlotDefinition(
                    name="carb",
                    sources=["fruits", "starchy"],
                    target_grams=100,
                    min_grams=50,
                    max_grams=200,
                ),
                _vegetable_slot(["leafy_greens"], target=100),
            ],
            calorie_range=(400, 600),
            protein_min=25,
        ),
        MealTemplate(
            meal_type=MealType.LUNCH,
            slots=[
                _protein_slot(["fish", "poultry"], target=200),
                _vegetable_slot(target=200, count=2),
                SlotDefinition(
                    name="carb",
                    sources=["starchy", "fruits"],
                    target_grams=100,
                    min_grams=50,
                    max_grams=150,
                ),
            ],
            calorie_range=(550, 750),
            protein_min=40,
        ),
        MealTemplate(
            meal_type=MealType.DINNER,
            slots=[
                _protein_slot(["red_meat", "fish", "poultry"], target=200),
                _vegetable_slot(target=200, count=2),
                SlotDefinition(
                    name="fat",
                    sources=["avocado", "oils"],
                    target_grams=30,
                    min_grams=15,
                    max_grams=75,
                ),
            ],
            calorie_range=(550, 750),
            protein_min=40,
        ),
        MealTemplate(
            meal_type=MealType.SNACK,
            slots=[
                _fat_slot(["nuts", "fruits"], target=50),
            ],
            calorie_range=(100, 200),
            protein_min=5,
        ),
    ],
    diversity_rule=DiversityRule.NO_REPEAT,
    daily_calorie_range=(1800, 2400),
    daily_protein_min=130,
)


# =============================================================================
# Template Registry
# =============================================================================

TEMPLATE_REGISTRY: dict[str, DietTemplate] = {
    "pescatarian_slowcarb": PESCATARIAN_SLOWCARB_TEMPLATE,
    "pescatarian": PESCATARIAN_TEMPLATE,
    "keto": KETO_TEMPLATE,
    "vegan": VEGAN_TEMPLATE,
    "vegetarian": VEGETARIAN_TEMPLATE,
    "mediterranean": MEDITERRANEAN_TEMPLATE,
    "paleo": PALEO_TEMPLATE,
}

# Combination mappings for common pattern combinations
COMBINATION_TEMPLATES: dict[frozenset[str], str] = {
    frozenset(["pescatarian", "slow_carb"]): "pescatarian_slowcarb",
    frozenset(["slow_carb", "pescatarian"]): "pescatarian_slowcarb",
}


def get_template(name: str) -> DietTemplate | None:
    """Get a template by name.

    Args:
        name: Template name (e.g., "pescatarian", "keto")

    Returns:
        DietTemplate or None if not found.
    """
    return TEMPLATE_REGISTRY.get(name.lower())


def get_template_for_patterns(patterns: list[str]) -> DietTemplate | None:
    """Get the best template for a combination of dietary patterns.

    Args:
        patterns: List of pattern names (e.g., ["pescatarian", "slow_carb"])

    Returns:
        DietTemplate or None if no suitable template exists.
    """
    # Check for exact combination match
    pattern_set = frozenset(p.lower() for p in patterns)
    if pattern_set in COMBINATION_TEMPLATES:
        return get_template(COMBINATION_TEMPLATES[pattern_set])

    # Single pattern - direct lookup
    if len(patterns) == 1:
        return get_template(patterns[0])

    # No combination template exists
    return None


def list_available_templates() -> list[str]:
    """Return list of available template names."""
    return list(TEMPLATE_REGISTRY.keys())
