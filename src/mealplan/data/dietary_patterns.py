"""Dietary pattern definitions for automatic food pool filtering.

Pre-defined dietary patterns (pescatarian, vegetarian, vegan, keto, slow_carb,
mediterranean) with include/exclude rules to automatically filter the USDA food
database without manual SQL.

Usage:
    from mealplan.data.dietary_patterns import DIETARY_PATTERNS, get_pattern
    pattern = get_pattern("pescatarian")
    # Use pattern.include_keywords, pattern.exclude_keywords for filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MacroTargets:
    """Macronutrient targets for a dietary pattern.

    Values are percentages of total calories (0.0 to 1.0 range).
    """

    fat_min: Optional[float] = None
    fat_max: Optional[float] = None
    carb_min: Optional[float] = None
    carb_max: Optional[float] = None
    protein_min: Optional[float] = None
    protein_max: Optional[float] = None
    carb_grams_max: Optional[float] = None  # For keto-style absolute limits


@dataclass
class DietaryPattern:
    """Definition of a dietary pattern for food filtering.

    Attributes:
        name: Pattern identifier (e.g., "pescatarian")
        description: Human-readable description
        include_keywords: Keywords that foods SHOULD contain (OR logic)
        exclude_keywords: Keywords that foods must NOT contain
        include_categories: USDA food categories to include (if set, others excluded)
        exclude_categories: USDA food categories to exclude
        required_categories: Categories that MUST have foods in the pool
        macro_targets: Optional macronutrient targets
        emphasize_keywords: Keywords for foods to prefer (soft constraint)
        limit_keywords: Keywords for foods to minimize (soft constraint)
    """

    name: str
    description: str
    include_keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    include_categories: list[str] = field(default_factory=list)
    exclude_categories: list[str] = field(default_factory=list)
    required_categories: list[str] = field(default_factory=list)
    macro_targets: Optional[MacroTargets] = None
    emphasize_keywords: list[str] = field(default_factory=list)
    limit_keywords: list[str] = field(default_factory=list)


# Common exclusion keyword sets for reuse
MEAT_KEYWORDS = [
    "beef", "pork", "chicken", "turkey", "lamb", "veal", "duck", "goose",
    "bacon", "ham", "sausage", "hot dog", "frankfurter", "pepperoni",
    "salami", "prosciutto", "chorizo", "deli meat", "bologna", "pastrami",
    "roast beef", "ground beef", "ground pork", "ground turkey",
    "ground chicken", "steak", "rib", "brisket", "tenderloin",
    "venison", "bison", "game", "rabbit", "goat",
]

FISH_KEYWORDS = [
    "fish", "salmon", "tuna", "cod", "tilapia", "halibut", "sardine",
    "anchovy", "mackerel", "trout", "bass", "haddock", "herring",
    "shrimp", "crab", "lobster", "clam", "mussel", "oyster", "scallop",
    "squid", "calamari", "octopus", "seafood", "caviar", "roe",
]

DAIRY_KEYWORDS = [
    "milk", "cheese", "yogurt", "cream", "butter", "whey", "casein",
    "cottage cheese", "ricotta", "mozzarella", "cheddar", "parmesan",
    "sour cream", "cream cheese", "ice cream", "gelato",
]

EGG_KEYWORDS = [
    "egg", "omelette", "omelet", "scrambled egg", "fried egg",
]

REFINED_CARB_KEYWORDS = [
    "bread", "pasta", "rice", "noodle", "cracker", "cereal",
    "bagel", "muffin", "croissant", "pastry", "cookie", "cake",
    "pie", "donut", "doughnut", "pancake", "waffle", "tortilla",
    "white flour", "wheat flour", "refined flour",
]

SUGAR_KEYWORDS = [
    "sugar", "syrup", "honey", "molasses", "candy", "chocolate",
    "soda", "cola", "sweetened", "dessert", "frosting", "icing",
    "ice cream", "gelato", "sorbet", "jam", "jelly", "marmalade",
]

FRUIT_KEYWORDS = [
    "apple", "banana", "orange", "grape", "berry", "strawberry",
    "blueberry", "raspberry", "blackberry", "mango", "pineapple",
    "peach", "pear", "plum", "cherry", "melon", "watermelon",
    "cantaloupe", "honeydew", "kiwi", "papaya", "guava",
]

LEGUME_KEYWORDS = [
    "bean", "lentil", "chickpea", "pea", "hummus", "pinto",
    "black bean", "kidney", "navy bean", "lima", "edamame",
    "soybean", "tofu", "tempeh", "dal", "fava",
]

VEGETABLE_KEYWORDS = [
    "spinach", "kale", "broccoli", "cauliflower", "lettuce", "cabbage",
    "carrot", "celery", "cucumber", "pepper", "tomato", "onion",
    "garlic", "asparagus", "zucchini", "squash", "eggplant", "mushroom",
    "artichoke", "beet", "radish", "turnip", "chard", "collard",
    "arugula", "brussels sprout", "bok choy", "fennel",
]

NUT_KEYWORDS = [
    "almond", "walnut", "cashew", "peanut", "pecan", "pistachio",
    "macadamia", "hazelnut", "brazil nut", "pine nut", "nut butter",
]


# ============================================================================
# Dietary Pattern Definitions
# ============================================================================

PESCATARIAN = DietaryPattern(
    name="pescatarian",
    description="Fish and seafood plus vegetarian foods (no other meat)",
    include_keywords=FISH_KEYWORDS + EGG_KEYWORDS + DAIRY_KEYWORDS + LEGUME_KEYWORDS + VEGETABLE_KEYWORDS,
    exclude_keywords=MEAT_KEYWORDS,
    include_categories=["fish", "seafood", "eggs", "dairy", "legumes", "vegetables", "grains", "nuts"],
)

VEGETARIAN = DietaryPattern(
    name="vegetarian",
    description="No meat or fish, includes eggs and dairy",
    include_keywords=EGG_KEYWORDS + DAIRY_KEYWORDS + LEGUME_KEYWORDS + VEGETABLE_KEYWORDS,
    exclude_keywords=MEAT_KEYWORDS + FISH_KEYWORDS,
    include_categories=["eggs", "dairy", "legumes", "vegetables", "grains", "fruits", "nuts"],
)

VEGAN = DietaryPattern(
    name="vegan",
    description="Plant-based only, no animal products",
    include_keywords=LEGUME_KEYWORDS + VEGETABLE_KEYWORDS + NUT_KEYWORDS,
    exclude_keywords=MEAT_KEYWORDS + FISH_KEYWORDS + DAIRY_KEYWORDS + EGG_KEYWORDS + ["honey", "gelatin"],
    include_categories=["legumes", "vegetables", "grains", "fruits", "nuts"],
)

KETO = DietaryPattern(
    name="keto",
    description="Very low carbohydrate, high fat diet",
    exclude_keywords=REFINED_CARB_KEYWORDS + SUGAR_KEYWORDS + FRUIT_KEYWORDS + [
        "potato", "corn", "bean", "lentil", "chickpea",  # High-carb foods
    ],
    emphasize_keywords=["olive oil", "avocado", "coconut", "butter", "cheese", "egg", "fish", "meat"],
    macro_targets=MacroTargets(
        fat_min=0.65,
        fat_max=0.80,
        carb_max=0.10,  # Max 10% calories from carbs
        protein_min=0.15,
        protein_max=0.25,
        carb_grams_max=50.0,  # Absolute limit of 50g carbs
    ),
)

SLOW_CARB = DietaryPattern(
    name="slow_carb",
    description="Tim Ferriss 4-Hour Body style: protein, legumes, vegetables only",
    include_keywords=LEGUME_KEYWORDS + VEGETABLE_KEYWORDS + MEAT_KEYWORDS + FISH_KEYWORDS + EGG_KEYWORDS,
    exclude_keywords=REFINED_CARB_KEYWORDS + SUGAR_KEYWORDS + FRUIT_KEYWORDS + DAIRY_KEYWORDS + [
        "potato", "sweet potato", "corn", "white rice",
    ],
    required_categories=["legumes"],  # Must have legumes at each meal
    emphasize_keywords=["lentil", "black bean", "pinto bean", "spinach", "broccoli"],
)

MEDITERRANEAN = DietaryPattern(
    name="mediterranean",
    description="Traditional Mediterranean diet: fish, olive oil, vegetables, whole grains",
    emphasize_keywords=[
        "olive oil", "fish", "salmon", "sardine", "legume", "bean", "lentil",
        "nut", "almond", "walnut", "tomato", "leafy green", "whole grain",
    ],
    limit_keywords=[
        "red meat", "beef", "pork", "processed", "sausage", "bacon",
        "sugar", "sweetened", "refined",
    ],
    include_categories=["fish", "seafood", "legumes", "vegetables", "fruits", "nuts", "grains"],
)

PALEO = DietaryPattern(
    name="paleo",
    description="Paleolithic-style: meat, fish, vegetables, fruits, nuts (no grains/legumes/dairy)",
    include_keywords=MEAT_KEYWORDS + FISH_KEYWORDS + VEGETABLE_KEYWORDS + FRUIT_KEYWORDS + NUT_KEYWORDS,
    exclude_keywords=REFINED_CARB_KEYWORDS + SUGAR_KEYWORDS + DAIRY_KEYWORDS + LEGUME_KEYWORDS + [
        "grain", "wheat", "oat", "corn", "rice", "quinoa", "barley",
    ],
)

WHOLE30 = DietaryPattern(
    name="whole30",
    description="Whole foods only: no sugar, alcohol, grains, legumes, dairy",
    include_keywords=MEAT_KEYWORDS + FISH_KEYWORDS + VEGETABLE_KEYWORDS + FRUIT_KEYWORDS + NUT_KEYWORDS,
    exclude_keywords=SUGAR_KEYWORDS + DAIRY_KEYWORDS + LEGUME_KEYWORDS + REFINED_CARB_KEYWORDS + [
        "alcohol", "wine", "beer", "liquor", "soy", "peanut",
    ],
)


# ============================================================================
# Pattern Registry and Access Functions
# ============================================================================

DIETARY_PATTERNS: dict[str, DietaryPattern] = {
    "pescatarian": PESCATARIAN,
    "vegetarian": VEGETARIAN,
    "vegan": VEGAN,
    "keto": KETO,
    "slow_carb": SLOW_CARB,
    "mediterranean": MEDITERRANEAN,
    "paleo": PALEO,
    "whole30": WHOLE30,
}


def get_pattern(name: str) -> Optional[DietaryPattern]:
    """Get a dietary pattern by name.

    Args:
        name: Pattern name (case-insensitive, underscores and hyphens equivalent)

    Returns:
        DietaryPattern if found, None otherwise
    """
    # Normalize name: lowercase, convert hyphens to underscores
    normalized = name.lower().replace("-", "_")
    return DIETARY_PATTERNS.get(normalized)


def list_patterns() -> list[dict]:
    """List all available dietary patterns.

    Returns:
        List of pattern info dicts with name and description
    """
    return [
        {"name": p.name, "description": p.description}
        for p in DIETARY_PATTERNS.values()
    ]


def combine_patterns(*pattern_names: str) -> DietaryPattern:
    """Combine multiple dietary patterns into one.

    Rules for combining:
    - exclude_keywords: Union of all patterns (if any pattern excludes it, it's excluded)
    - include_keywords: Intersection approach - only include if not excluded
    - required_categories: Union of all patterns
    - macro_targets: Uses the most restrictive from any pattern

    Args:
        *pattern_names: Names of patterns to combine

    Returns:
        New DietaryPattern combining all specified patterns

    Raises:
        ValueError: If any pattern name is unknown
    """
    patterns = []
    for name in pattern_names:
        pattern = get_pattern(name)
        if pattern is None:
            raise ValueError(f"Unknown dietary pattern: {name}")
        patterns.append(pattern)

    if not patterns:
        raise ValueError("At least one pattern must be specified")

    if len(patterns) == 1:
        return patterns[0]

    # Combine exclude keywords (union)
    combined_exclude = set()
    for p in patterns:
        combined_exclude.update(p.exclude_keywords)

    # Combine include keywords (union, but remove any that are excluded)
    combined_include = set()
    for p in patterns:
        combined_include.update(p.include_keywords)
    combined_include -= combined_exclude

    # Combine required categories (union)
    combined_required = set()
    for p in patterns:
        combined_required.update(p.required_categories)

    # Combine macro targets (most restrictive)
    combined_macros = None
    for p in patterns:
        if p.macro_targets is not None:
            if combined_macros is None:
                combined_macros = MacroTargets(
                    fat_min=p.macro_targets.fat_min,
                    fat_max=p.macro_targets.fat_max,
                    carb_min=p.macro_targets.carb_min,
                    carb_max=p.macro_targets.carb_max,
                    protein_min=p.macro_targets.protein_min,
                    protein_max=p.macro_targets.protein_max,
                    carb_grams_max=p.macro_targets.carb_grams_max,
                )
            else:
                # Take most restrictive values
                if p.macro_targets.fat_min is not None:
                    combined_macros.fat_min = max(
                        combined_macros.fat_min or 0, p.macro_targets.fat_min
                    )
                if p.macro_targets.fat_max is not None:
                    combined_macros.fat_max = min(
                        combined_macros.fat_max or 1.0, p.macro_targets.fat_max
                    )
                if p.macro_targets.carb_max is not None:
                    combined_macros.carb_max = min(
                        combined_macros.carb_max or 1.0, p.macro_targets.carb_max
                    )
                if p.macro_targets.carb_grams_max is not None:
                    combined_macros.carb_grams_max = min(
                        combined_macros.carb_grams_max or float("inf"),
                        p.macro_targets.carb_grams_max,
                    )
                if p.macro_targets.protein_min is not None:
                    combined_macros.protein_min = max(
                        combined_macros.protein_min or 0, p.macro_targets.protein_min
                    )
                if p.macro_targets.protein_max is not None:
                    combined_macros.protein_max = min(
                        combined_macros.protein_max or 1.0, p.macro_targets.protein_max
                    )

    # Combine emphasize/limit keywords (union)
    combined_emphasize = set()
    combined_limit = set()
    for p in patterns:
        combined_emphasize.update(p.emphasize_keywords)
        combined_limit.update(p.limit_keywords)

    return DietaryPattern(
        name="+".join(pattern_names),
        description=f"Combined: {', '.join(pattern_names)}",
        include_keywords=list(combined_include),
        exclude_keywords=list(combined_exclude),
        required_categories=list(combined_required),
        macro_targets=combined_macros,
        emphasize_keywords=list(combined_emphasize),
        limit_keywords=list(combined_limit),
    )
