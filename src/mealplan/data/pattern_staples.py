"""Curated staple food lists for each dietary pattern.

These lists contain FDC IDs of common, grocery-store-available foods
that are appropriate for each dietary pattern. Using these curated lists
instead of keyword filtering prevents exotic foods (like "winged beans")
from appearing in meal plans.
"""

from __future__ import annotations

# =============================================================================
# Shared Food Categories (used as building blocks for patterns)
# =============================================================================

# Common fish and seafood
FISH_STAPLES = [
    # Salmon
    175167,  # Fish, salmon, Atlantic, farmed, raw
    175168,  # Fish, salmon, Atlantic, farmed, cooked, dry heat
    173686,  # Fish, salmon, Atlantic, wild, raw
    171998,  # Fish, salmon, Atlantic, wild, cooked, dry heat
    173689,  # Fish, salmon, chum, raw
    175136,  # Fish, salmon, coho, wild, raw
    173724,  # Fish, salmon, pink, canned, drained solids
    # Tuna
    175159,  # Fish, tuna, fresh, yellowfin, raw
    173709,  # Fish, tuna, light, canned in water, drained solids
    171986,  # Fish, tuna, light, canned in water, without salt
    175158,  # Fish, tuna, white, canned in water, drained solids
    # Cod
    171955,  # Fish, cod, Atlantic, raw
    171956,  # Fish, cod, Atlantic, cooked, dry heat
    175178,  # Fish, cod, Pacific, cooked
    # Other fish
    175176,  # Fish, tilapia, raw
    175177,  # Fish, tilapia, cooked, dry heat
    174200,  # Fish, halibut, Atlantic and Pacific, raw
    174201,  # Fish, halibut, Atlantic and Pacific, cooked
    173717,  # Fish, trout, rainbow, farmed, raw
    175155,  # Fish, trout, rainbow, wild, cooked, dry heat
    175119,  # Fish, mackerel, Atlantic, raw
    175120,  # Fish, mackerel, Atlantic, cooked, dry heat
    175139,  # Fish, sardine, Atlantic, canned in oil
    # Shellfish
    175179,  # Crustaceans, shrimp, raw
    175180,  # Crustaceans, shrimp, cooked
    174210,  # Crustaceans, shrimp, mixed species, raw
]

# Eggs
EGG_STAPLES = [
    171287,  # Egg, whole, raw, fresh
    172187,  # Egg, whole, cooked, scrambled
    173424,  # Egg, whole, cooked, hard-boiled
    173423,  # Egg, whole, cooked, fried
    172185,  # Egg, whole, cooked, omelet
    172186,  # Egg, whole, cooked, poached
    172183,  # Egg, white, raw, fresh
]

# Chicken
CHICKEN_STAPLES = [
    171077,  # Chicken, broiler or fryers, breast, skinless, boneless, meat only, raw
    171140,  # Chicken, broiler or fryers, breast, skinless, boneless, meat only, cooked, braised
    171534,  # Chicken, broiler or fryers, breast, skinless, boneless, meat only, cooked, grilled
    171445,  # Chicken, broiler, rotisserie, BBQ, breast, meat only
    171118,  # Chicken, broiler, rotisserie, BBQ, thigh, meat only
    171114,  # Chicken, broiler, rotisserie, BBQ, drumstick, meat only
]

# Beef
BEEF_STAPLES = [
    174036,  # Beef, ground, 80% lean meat / 20% fat, raw
    174037,  # Beef, ground, 75% lean meat / 25% fat, raw
    171797,  # Beef, ground, 80% lean meat / 20% fat, patty, cooked, broiled
    171733,  # Beef, Australian, imported, grass-fed, loin, top loin steak/roast, boneless, raw
    173965,  # Beef, Australian, imported, grass-fed, loin, tenderloin, lean only, raw
]

# Pork
PORK_STAPLES = [
    167902,  # Pork, fresh, ground, raw
    167903,  # Pork, fresh, ground, cooked
    168361,  # Pork, fresh, enhanced, loin, tenderloin, separable lean only, raw
    167914,  # Pork, cured, bacon, cooked, baked
    168322,  # Pork, cured, bacon, pre-sliced, cooked, pan-fried
]

# Turkey
TURKEY_STAPLES = [
    172876,  # Turkey, breast, from whole bird, meat only, raw
    174519,  # Turkey, breast, from whole bird, meat only, roasted
]

# Lamb
LAMB_STAPLES = [
    172608,  # Lamb, Australian, ground, 85% lean / 15% fat, raw
    174397,  # Lamb, Australian, imported, fresh, composite, lean and fat, raw
    174399,  # Lamb, Australian, imported, fresh, composite, lean only, raw
]

# Common legumes
LEGUME_STAPLES = [
    173735,  # Beans, black, mature seeds, cooked, boiled, without salt
    175187,  # Beans, black turtle, mature seeds, cooked, boiled, without salt
    173757,  # Chickpeas (garbanzo beans), mature seeds, cooked, boiled, without salt
    173800,  # Chickpeas (garbanzo beans), mature seeds, canned, drained solids
    172421,  # Lentils, mature seeds, cooked, boiled, without salt
    172420,  # Lentils, raw
    174284,  # Lentils, pink or red, raw
    173740,  # Beans, kidney, all types, mature seeds, cooked, boiled, without salt
    175191,  # Beans, great northern, mature seeds, cooked, boiled, without salt
    173790,  # Beans, great northern, mature seeds, cooked, boiled, with salt
    175182,  # Beans, baked, canned, plain or vegetarian
    168410,  # Edamame, frozen, unprepared
    168411,  # Edamame, frozen, prepared
]

# =============================================================================
# Vegetable Sub-Categories (for template-based meal composition)
# =============================================================================

# Leafy greens - high volume, low calorie
LEAFY_GREENS = [
    168462,  # Spinach, raw
    170530,  # Spinach, cooked, boiled, drained, with salt
    1999632, # Spinach, baby
    168421,  # Kale, raw
    169238,  # Kale, cooked, boiled, drained, without salt
    169247,  # Lettuce, cos or romaine, raw
    169248,  # Lettuce, iceberg (includes crisphead types), raw
    168429,  # Lettuce, butterhead (includes boston and bibb types), raw
    169249,  # Lettuce, green leaf, raw
]

# Cruciferous vegetables - broccoli, cauliflower, etc.
CRUCIFEROUS = [
    170379,  # Broccoli, raw
    169967,  # Broccoli, cooked, boiled, drained, without salt
    169986,  # Cauliflower, raw
    170397,  # Cauliflower, cooked, boiled, drained, without salt
    170383,  # Brussels sprouts, raw
    169971,  # Brussels sprouts, cooked, boiled, drained, without salt
    2346407, # Cabbage, green, raw
    169976,  # Cabbage, cooked, boiled, drained, without salt
    170390,  # Cabbage, chinese (pak-choi), raw
]

# Other non-starchy vegetables
OTHER_VEGETABLES = [
    168389,  # Asparagus, raw
    168390,  # Asparagus, cooked, boiled, drained
    169291,  # Squash, summer, zucchini, includes skin, raw
    169292,  # Squash, summer, zucchini, includes skin, cooked, boiled
    169961,  # Beans, snap, green, raw
    169141,  # Beans, snap, green, cooked, boiled, drained, without salt
    170000,  # Onions, raw
    170001,  # Onions, cooked, boiled, drained, without salt
    168568,  # Carrots, baby, raw
    170394,  # Carrots, cooked, boiled, drained, without salt
    169988,  # Celery, raw
    168409,  # Cucumber, with peel, raw
    170457,  # Tomatoes, red, ripe, raw, year round average
    2258588, # Peppers, bell, green, raw
    2258590, # Peppers, bell, red, raw
    169228,  # Eggplant, raw
    169229,  # Eggplant, cooked, boiled, drained, without salt
]

# Mushrooms
MUSHROOMS = [
    168434,  # Mushrooms, brown, italian, or crimini, raw
    2003598, # Mushroom, portabella
    169376,  # Mushroom, white, exposed to ultraviolet light, raw
]

# Combined vegetable list (for backward compatibility)
VEGETABLE_STAPLES = LEAFY_GREENS + CRUCIFEROUS + OTHER_VEGETABLES + MUSHROOMS

# Common nuts and seeds
NUT_STAPLES = [
    170567,  # Nuts, almonds
    170158,  # Nuts, almonds, dry roasted, without salt added
    170187,  # Nuts, walnuts, english
    170162,  # Nuts, cashew nuts, raw
    170571,  # Nuts, cashew nuts, dry roasted, without salt added
    172430,  # Peanuts, all types, raw
    173806,  # Peanuts, all types, dry-roasted, without salt
    170178,  # Nuts, macadamia nuts, raw
    170182,  # Nuts, pecans
    170181,  # Nuts, mixed nuts, dry roasted, with peanuts, salt added
    170588,  # Nuts, mixed nuts, oil roasted, without peanuts, without salt added
    # Seeds
    170562,  # Seeds, sunflower seed kernels, dried
    170563,  # Seeds, sunflower seed kernels, dry roasted, without salt
    170554,  # Seeds, chia seeds, dried
    169414,  # Seeds, flaxseed
    2515380, # Seeds, pumpkin seeds (pepitas), raw
]

# Common dairy
DAIRY_STAPLES = [
    # Yogurt
    170894,  # Yogurt, Greek, plain, nonfat
    171304,  # Yogurt, Greek, plain, whole milk
    170903,  # Yogurt, Greek, plain, lowfat
    170905,  # Yogurt, Greek, fruit, whole milk
    # Cheese
    328637,  # Cheese, cheddar
    170845,  # Cheese, mozzarella, whole milk
    171244,  # Cheese, mozzarella, low moisture, part-skim
    172179,  # Cheese, cottage, creamed, large or small curd
    173417,  # Cheese, cottage, lowfat, 1% milkfat
    172182,  # Cheese, cottage, lowfat, 2% milkfat
    173418,  # Cheese, cream
    172175,  # Cheese, blue
    # Butter
    173410,  # Butter, salted
    789828,  # Butter, stick, unsalted
]

# Healthy fats and oils
FAT_STAPLES = [
    171413,  # Oil, olive, salad or cooking
    748608,  # Oil, olive, extra virgin
    171412,  # Oil, coconut
    173573,  # Oil, avocado
    171705,  # Avocados, raw, all commercial varieties
    171706,  # Avocados, raw, California
]

# Tofu and tempeh (plant proteins)
PLANT_PROTEIN_STAPLES = [
    172448,  # Tofu, firm, prepared with calcium sulfate and magnesium chloride
    172475,  # Tofu, raw, firm, prepared with calcium sulfate
    172476,  # Tofu, raw, regular, prepared with calcium sulfate
    174272,  # Tempeh
    172467,  # Tempeh, cooked
]

# Whole grains (for patterns that allow them)
GRAIN_STAPLES = [
    168917,  # Quinoa, cooked
    168874,  # Quinoa, uncooked
    169704,  # Rice, brown, long-grain, cooked
    169703,  # Rice, brown, long-grain, raw
    168878,  # Rice, white, long-grain, regular, enriched, cooked
    173904,  # Cereals, oats, regular and quick, not fortified, dry
    173905,  # Cereals, oats, regular and quick, unenriched, cooked
]

# Common fruits
FRUIT_STAPLES = [
    171688,  # Apples, raw, with skin
    173944,  # Bananas, raw
    171711,  # Blueberries, raw
    167762,  # Strawberries, raw
    173946,  # Blackberries, raw
    169097,  # Oranges, raw, all commercial varieties
    169105,  # Tangerines, (mandarin oranges), raw
]

# Starchy vegetables (for some patterns)
STARCHY_STAPLES = [
    170093,  # Potatoes, baked, flesh and skin, without salt
    170033,  # Potatoes, baked, flesh, without salt
    # Sweet potatoes would go here if we found good IDs
]


# =============================================================================
# Pattern-Specific Staple Lists
# =============================================================================

PESCATARIAN_STAPLES = (
    FISH_STAPLES +
    EGG_STAPLES +
    LEGUME_STAPLES +
    VEGETABLE_STAPLES +
    NUT_STAPLES +
    DAIRY_STAPLES +
    FAT_STAPLES +
    PLANT_PROTEIN_STAPLES +
    GRAIN_STAPLES +
    FRUIT_STAPLES
)

VEGETARIAN_STAPLES = (
    EGG_STAPLES +
    LEGUME_STAPLES +
    VEGETABLE_STAPLES +
    NUT_STAPLES +
    DAIRY_STAPLES +
    FAT_STAPLES +
    PLANT_PROTEIN_STAPLES +
    GRAIN_STAPLES +
    FRUIT_STAPLES
)

VEGAN_STAPLES = (
    LEGUME_STAPLES +
    VEGETABLE_STAPLES +
    NUT_STAPLES +
    [171412, 748608, 173573, 171705, 171706] +  # Oils and avocado (no butter)
    PLANT_PROTEIN_STAPLES +
    GRAIN_STAPLES +
    FRUIT_STAPLES
)

# Keto: high fat, very low carb, moderate protein
KETO_STAPLES = (
    FISH_STAPLES +
    EGG_STAPLES +
    CHICKEN_STAPLES +
    BEEF_STAPLES +
    PORK_STAPLES +
    TURKEY_STAPLES +
    LAMB_STAPLES +
    # Only low-carb vegetables
    [
        168462,  # Spinach, raw
        168421,  # Kale, raw
        169247,  # Lettuce, romaine
        169248,  # Lettuce, iceberg
        170379,  # Broccoli, raw
        169986,  # Cauliflower, raw
        170383,  # Brussels sprouts, raw
        2346407, # Cabbage, raw
        168389,  # Asparagus, raw
        169291,  # Zucchini, raw
        168409,  # Cucumber, raw
        168434,  # Mushrooms, crimini
        2003598, # Mushroom, portabella
        2258588, # Bell peppers, green
    ] +
    NUT_STAPLES +
    DAIRY_STAPLES +
    FAT_STAPLES
    # No grains, legumes, or fruits (except small berries occasionally)
)

# Slow carb (Tim Ferriss): legumes allowed, no grains/fruits/dairy
SLOW_CARB_STAPLES = (
    FISH_STAPLES +
    EGG_STAPLES +
    CHICKEN_STAPLES +
    BEEF_STAPLES +
    PORK_STAPLES +
    TURKEY_STAPLES +
    LEGUME_STAPLES +
    VEGETABLE_STAPLES +
    [171413, 748608, 171412, 173573, 171705, 171706]  # Oils and avocado only
    # No grains, no fruits, no dairy (except cottage cheese)
)

# Mediterranean: fish, olive oil, legumes, whole grains, vegetables, fruit, nuts
MEDITERRANEAN_STAPLES = (
    FISH_STAPLES +
    EGG_STAPLES +
    CHICKEN_STAPLES +  # Moderate poultry
    LEGUME_STAPLES +
    VEGETABLE_STAPLES +
    NUT_STAPLES +
    [  # Olive oil focused, limited dairy (mainly yogurt/cheese)
        171413,  # Oil, olive, salad or cooking
        748608,  # Oil, olive, extra virgin
        171705,  # Avocados, raw
        170894,  # Yogurt, Greek, plain, nonfat
        171304,  # Yogurt, Greek, plain, whole milk
        328637,  # Cheese, cheddar (moderate)
    ] +
    GRAIN_STAPLES +
    FRUIT_STAPLES
)

# Paleo: meat, fish, eggs, vegetables, fruit, nuts; no grains/legumes/dairy
PALEO_STAPLES = (
    FISH_STAPLES +
    EGG_STAPLES +
    CHICKEN_STAPLES +
    BEEF_STAPLES +
    PORK_STAPLES +
    TURKEY_STAPLES +
    LAMB_STAPLES +
    VEGETABLE_STAPLES +
    NUT_STAPLES +
    [171413, 748608, 171412, 173573, 171705, 171706] +  # Oils and avocado
    FRUIT_STAPLES +
    STARCHY_STAPLES  # Sweet potatoes, potatoes allowed in some paleo variants
    # No grains, no legumes, no dairy
)

# Whole30: like paleo but stricter (no sugar, alcohol, grains, legumes, dairy)
WHOLE30_STAPLES = (
    FISH_STAPLES +
    EGG_STAPLES +
    CHICKEN_STAPLES +
    BEEF_STAPLES +
    PORK_STAPLES +
    TURKEY_STAPLES +
    LAMB_STAPLES +
    VEGETABLE_STAPLES +
    # Nuts allowed but no peanuts (they're legumes)
    [
        170567,  # Almonds
        170158,  # Almonds, dry roasted
        170187,  # Walnuts
        170162,  # Cashews, raw
        170571,  # Cashews, dry roasted
        170178,  # Macadamia nuts
        170182,  # Pecans
        170562,  # Sunflower seeds
        170554,  # Chia seeds
        169414,  # Flaxseed
        2515380, # Pumpkin seeds
    ] +
    [171413, 748608, 171412, 173573, 171705, 171706] +  # Oils and avocado
    FRUIT_STAPLES +
    STARCHY_STAPLES
    # No grains, no legumes, no dairy, no peanuts
)


# =============================================================================
# Pattern Mapping
# =============================================================================

PATTERN_STAPLES: dict[str, list[int]] = {
    "pescatarian": PESCATARIAN_STAPLES,
    "vegetarian": VEGETARIAN_STAPLES,
    "vegan": VEGAN_STAPLES,
    "keto": KETO_STAPLES,
    "slow_carb": SLOW_CARB_STAPLES,
    "mediterranean": MEDITERRANEAN_STAPLES,
    "paleo": PALEO_STAPLES,
    "whole30": WHOLE30_STAPLES,
}


def get_staples(pattern_name: str) -> list[int] | None:
    """Get the staple food IDs for a dietary pattern.

    Args:
        pattern_name: Name of the dietary pattern (e.g., "pescatarian")

    Returns:
        List of FDC IDs, or None if pattern has no staples defined.
    """
    return PATTERN_STAPLES.get(pattern_name.lower())


def combine_staples(*pattern_names: str) -> list[int]:
    """Combine staple lists from multiple patterns (intersection).

    When combining patterns like "pescatarian" + "slow_carb", we want
    foods that are valid for BOTH patterns. This means:
    - Take the intersection of allowed foods
    - If a pattern has no staples defined, skip it

    Args:
        *pattern_names: Names of dietary patterns to combine

    Returns:
        List of FDC IDs that are valid for all specified patterns.
    """
    if not pattern_names:
        return []

    staple_sets = []
    for name in pattern_names:
        staples = get_staples(name)
        if staples:
            staple_sets.append(set(staples))

    if not staple_sets:
        return []

    # Intersection of all pattern staples
    result = staple_sets[0]
    for s in staple_sets[1:]:
        result = result.intersection(s)

    return list(result)


def list_available_patterns() -> list[str]:
    """Return list of pattern names that have staples defined."""
    return list(PATTERN_STAPLES.keys())


# =============================================================================
# Source Mapping for Template-Based Meal Composition
# =============================================================================

# Maps slot source names to their corresponding staple lists
# Used by the template system to look up foods for each slot
SOURCE_MAPPING: dict[str, list[int]] = {
    # Protein sources
    "eggs": EGG_STAPLES,
    "fish": FISH_STAPLES,
    "poultry": CHICKEN_STAPLES + TURKEY_STAPLES,
    "chicken": CHICKEN_STAPLES,
    "turkey": TURKEY_STAPLES,
    "red_meat": BEEF_STAPLES + PORK_STAPLES + LAMB_STAPLES,
    "beef": BEEF_STAPLES,
    "pork": PORK_STAPLES,
    "lamb": LAMB_STAPLES,
    "plant_protein": PLANT_PROTEIN_STAPLES,
    "tofu": PLANT_PROTEIN_STAPLES,  # Alias
    # Legumes
    "legumes": LEGUME_STAPLES,
    "beans": LEGUME_STAPLES,  # Alias
    "lentils": [172421, 172420, 174284],  # Subset of legumes
    # Vegetables
    "vegetables": VEGETABLE_STAPLES,
    "leafy_greens": LEAFY_GREENS,
    "cruciferous": CRUCIFEROUS,
    "other_vegetables": OTHER_VEGETABLES,
    "mushrooms": MUSHROOMS,
    "low_carb_vegetables": LEAFY_GREENS + CRUCIFEROUS + MUSHROOMS,
    # Fats and dairy
    "nuts": NUT_STAPLES,
    "dairy": DAIRY_STAPLES,
    "fats": FAT_STAPLES,
    "oils": [171413, 748608, 171412, 173573],  # Oils only (no avocado)
    "avocado": [171705, 171706],
    "cheese": [328637, 170845, 171244, 172179, 173417, 172182, 173418, 172175],
    "yogurt": [170894, 171304, 170903, 170905],
    # Grains and starches
    "grains": GRAIN_STAPLES,
    "starchy": STARCHY_STAPLES,
    # Fruits
    "fruits": FRUIT_STAPLES,
}


def get_foods_for_source(source_name: str) -> list[int] | None:
    """Get food IDs for a template slot source.

    Args:
        source_name: Name of the source (e.g., "eggs", "fish", "leafy_greens")

    Returns:
        List of FDC IDs, or None if source not found.
    """
    return SOURCE_MAPPING.get(source_name.lower())


def get_foods_for_sources(source_names: list[str]) -> list[int]:
    """Get combined food IDs for multiple template slot sources.

    Args:
        source_names: List of source names

    Returns:
        Combined list of FDC IDs (union, deduplicated).
    """
    result = set()
    for name in source_names:
        foods = get_foods_for_source(name)
        if foods:
            result.update(foods)
    return list(result)


def list_available_sources() -> list[str]:
    """Return list of available source names for template slots."""
    return list(SOURCE_MAPPING.keys())
