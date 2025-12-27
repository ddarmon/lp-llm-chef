"""Smart food pool suggestion for LLM-driven optimization."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Any, Optional

from mealplan.data.food_categories import FoodCategory, classify_foods_in_db


@dataclass
class PoolSuggestion:
    """A suggested food pool with rationale."""

    name: str
    description: str
    food_ids: list[int]
    strategy: str  # "balanced", "high_protein", "budget", "tag_based"
    estimated_variety: int  # Number of distinct foods

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "name": self.name,
            "description": self.description,
            "food_ids": self.food_ids,
            "strategy": self.strategy,
            "food_count": len(self.food_ids),
        }


def suggest_balanced_pool(
    conn: Connection,
    target_size: int = 30,
    require_price: bool = False,
    tag: Optional[str] = None,
) -> PoolSuggestion:
    """Suggest a balanced pool with foods from each macro category.

    Selects foods proportionally from protein, carb, fat, vegetable, and legume
    categories to ensure dietary variety.

    Args:
        conn: Database connection
        target_size: Target number of foods in pool
        require_price: Only include foods with prices
        tag: Optional tag to filter by (e.g., "staple")

    Returns:
        PoolSuggestion with balanced food selection
    """
    # Category proportions for a balanced diet
    category_proportions = {
        FoodCategory.PROTEIN: 0.30,
        FoodCategory.VEGETABLE: 0.25,
        FoodCategory.LEGUME: 0.15,
        FoodCategory.CARB: 0.15,
        FoodCategory.FAT: 0.10,
        FoodCategory.FRUIT: 0.05,
    }

    # Build base query
    params: list[Any] = []
    tag_join = ""
    tag_filter = ""
    price_filter = ""

    if tag:
        tag_join = "JOIN food_tags ft ON f.fdc_id = ft.fdc_id"
        tag_filter = "AND ft.tag = ?"
        params.append(tag.lower())

    if require_price:
        price_filter = "AND p.price_per_100g IS NOT NULL"

    # Get all eligible foods with their nutrients
    query = f"""
        SELECT
            f.fdc_id,
            f.description,
            (SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1003) as protein,
            (SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1005) as carbs,
            (SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1004) as fat,
            (SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1079) as fiber,
            (SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1008) as energy
        FROM foods f
        LEFT JOIN prices p ON f.fdc_id = p.fdc_id
        {tag_join}
        WHERE f.is_active = TRUE
        {tag_filter}
        {price_filter}
    """

    rows = conn.execute(query, params).fetchall()

    if not rows:
        return PoolSuggestion(
            name="empty_pool",
            description="No eligible foods found",
            food_ids=[],
            strategy="balanced",
            estimated_variety=0,
        )

    # Classify all foods
    fdc_ids = [row["fdc_id"] for row in rows]
    categories = classify_foods_in_db(conn, fdc_ids)

    # Group foods by category
    foods_by_category: dict[FoodCategory, list[int]] = {cat: [] for cat in FoodCategory}
    for row in rows:
        fdc_id = row["fdc_id"]
        cat = categories.get(fdc_id, FoodCategory.MIXED)
        foods_by_category[cat].append(fdc_id)

    # Select foods from each category according to proportions
    selected: list[int] = []
    for category, proportion in category_proportions.items():
        n_to_select = int(target_size * proportion)
        available = foods_by_category.get(category, [])
        if available:
            n_to_select = min(n_to_select, len(available))
            selected.extend(random.sample(available, n_to_select))

    # Fill remaining slots with MIXED category or random from any
    remaining = target_size - len(selected)
    if remaining > 0:
        mixed_foods = foods_by_category.get(FoodCategory.MIXED, [])
        unused_mixed = [f for f in mixed_foods if f not in selected]
        if unused_mixed:
            n_to_add = min(remaining, len(unused_mixed))
            selected.extend(random.sample(unused_mixed, n_to_add))

    return PoolSuggestion(
        name="balanced_pool",
        description=f"Balanced selection of {len(selected)} foods across macro categories",
        food_ids=selected,
        strategy="balanced",
        estimated_variety=len(selected),
    )


def suggest_high_protein_pool(
    conn: Connection,
    target_size: int = 30,
    min_protein_per_100g: float = 20.0,
    tag: Optional[str] = None,
) -> PoolSuggestion:
    """Suggest a pool optimized for high protein content.

    Args:
        conn: Database connection
        target_size: Target number of foods
        min_protein_per_100g: Minimum protein per 100g
        tag: Optional tag filter

    Returns:
        PoolSuggestion with high-protein foods
    """
    params: list[Any] = [min_protein_per_100g]
    tag_join = ""
    tag_filter = ""

    if tag:
        tag_join = "JOIN food_tags ft ON f.fdc_id = ft.fdc_id"
        tag_filter = "AND ft.tag = ?"
        params.append(tag.lower())

    params.append(target_size * 2)  # Fetch more for variety

    query = f"""
        SELECT
            f.fdc_id,
            f.description,
            fn.amount as protein_per_100g,
            (SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1008) as energy
        FROM foods f
        JOIN food_nutrients fn ON f.fdc_id = fn.fdc_id
        {tag_join}
        WHERE fn.nutrient_id = 1003
          AND fn.amount >= ?
          AND f.is_active = TRUE
          {tag_filter}
        ORDER BY fn.amount DESC
        LIMIT ?
    """

    rows = conn.execute(query, params).fetchall()

    if not rows:
        return PoolSuggestion(
            name="high_protein_empty",
            description=f"No foods found with >= {min_protein_per_100g}g protein per 100g",
            food_ids=[],
            strategy="high_protein",
            estimated_variety=0,
        )

    # Select a diverse subset (not just top N)
    food_ids = [row["fdc_id"] for row in rows]
    if len(food_ids) > target_size:
        # Take top half, then sample randomly from the rest
        top_half = food_ids[: target_size // 2]
        rest = food_ids[target_size // 2 :]
        random_selection = random.sample(rest, min(target_size - len(top_half), len(rest)))
        selected = top_half + random_selection
    else:
        selected = food_ids

    return PoolSuggestion(
        name="high_protein_pool",
        description=f"High protein foods (>= {min_protein_per_100g}g/100g)",
        food_ids=selected,
        strategy="high_protein",
        estimated_variety=len(selected),
    )


def suggest_budget_pool(
    conn: Connection,
    target_size: int = 30,
    max_price_per_100g: float = 1.0,
    tag: Optional[str] = None,
) -> PoolSuggestion:
    """Suggest a pool of budget-friendly foods.

    Args:
        conn: Database connection
        target_size: Target number of foods
        max_price_per_100g: Maximum price per 100g
        tag: Optional tag filter

    Returns:
        PoolSuggestion with budget-friendly foods
    """
    params: list[Any] = [max_price_per_100g]
    tag_join = ""
    tag_filter = ""

    if tag:
        tag_join = "JOIN food_tags ft ON f.fdc_id = ft.fdc_id"
        tag_filter = "AND ft.tag = ?"
        params.append(tag.lower())

    params.append(target_size * 2)

    query = f"""
        SELECT
            f.fdc_id,
            f.description,
            p.price_per_100g
        FROM foods f
        JOIN prices p ON f.fdc_id = p.fdc_id
        {tag_join}
        WHERE p.price_per_100g <= ?
          AND p.price_per_100g > 0
          AND f.is_active = TRUE
          {tag_filter}
        ORDER BY p.price_per_100g ASC
        LIMIT ?
    """

    rows = conn.execute(query, params).fetchall()

    if not rows:
        return PoolSuggestion(
            name="budget_empty",
            description=f"No foods found with price <= ${max_price_per_100g}/100g",
            food_ids=[],
            strategy="budget",
            estimated_variety=0,
        )

    # Diversify the selection
    food_ids = [row["fdc_id"] for row in rows]
    categories = classify_foods_in_db(conn, food_ids)

    # Try to get variety across categories
    by_category: dict[FoodCategory, list[int]] = {}
    for fdc_id in food_ids:
        cat = categories.get(fdc_id, FoodCategory.MIXED)
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(fdc_id)

    # Round-robin selection from categories
    selected: list[int] = []
    category_lists = list(by_category.values())
    idx = 0
    while len(selected) < target_size and category_lists:
        for cat_list in category_lists[:]:
            if cat_list and len(selected) < target_size:
                selected.append(cat_list.pop(0))
            if not cat_list:
                category_lists.remove(cat_list)

    return PoolSuggestion(
        name="budget_pool",
        description=f"Budget-friendly foods (<= ${max_price_per_100g}/100g)",
        food_ids=selected,
        strategy="budget",
        estimated_variety=len(selected),
    )


def suggest_pools(
    conn: Connection,
    strategies: Optional[list[str]] = None,
    target_size: int = 30,
    tag: Optional[str] = None,
    require_price: bool = False,
) -> list[PoolSuggestion]:
    """Generate multiple pool suggestions using different strategies.

    Args:
        conn: Database connection
        strategies: List of strategies to use (default: all)
        target_size: Target size for each pool
        tag: Optional tag to filter foods
        require_price: Whether to require prices

    Returns:
        List of PoolSuggestion objects
    """
    available_strategies = {
        "balanced": lambda: suggest_balanced_pool(conn, target_size, require_price, tag),
        "high_protein": lambda: suggest_high_protein_pool(conn, target_size, 20.0, tag),
        "budget": lambda: suggest_budget_pool(conn, target_size, 1.0, tag),
    }

    if strategies is None:
        strategies = list(available_strategies.keys())

    suggestions = []
    for strategy in strategies:
        if strategy in available_strategies:
            suggestion = available_strategies[strategy]()
            if suggestion.food_ids:  # Only include non-empty pools
                suggestions.append(suggestion)

    return suggestions


def format_pool_suggestions(suggestions: list[PoolSuggestion]) -> dict[str, Any]:
    """Format pool suggestions for JSON output.

    Args:
        suggestions: List of pool suggestions

    Returns:
        Formatted dictionary for JSON
    """
    return {
        "pools": [s.to_dict() for s in suggestions],
        "count": len(suggestions),
    }
