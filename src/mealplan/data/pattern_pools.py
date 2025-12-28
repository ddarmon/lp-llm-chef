"""Food pool builder based on dietary patterns.

Builds food ID lists from the USDA database that match dietary patterns like
pescatarian, keto, slow_carb, etc. Replaces the need for manual SQL queries
when creating food pools for optimization.

Usage:
    from mealplan.data.pattern_pools import build_food_pool

    with db.get_connection() as conn:
        food_ids = build_food_pool(conn, ["pescatarian", "slow_carb"], max_foods=300)
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from mealplan.data.dietary_patterns import (
    DietaryPattern,
    combine_patterns,
    get_pattern,
)
from mealplan.data.food_categories import (
    FoodCategory,
    classify_foods_in_db,
)
from mealplan.data.pattern_staples import (
    combine_staples,
    get_staples,
)


# USDA nutrient IDs
ENERGY_ID = 1008
PROTEIN_ID = 1003


def build_food_pool(
    conn: sqlite3.Connection,
    patterns: list[str],
    max_foods: int = 300,
    min_protein_per_100g: Optional[float] = None,
    min_calories_per_100g: Optional[float] = None,
    max_calories_per_100g: Optional[float] = None,
    require_calorie_data: bool = True,
    balance_categories: bool = True,
    use_staples: bool = True,
) -> list[int]:
    """Build a food ID pool matching dietary patterns.

    By default, uses curated staple food lists for common, grocery-store-available
    foods. This prevents exotic foods (like "winged beans") from appearing in
    meal plans. Set use_staples=False to fall back to keyword-based filtering
    of the full USDA database.

    Args:
        conn: Database connection
        patterns: List of pattern names (e.g., ["pescatarian", "slow_carb"])
        max_foods: Maximum foods to return
        min_protein_per_100g: Minimum protein per 100g (optional filter)
        min_calories_per_100g: Minimum calories per 100g (filters out zero-cal)
        max_calories_per_100g: Maximum calories per 100g
        require_calorie_data: If True, exclude foods missing calorie data
        balance_categories: If True, ensure balanced representation across categories
        use_staples: If True (default), use curated staple food lists instead of
            keyword filtering. Falls back to keyword filtering if patterns have
            no staples defined.

    Returns:
        List of fdc_id values matching the patterns

    Raises:
        ValueError: If any pattern name is unknown
    """
    # Try to use curated staples first (if enabled)
    if use_staples:
        staple_ids = _get_staple_food_ids(conn, patterns)
        if staple_ids:
            # Apply nutrient filters if specified
            if min_protein_per_100g or min_calories_per_100g or max_calories_per_100g:
                staple_ids = _filter_by_nutrients(
                    conn,
                    staple_ids,
                    min_protein_per_100g=min_protein_per_100g,
                    min_calories_per_100g=min_calories_per_100g,
                    max_calories_per_100g=max_calories_per_100g,
                )

            # Limit to max_foods
            if len(staple_ids) > max_foods:
                import random
                staple_ids = random.sample(staple_ids, max_foods)

            return staple_ids

    # Fall back to keyword-based filtering
    # Combine all patterns
    if len(patterns) == 1:
        pattern = get_pattern(patterns[0])
        if pattern is None:
            raise ValueError(f"Unknown dietary pattern: {patterns[0]}")
    else:
        pattern = combine_patterns(*patterns)

    # Build the query with keyword filters
    food_ids = _query_foods_by_pattern(
        conn,
        pattern,
        min_protein_per_100g=min_protein_per_100g,
        min_calories_per_100g=min_calories_per_100g,
        max_calories_per_100g=max_calories_per_100g,
        require_calorie_data=require_calorie_data,
    )

    if not food_ids:
        return []

    # Balance across categories if requested
    if balance_categories and len(food_ids) > max_foods:
        food_ids = _balance_by_category(conn, food_ids, max_foods)
    elif len(food_ids) > max_foods:
        # Simple random sample
        import random
        food_ids = random.sample(food_ids, max_foods)

    return food_ids


def _get_staple_food_ids(
    conn: sqlite3.Connection,
    patterns: list[str],
) -> list[int]:
    """Get staple food IDs for the given patterns.

    Combines staple lists and validates that foods exist in the database.

    Returns:
        List of valid FDC IDs, or empty list if no staples available.
    """
    # Get combined staples for all patterns
    if len(patterns) == 1:
        staple_ids = get_staples(patterns[0])
    else:
        staple_ids = combine_staples(*patterns)

    if not staple_ids:
        return []

    # Validate that these foods exist in the database
    placeholders = ",".join("?" * len(staple_ids))
    cursor = conn.execute(
        f"""
        SELECT fdc_id FROM foods
        WHERE fdc_id IN ({placeholders}) AND is_active = TRUE
        """,
        staple_ids,
    )
    valid_ids = [row[0] for row in cursor.fetchall()]

    return valid_ids


def _filter_by_nutrients(
    conn: sqlite3.Connection,
    food_ids: list[int],
    min_protein_per_100g: Optional[float] = None,
    min_calories_per_100g: Optional[float] = None,
    max_calories_per_100g: Optional[float] = None,
) -> list[int]:
    """Filter food IDs by nutrient criteria."""
    if not food_ids:
        return []

    placeholders = ",".join("?" * len(food_ids))
    params: list = list(food_ids)

    # Build query with nutrient filters
    query = f"""
        SELECT DISTINCT fdc_id FROM food_nutrients
        WHERE fdc_id IN ({placeholders})
    """

    if min_protein_per_100g is not None:
        query += f"""
            AND fdc_id IN (
                SELECT fdc_id FROM food_nutrients
                WHERE nutrient_id = {PROTEIN_ID} AND amount >= ?
            )
        """
        params.append(min_protein_per_100g)

    if min_calories_per_100g is not None:
        query += f"""
            AND fdc_id IN (
                SELECT fdc_id FROM food_nutrients
                WHERE nutrient_id = {ENERGY_ID} AND amount >= ?
            )
        """
        params.append(min_calories_per_100g)

    if max_calories_per_100g is not None:
        query += f"""
            AND fdc_id IN (
                SELECT fdc_id FROM food_nutrients
                WHERE nutrient_id = {ENERGY_ID} AND amount <= ?
            )
        """
        params.append(max_calories_per_100g)

    cursor = conn.execute(query, params)
    return [row[0] for row in cursor.fetchall()]


def _query_foods_by_pattern(
    conn: sqlite3.Connection,
    pattern: DietaryPattern,
    min_protein_per_100g: Optional[float] = None,
    min_calories_per_100g: Optional[float] = None,
    max_calories_per_100g: Optional[float] = None,
    require_calorie_data: bool = True,
) -> list[int]:
    """Query foods matching a dietary pattern.

    Uses SQL LIKE patterns for include/exclude keyword matching.
    """
    # Start with base query for active foods with calorie data
    if require_calorie_data:
        query = """
            SELECT DISTINCT f.fdc_id, f.description
            FROM foods f
            INNER JOIN food_nutrients fn ON f.fdc_id = fn.fdc_id
            WHERE f.is_active = TRUE
              AND fn.nutrient_id = 1008
              AND fn.amount > 0
        """
    else:
        query = """
            SELECT DISTINCT f.fdc_id, f.description
            FROM foods f
            WHERE f.is_active = TRUE
        """

    params: list = []

    # Add calorie range filters
    if min_calories_per_100g is not None:
        query += """
            AND f.fdc_id IN (
                SELECT fdc_id FROM food_nutrients
                WHERE nutrient_id = 1008 AND amount >= ?
            )
        """
        params.append(min_calories_per_100g)

    if max_calories_per_100g is not None:
        query += """
            AND f.fdc_id IN (
                SELECT fdc_id FROM food_nutrients
                WHERE nutrient_id = 1008 AND amount <= ?
            )
        """
        params.append(max_calories_per_100g)

    # Add protein filter
    if min_protein_per_100g is not None:
        query += """
            AND f.fdc_id IN (
                SELECT fdc_id FROM food_nutrients
                WHERE nutrient_id = 1003 AND amount >= ?
            )
        """
        params.append(min_protein_per_100g)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    # Apply keyword filters in Python (more flexible than SQL LIKE patterns)
    filtered = []
    exclude_keywords = [kw.lower() for kw in pattern.exclude_keywords]
    include_keywords = [kw.lower() for kw in pattern.include_keywords]

    for fdc_id, description in rows:
        desc_lower = description.lower() if description else ""

        # Check exclusions first
        if any(kw in desc_lower for kw in exclude_keywords):
            continue

        # If include keywords are specified, require at least one match
        if include_keywords:
            if any(kw in desc_lower for kw in include_keywords):
                filtered.append(fdc_id)
        else:
            # No include filter, accept all that passed exclude
            filtered.append(fdc_id)

    return filtered


def _balance_by_category(
    conn: sqlite3.Connection,
    food_ids: list[int],
    max_foods: int,
) -> list[int]:
    """Balance food selection across categories.

    Ensures we don't over-represent any single category (e.g., 80% vegetables).
    Aims for roughly equal representation from each category present.
    """
    import random

    # Classify all foods
    categories = classify_foods_in_db(conn, food_ids)

    # Group by category
    by_category: dict[FoodCategory, list[int]] = {}
    for fdc_id in food_ids:
        cat = categories.get(fdc_id, FoodCategory.MIXED)
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(fdc_id)

    # Calculate per-category quota
    n_categories = len(by_category)
    if n_categories == 0:
        return []

    base_quota = max_foods // n_categories
    remainder = max_foods % n_categories

    # Sample from each category
    result = []
    for i, (cat, ids) in enumerate(by_category.items()):
        quota = base_quota + (1 if i < remainder else 0)
        if len(ids) <= quota:
            result.extend(ids)
        else:
            result.extend(random.sample(ids, quota))

    return result


def get_pattern_summary(
    conn: sqlite3.Connection,
    patterns: list[str],
    max_foods: int = 300,
) -> dict:
    """Get a summary of what a pattern combination would produce.

    Returns category distribution and sample foods without building the full pool.
    """
    # Build the pool
    food_ids = build_food_pool(
        conn,
        patterns,
        max_foods=max_foods,
        balance_categories=False,  # Get raw distribution first
    )

    if not food_ids:
        return {
            "patterns": patterns,
            "total_foods": 0,
            "categories": {},
            "sample_foods": [],
        }

    # Classify
    categories = classify_foods_in_db(conn, food_ids)

    # Count by category
    category_counts: dict[str, int] = {}
    for fdc_id in food_ids:
        cat = categories.get(fdc_id, FoodCategory.MIXED)
        cat_name = cat.value
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

    # Get sample foods (first 10)
    sample_ids = food_ids[:10]
    placeholders = ",".join("?" * len(sample_ids))
    cursor = conn.execute(
        f"SELECT fdc_id, description FROM foods WHERE fdc_id IN ({placeholders})",
        sample_ids,
    )
    samples = [{"fdc_id": row[0], "description": row[1]} for row in cursor.fetchall()]

    return {
        "patterns": patterns,
        "total_foods": len(food_ids),
        "categories": category_counts,
        "sample_foods": samples,
    }


def list_excluded_foods(
    conn: sqlite3.Connection,
    patterns: list[str],
    limit: int = 20,
) -> list[dict]:
    """Show foods that would be excluded by the pattern.

    Useful for understanding what a pattern filters out.
    """
    # Combine patterns
    if len(patterns) == 1:
        pattern = get_pattern(patterns[0])
        if pattern is None:
            return []
    else:
        pattern = combine_patterns(*patterns)

    exclude_keywords = [kw.lower() for kw in pattern.exclude_keywords]

    # Find foods matching any exclude keyword
    cursor = conn.execute(
        "SELECT fdc_id, description FROM foods WHERE is_active = TRUE LIMIT 5000"
    )
    rows = cursor.fetchall()

    excluded = []
    for fdc_id, description in rows:
        desc_lower = description.lower() if description else ""
        for kw in exclude_keywords:
            if kw in desc_lower:
                excluded.append({
                    "fdc_id": fdc_id,
                    "description": description,
                    "excluded_by": kw,
                })
                break
        if len(excluded) >= limit:
            break

    return excluded
