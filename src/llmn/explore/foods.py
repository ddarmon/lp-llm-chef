"""Food discovery and exploration functions."""

from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection
from typing import Any, Optional

from llmn.data.nutrient_ids import NUTRIENT_IDS


@dataclass
class FoodFilter:
    """Filter criteria for food searches."""

    query: Optional[str] = None
    min_protein: Optional[float] = None  # per 100g
    max_protein: Optional[float] = None
    min_calories: Optional[float] = None
    max_calories: Optional[float] = None
    min_cost: Optional[float] = None
    max_cost: Optional[float] = None
    has_tag: Optional[str] = None
    has_price: Optional[bool] = None
    category: Optional[str] = None  # protein, fat, carb, legume, vegetable, fruit, mixed
    limit: int = 50


def explore_foods(
    conn: Connection,
    filters: FoodFilter,
) -> list[dict[str, Any]]:
    """Search and filter foods based on criteria.

    Args:
        conn: Database connection
        filters: Filter criteria

    Returns:
        List of food info dictionaries
    """
    # Build dynamic query
    select_parts = [
        "f.fdc_id",
        "f.description",
        "f.data_type",
        "p.price_per_100g",
        "GROUP_CONCAT(DISTINCT ft.tag) as tags",
    ]

    # Add nutrient subqueries for filtering/display
    select_parts.append(
        "(SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1003) as protein_per_100g"
    )
    select_parts.append(
        "(SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1008) as calories_per_100g"
    )
    select_parts.append(
        "(SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1004) as fat_per_100g"
    )
    select_parts.append(
        "(SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1005) as carbs_per_100g"
    )

    from_clause = """
        FROM foods f
        LEFT JOIN prices p ON f.fdc_id = p.fdc_id
        LEFT JOIN food_tags ft ON f.fdc_id = ft.fdc_id
    """

    where_parts = ["f.is_active = TRUE"]
    params: list[Any] = []

    # Text search
    if filters.query:
        where_parts.append("f.description LIKE ?")
        params.append(f"%{filters.query}%")

    # Price filter
    if filters.has_price is True:
        where_parts.append("p.price_per_100g IS NOT NULL")
    elif filters.has_price is False:
        where_parts.append("p.price_per_100g IS NULL")

    if filters.min_cost is not None:
        where_parts.append("p.price_per_100g >= ?")
        params.append(filters.min_cost)
    if filters.max_cost is not None:
        where_parts.append("p.price_per_100g <= ?")
        params.append(filters.max_cost)

    # Tag filter
    if filters.has_tag:
        where_parts.append(
            "EXISTS (SELECT 1 FROM food_tags WHERE fdc_id = f.fdc_id AND tag = ?)"
        )
        params.append(filters.has_tag.lower())

    # Build query
    query = f"""
        SELECT {', '.join(select_parts)}
        {from_clause}
        WHERE {' AND '.join(where_parts)}
        GROUP BY f.fdc_id
    """

    # Add nutrient HAVING clauses (after GROUP BY)
    having_parts = []
    if filters.min_protein is not None:
        having_parts.append("protein_per_100g >= ?")
        params.append(filters.min_protein)
    if filters.max_protein is not None:
        having_parts.append("protein_per_100g <= ?")
        params.append(filters.max_protein)
    if filters.min_calories is not None:
        having_parts.append("calories_per_100g >= ?")
        params.append(filters.min_calories)
    if filters.max_calories is not None:
        having_parts.append("calories_per_100g <= ?")
        params.append(filters.max_calories)

    if having_parts:
        query += f" HAVING {' AND '.join(having_parts)}"

    # When category filter is used, fetch more results since we'll filter after
    fetch_limit = filters.limit * 5 if filters.category else filters.limit
    query += " ORDER BY f.description LIMIT ?"
    params.append(fetch_limit)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append({
            "fdc_id": row["fdc_id"],
            "description": row["description"],
            "data_type": row["data_type"],
            "price_per_100g": row["price_per_100g"],
            "tags": row["tags"].split(",") if row["tags"] else [],
            "protein_per_100g": row["protein_per_100g"],
            "calories_per_100g": row["calories_per_100g"],
            "fat_per_100g": row["fat_per_100g"],
            "carbs_per_100g": row["carbs_per_100g"],
        })

    # Apply category filter if specified
    if filters.category:
        from llmn.data.food_categories import FoodCategory, classify_foods_in_db

        try:
            target_category = FoodCategory(filters.category.lower())
        except ValueError:
            # Invalid category, return empty
            return []

        fdc_ids = [r["fdc_id"] for r in results]
        categories = classify_foods_in_db(conn, fdc_ids)

        results = [
            r for r in results
            if categories.get(r["fdc_id"]) == target_category
        ][:filters.limit]  # Apply limit after category filtering

    return results


def get_food_nutrients(
    conn: Connection,
    fdc_ids: list[int],
    nutrient_ids: Optional[list[int]] = None,
) -> dict[int, dict[str, Any]]:
    """Get nutrient data for multiple foods.

    Args:
        conn: Database connection
        fdc_ids: List of food IDs
        nutrient_ids: Optional list of nutrient IDs to fetch (default: all)

    Returns:
        Dict mapping fdc_id to nutrient data
    """
    if not fdc_ids:
        return {}

    placeholders = ",".join("?" * len(fdc_ids))
    params: list[Any] = list(fdc_ids)

    nutrient_filter = ""
    if nutrient_ids:
        nutrient_placeholders = ",".join("?" * len(nutrient_ids))
        nutrient_filter = f"AND fn.nutrient_id IN ({nutrient_placeholders})"
        params.extend(nutrient_ids)

    query = f"""
        SELECT
            fn.fdc_id,
            fn.nutrient_id,
            n.name,
            n.display_name,
            n.unit,
            fn.amount
        FROM food_nutrients fn
        JOIN nutrients n ON fn.nutrient_id = n.nutrient_id
        WHERE fn.fdc_id IN ({placeholders})
        {nutrient_filter}
        ORDER BY fn.fdc_id, n.is_macro DESC, n.name
    """

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        fdc_id = row["fdc_id"]
        if fdc_id not in result:
            result[fdc_id] = {"nutrients": {}}

        result[fdc_id]["nutrients"][row["nutrient_id"]] = {
            "name": row["name"],
            "display_name": row["display_name"],
            "unit": row["unit"],
            "amount": row["amount"],
        }

    return result


def compare_foods(
    conn: Connection,
    fdc_ids: list[int],
    nutrient_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Compare nutrient content of multiple foods side-by-side.

    Args:
        conn: Database connection
        fdc_ids: List of food IDs to compare
        nutrient_names: Optional list of nutrient names to include

    Returns:
        Comparison data with foods and nutrients
    """
    if not fdc_ids:
        return {"foods": [], "comparison": []}

    # Get food descriptions
    placeholders = ",".join("?" * len(fdc_ids))
    food_query = f"""
        SELECT fdc_id, description,
               (SELECT price_per_100g FROM prices WHERE fdc_id = f.fdc_id) as price
        FROM foods f
        WHERE fdc_id IN ({placeholders})
    """
    food_rows = conn.execute(food_query, fdc_ids).fetchall()
    foods = {
        row["fdc_id"]: {"description": row["description"], "price": row["price"]}
        for row in food_rows
    }

    # Determine which nutrients to compare
    nutrient_ids_to_fetch = None
    if nutrient_names:
        nutrient_ids_to_fetch = []
        for name in nutrient_names:
            name_lower = name.lower().replace(" ", "_").replace("-", "_")
            if name_lower in NUTRIENT_IDS:
                nutrient_ids_to_fetch.append(NUTRIENT_IDS[name_lower])

    # Get nutrients for all foods
    nutrient_data = get_food_nutrients(conn, fdc_ids, nutrient_ids_to_fetch)

    # Build comparison table
    # Collect all nutrient IDs across foods
    all_nutrient_ids: set[int] = set()
    for fdc_id in fdc_ids:
        if fdc_id in nutrient_data:
            all_nutrient_ids.update(nutrient_data[fdc_id]["nutrients"].keys())

    # Build comparison rows
    comparison = []
    for nutrient_id in sorted(all_nutrient_ids):
        row: dict[str, Any] = {"nutrient_id": nutrient_id}

        # Get nutrient name/unit from first food that has it
        for fdc_id in fdc_ids:
            if (
                fdc_id in nutrient_data
                and nutrient_id in nutrient_data[fdc_id]["nutrients"]
            ):
                info = nutrient_data[fdc_id]["nutrients"][nutrient_id]
                row["name"] = info["display_name"] or info["name"]
                row["unit"] = info["unit"]
                break

        # Get value for each food
        row["values"] = {}
        for fdc_id in fdc_ids:
            if (
                fdc_id in nutrient_data
                and nutrient_id in nutrient_data[fdc_id]["nutrients"]
            ):
                row["values"][fdc_id] = nutrient_data[fdc_id]["nutrients"][
                    nutrient_id
                ]["amount"]
            else:
                row["values"][fdc_id] = None

        comparison.append(row)

    return {
        "foods": [
            {
                "fdc_id": fdc_id,
                "description": foods.get(fdc_id, {}).get("description", "Unknown"),
                "price_per_100g": foods.get(fdc_id, {}).get("price"),
            }
            for fdc_id in fdc_ids
        ],
        "comparison": comparison,
    }


def find_high_nutrient_foods(
    conn: Connection,
    nutrient_name: str,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    has_tag: Optional[str] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Find foods high in a specific nutrient.

    Args:
        conn: Database connection
        nutrient_name: Name of nutrient (e.g., "protein", "iron")
        min_amount: Minimum amount per 100g
        max_amount: Maximum amount per 100g
        has_tag: Filter to foods with this tag
        limit: Maximum results

    Returns:
        List of foods with nutrient amounts, sorted by amount descending
    """
    name_lower = nutrient_name.lower().replace(" ", "_").replace("-", "_")
    if name_lower not in NUTRIENT_IDS:
        return []

    nutrient_id = NUTRIENT_IDS[name_lower]

    params: list[Any] = [nutrient_id]

    tag_join = ""
    tag_filter = ""
    if has_tag:
        tag_join = "JOIN food_tags ft ON f.fdc_id = ft.fdc_id"
        tag_filter = "AND ft.tag = ?"
        params.append(has_tag.lower())

    amount_filter = ""
    if min_amount is not None:
        amount_filter += " AND fn.amount >= ?"
        params.append(min_amount)
    if max_amount is not None:
        amount_filter += " AND fn.amount <= ?"
        params.append(max_amount)

    params.append(limit)

    query = f"""
        SELECT
            f.fdc_id,
            f.description,
            fn.amount,
            p.price_per_100g,
            (SELECT amount FROM food_nutrients WHERE fdc_id = f.fdc_id AND nutrient_id = 1008) as calories
        FROM foods f
        JOIN food_nutrients fn ON f.fdc_id = fn.fdc_id
        LEFT JOIN prices p ON f.fdc_id = p.fdc_id
        {tag_join}
        WHERE fn.nutrient_id = ?
          AND f.is_active = TRUE
          {tag_filter}
          {amount_filter}
        ORDER BY fn.amount DESC
        LIMIT ?
    """

    rows = conn.execute(query, params).fetchall()

    return [
        {
            "fdc_id": row["fdc_id"],
            "description": row["description"],
            "amount_per_100g": row["amount"],
            "calories_per_100g": row["calories"],
            "price_per_100g": row["price_per_100g"],
        }
        for row in rows
    ]
