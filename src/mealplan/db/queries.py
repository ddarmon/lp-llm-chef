"""Common database query functions."""

from __future__ import annotations

import json
import sqlite3
from typing import Optional


class FoodQueries:
    """Query functions for foods table."""

    @staticmethod
    def search_foods(
        conn: sqlite3.Connection, search_term: str, limit: int = 50
    ) -> list[sqlite3.Row]:
        """Search foods by description using LIKE matching.

        Args:
            conn: Database connection
            search_term: Search string
            limit: Maximum results to return

        Returns:
            List of matching food rows
        """
        query = """
            SELECT
                f.fdc_id,
                f.description,
                f.data_type,
                f.food_category,
                f.is_active,
                p.price_per_100g,
                GROUP_CONCAT(ft.tag, ', ') as tags
            FROM foods f
            LEFT JOIN prices p ON f.fdc_id = p.fdc_id
            LEFT JOIN food_tags ft ON f.fdc_id = ft.fdc_id
            WHERE f.description LIKE ? AND f.is_active = TRUE
            GROUP BY f.fdc_id
            ORDER BY f.description
            LIMIT ?
        """
        return conn.execute(query, (f"%{search_term}%", limit)).fetchall()

    @staticmethod
    def get_food_by_id(
        conn: sqlite3.Connection, fdc_id: int
    ) -> Optional[sqlite3.Row]:
        """Get a single food by FDC ID.

        Args:
            conn: Database connection
            fdc_id: USDA FDC ID

        Returns:
            Food row or None if not found
        """
        query = "SELECT * FROM foods WHERE fdc_id = ?"
        return conn.execute(query, (fdc_id,)).fetchone()

    @staticmethod
    def get_food_nutrients(
        conn: sqlite3.Connection, fdc_id: int
    ) -> list[sqlite3.Row]:
        """Get all nutrients for a food with nutrient names.

        Args:
            conn: Database connection
            fdc_id: USDA FDC ID

        Returns:
            List of nutrient rows with name, unit, and amount
        """
        query = """
            SELECT fn.nutrient_id, n.name, n.unit, fn.amount, n.display_name
            FROM food_nutrients fn
            JOIN nutrients n ON fn.nutrient_id = n.nutrient_id
            WHERE fn.fdc_id = ?
            ORDER BY n.is_macro DESC, n.name
        """
        return conn.execute(query, (fdc_id,)).fetchall()

    @staticmethod
    def get_all_active_food_ids(conn: sqlite3.Connection) -> set[int]:
        """Get set of all active food IDs.

        Args:
            conn: Database connection

        Returns:
            Set of fdc_id values
        """
        query = "SELECT fdc_id FROM foods WHERE is_active = TRUE"
        return {row[0] for row in conn.execute(query).fetchall()}

    @staticmethod
    def set_food_active(
        conn: sqlite3.Connection, fdc_id: int, is_active: bool
    ) -> None:
        """Set a food's active status.

        Args:
            conn: Database connection
            fdc_id: USDA FDC ID
            is_active: New active status
        """
        query = "UPDATE foods SET is_active = ? WHERE fdc_id = ?"
        conn.execute(query, (is_active, fdc_id))


class NutrientQueries:
    """Query functions for nutrients table."""

    @staticmethod
    def get_nutrient_by_id(
        conn: sqlite3.Connection, nutrient_id: int
    ) -> Optional[sqlite3.Row]:
        """Get a nutrient definition by ID.

        Args:
            conn: Database connection
            nutrient_id: USDA nutrient ID

        Returns:
            Nutrient row or None
        """
        query = "SELECT * FROM nutrients WHERE nutrient_id = ?"
        return conn.execute(query, (nutrient_id,)).fetchone()

    @staticmethod
    def get_all_nutrients(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        """Get all nutrient definitions.

        Args:
            conn: Database connection

        Returns:
            List of all nutrient rows
        """
        query = "SELECT * FROM nutrients ORDER BY is_macro DESC, name"
        return conn.execute(query).fetchall()

    @staticmethod
    def get_nutrient_names(
        conn: sqlite3.Connection, nutrient_ids: list[int]
    ) -> dict[int, tuple[str, str]]:
        """Get display names and units for nutrients.

        Args:
            conn: Database connection
            nutrient_ids: List of nutrient IDs

        Returns:
            Dict mapping nutrient_id to (display_name, unit)
        """
        if not nutrient_ids:
            return {}
        placeholders = ",".join("?" * len(nutrient_ids))
        query = f"""
            SELECT nutrient_id, COALESCE(display_name, name), unit
            FROM nutrients
            WHERE nutrient_id IN ({placeholders})
        """
        rows = conn.execute(query, nutrient_ids).fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}


class PriceQueries:
    """Query functions for prices table."""

    @staticmethod
    def get_price(
        conn: sqlite3.Connection, fdc_id: int
    ) -> Optional[sqlite3.Row]:
        """Get price for a food.

        Args:
            conn: Database connection
            fdc_id: USDA FDC ID

        Returns:
            Price row or None
        """
        query = "SELECT * FROM prices WHERE fdc_id = ?"
        return conn.execute(query, (fdc_id,)).fetchone()

    @staticmethod
    def get_foods_with_prices(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        """Get all foods that have prices.

        Args:
            conn: Database connection

        Returns:
            List of food rows with prices
        """
        query = """
            SELECT f.fdc_id, f.description, p.price_per_100g, p.price_source, p.notes
            FROM foods f
            INNER JOIN prices p ON f.fdc_id = p.fdc_id
            WHERE f.is_active = TRUE
            ORDER BY f.description
        """
        return conn.execute(query).fetchall()

    @staticmethod
    def get_foods_without_prices(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        """Get active foods that don't have prices.

        Args:
            conn: Database connection

        Returns:
            List of food rows without prices
        """
        query = """
            SELECT f.fdc_id, f.description
            FROM foods f
            LEFT JOIN prices p ON f.fdc_id = p.fdc_id
            WHERE p.fdc_id IS NULL AND f.is_active = TRUE
            ORDER BY f.description
        """
        return conn.execute(query).fetchall()

    @staticmethod
    def upsert_price(
        conn: sqlite3.Connection,
        fdc_id: int,
        price_per_100g: float,
        source: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Insert or update a price for a food.

        Args:
            conn: Database connection
            fdc_id: USDA FDC ID
            price_per_100g: Price per 100 grams
            source: Optional price source
            notes: Optional notes
        """
        query = """
            INSERT INTO prices (fdc_id, price_per_100g, price_source, price_date, notes)
            VALUES (?, ?, ?, DATE('now'), ?)
            ON CONFLICT(fdc_id) DO UPDATE SET
                price_per_100g = excluded.price_per_100g,
                price_source = excluded.price_source,
                price_date = excluded.price_date,
                notes = excluded.notes
        """
        conn.execute(query, (fdc_id, price_per_100g, source, notes))

    @staticmethod
    def delete_price(conn: sqlite3.Connection, fdc_id: int) -> bool:
        """Delete a price for a food.

        Args:
            conn: Database connection
            fdc_id: USDA FDC ID

        Returns:
            True if a row was deleted
        """
        query = "DELETE FROM prices WHERE fdc_id = ?"
        cursor = conn.execute(query, (fdc_id,))
        return cursor.rowcount > 0


class TagQueries:
    """Query functions for food_tags table."""

    @staticmethod
    def add_tag(conn: sqlite3.Connection, fdc_id: int, tag: str) -> None:
        """Add a tag to a food.

        Args:
            conn: Database connection
            fdc_id: USDA FDC ID
            tag: Tag string
        """
        query = "INSERT OR IGNORE INTO food_tags (fdc_id, tag) VALUES (?, ?)"
        conn.execute(query, (fdc_id, tag.lower()))

    @staticmethod
    def remove_tag(conn: sqlite3.Connection, fdc_id: int, tag: str) -> bool:
        """Remove a tag from a food.

        Args:
            conn: Database connection
            fdc_id: USDA FDC ID
            tag: Tag string

        Returns:
            True if tag was removed
        """
        query = "DELETE FROM food_tags WHERE fdc_id = ? AND tag = ?"
        cursor = conn.execute(query, (fdc_id, tag.lower()))
        return cursor.rowcount > 0

    @staticmethod
    def get_tags_for_food(conn: sqlite3.Connection, fdc_id: int) -> list[str]:
        """Get all tags for a food.

        Args:
            conn: Database connection
            fdc_id: USDA FDC ID

        Returns:
            List of tag strings
        """
        query = "SELECT tag FROM food_tags WHERE fdc_id = ? ORDER BY tag"
        return [row[0] for row in conn.execute(query, (fdc_id,)).fetchall()]

    @staticmethod
    def get_foods_with_tag(conn: sqlite3.Connection, tag: str) -> list[sqlite3.Row]:
        """Get all foods with a specific tag.

        Args:
            conn: Database connection
            tag: Tag string

        Returns:
            List of food rows
        """
        query = """
            SELECT f.fdc_id, f.description, f.food_category
            FROM foods f
            INNER JOIN food_tags ft ON f.fdc_id = ft.fdc_id
            WHERE ft.tag = ? AND f.is_active = TRUE
            ORDER BY f.description
        """
        return conn.execute(query, (tag.lower(),)).fetchall()

    @staticmethod
    def get_all_tags(conn: sqlite3.Connection) -> list[tuple[str, int]]:
        """Get all unique tags with counts.

        Args:
            conn: Database connection

        Returns:
            List of (tag, count) tuples
        """
        query = """
            SELECT tag, COUNT(*) as count
            FROM food_tags
            GROUP BY tag
            ORDER BY count DESC, tag
        """
        return [(row[0], row[1]) for row in conn.execute(query).fetchall()]


class ProfileQueries:
    """Query functions for constraint_profiles table."""

    @staticmethod
    def create_profile(
        conn: sqlite3.Connection,
        name: str,
        constraints_json: str,
        description: Optional[str] = None,
    ) -> int:
        """Create a new constraint profile.

        Args:
            conn: Database connection
            name: Profile name (must be unique)
            constraints_json: JSON string of constraints
            description: Optional description

        Returns:
            New profile_id
        """
        query = """
            INSERT INTO constraint_profiles (name, description, constraints_json)
            VALUES (?, ?, ?)
        """
        cursor = conn.execute(query, (name, description, constraints_json))
        return cursor.lastrowid or 0

    @staticmethod
    def get_profile_by_name(
        conn: sqlite3.Connection, name: str
    ) -> Optional[sqlite3.Row]:
        """Get a profile by name.

        Args:
            conn: Database connection
            name: Profile name

        Returns:
            Profile row or None
        """
        query = "SELECT * FROM constraint_profiles WHERE name = ?"
        return conn.execute(query, (name,)).fetchone()

    @staticmethod
    def get_profile_by_id(
        conn: sqlite3.Connection, profile_id: int
    ) -> Optional[sqlite3.Row]:
        """Get a profile by ID.

        Args:
            conn: Database connection
            profile_id: Profile ID

        Returns:
            Profile row or None
        """
        query = "SELECT * FROM constraint_profiles WHERE profile_id = ?"
        return conn.execute(query, (profile_id,)).fetchone()

    @staticmethod
    def list_profiles(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        """List all profiles.

        Args:
            conn: Database connection

        Returns:
            List of profile rows
        """
        query = """
            SELECT profile_id, name, description, created_at
            FROM constraint_profiles
            ORDER BY name
        """
        return conn.execute(query).fetchall()

    @staticmethod
    def delete_profile(conn: sqlite3.Connection, name: str) -> bool:
        """Delete a profile by name.

        Args:
            conn: Database connection
            name: Profile name

        Returns:
            True if profile was deleted
        """
        query = "DELETE FROM constraint_profiles WHERE name = ?"
        cursor = conn.execute(query, (name,))
        return cursor.rowcount > 0

    @staticmethod
    def update_profile(
        conn: sqlite3.Connection,
        name: str,
        constraints_json: str,
        description: Optional[str] = None,
    ) -> bool:
        """Update an existing profile.

        Args:
            conn: Database connection
            name: Profile name
            constraints_json: New JSON constraints
            description: New description

        Returns:
            True if profile was updated
        """
        query = """
            UPDATE constraint_profiles
            SET constraints_json = ?, description = ?
            WHERE name = ?
        """
        cursor = conn.execute(query, (constraints_json, description, name))
        return cursor.rowcount > 0


class OptimizationRunQueries:
    """Query functions for optimization_runs table."""

    @staticmethod
    def save_run(
        conn: sqlite3.Connection,
        profile_id: Optional[int],
        status: str,
        total_cost: Optional[float],
        result_json: str,
    ) -> int:
        """Save an optimization run.

        Args:
            conn: Database connection
            profile_id: Optional profile ID used
            status: Run status ('success', 'infeasible', 'error')
            total_cost: Total daily cost if successful
            result_json: Full result as JSON

        Returns:
            New run_id
        """
        query = """
            INSERT INTO optimization_runs (profile_id, status, total_cost, result_json)
            VALUES (?, ?, ?, ?)
        """
        cursor = conn.execute(query, (profile_id, status, total_cost, result_json))
        return cursor.lastrowid or 0

    @staticmethod
    def get_run_by_id(
        conn: sqlite3.Connection, run_id: int
    ) -> Optional[sqlite3.Row]:
        """Get a specific optimization run.

        Args:
            conn: Database connection
            run_id: Run ID

        Returns:
            Run row or None
        """
        query = "SELECT * FROM optimization_runs WHERE run_id = ?"
        return conn.execute(query, (run_id,)).fetchone()

    @staticmethod
    def get_latest_run(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
        """Get the most recent optimization run.

        Args:
            conn: Database connection

        Returns:
            Latest run row or None
        """
        query = """
            SELECT * FROM optimization_runs
            ORDER BY run_id DESC
            LIMIT 1
        """
        return conn.execute(query).fetchone()

    @staticmethod
    def list_runs(
        conn: sqlite3.Connection, limit: int = 10
    ) -> list[sqlite3.Row]:
        """List recent optimization runs.

        Args:
            conn: Database connection
            limit: Maximum runs to return

        Returns:
            List of run rows with profile names
        """
        query = """
            SELECT r.run_id, r.run_date, r.status, r.total_cost,
                   COALESCE(p.name, 'ad-hoc') as profile_name
            FROM optimization_runs r
            LEFT JOIN constraint_profiles p ON r.profile_id = p.profile_id
            ORDER BY r.run_id DESC
            LIMIT ?
        """
        return conn.execute(query, (limit,)).fetchall()

    @staticmethod
    def delete_run(conn: sqlite3.Connection, run_id: int) -> bool:
        """Delete an optimization run.

        Args:
            conn: Database connection
            run_id: Run ID

        Returns:
            True if run was deleted
        """
        query = "DELETE FROM optimization_runs WHERE run_id = ?"
        cursor = conn.execute(query, (run_id,))
        return cursor.rowcount > 0
