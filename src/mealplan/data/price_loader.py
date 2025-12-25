"""Load and validate price data from CSV files."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


class PriceLoader:
    """Handles importing prices from CSV files."""

    REQUIRED_COLUMNS = ["fdc_id", "price_per_100g"]
    OPTIONAL_COLUMNS = ["price_source", "price_date", "notes"]

    def __init__(self, conn: sqlite3.Connection):
        """Initialize the price loader.

        Args:
            conn: SQLite database connection
        """
        self.conn = conn

    def load_from_csv(self, csv_path: Path) -> dict[str, int]:
        """Load prices from a CSV file.

        CSV format:
            fdc_id,price_per_100g,price_source,price_date,notes
            167512,0.35,walmart,2025-01-15,chicken breast boneless skinless

        Args:
            csv_path: Path to the CSV file

        Returns:
            Dict with counts: {'loaded': n, 'skipped_invalid_id': m, 'skipped_missing_price': k}

        Raises:
            ValueError: If required columns are missing
        """
        df = pd.read_csv(csv_path)

        # Validate required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Required columns are: {self.REQUIRED_COLUMNS}"
            )

        # Get valid food IDs from database
        valid_ids = self._get_valid_food_ids()

        loaded = 0
        skipped_invalid_id = 0
        skipped_missing_price = 0

        for _, row in df.iterrows():
            fdc_id = row["fdc_id"]
            price = row["price_per_100g"]

            # Skip if missing price
            if pd.isna(price):
                skipped_missing_price += 1
                continue

            # Skip if fdc_id not in database
            if int(fdc_id) not in valid_ids:
                skipped_invalid_id += 1
                continue

            # Upsert the price
            self._upsert_price(row)
            loaded += 1

        self.conn.commit()

        return {
            "loaded": loaded,
            "skipped_invalid_id": skipped_invalid_id,
            "skipped_missing_price": skipped_missing_price,
        }

    def _get_valid_food_ids(self) -> set[int]:
        """Get set of valid fdc_ids from foods table.

        Returns:
            Set of valid food IDs
        """
        cursor = self.conn.execute("SELECT fdc_id FROM foods")
        return {row[0] for row in cursor.fetchall()}

    def _upsert_price(self, row: pd.Series) -> None:
        """Insert or update a single price record.

        Args:
            row: Pandas Series with price data
        """
        query = """
            INSERT INTO prices (fdc_id, price_per_100g, price_source, price_date, notes)
            VALUES (?, ?, ?, COALESCE(?, DATE('now')), ?)
            ON CONFLICT(fdc_id) DO UPDATE SET
                price_per_100g = excluded.price_per_100g,
                price_source = excluded.price_source,
                price_date = excluded.price_date,
                notes = excluded.notes
        """

        # Handle optional columns
        source = row.get("price_source")
        date = row.get("price_date")
        notes = row.get("notes")

        # Convert NaN to None
        if pd.isna(source):
            source = None
        if pd.isna(date):
            date = None
        if pd.isna(notes):
            notes = None

        self.conn.execute(
            query,
            (
                int(row["fdc_id"]),
                float(row["price_per_100g"]),
                source,
                date,
                notes,
            ),
        )

    def export_template(self, output_path: Path, include_foods: bool = True) -> int:
        """Export a template CSV with foods for price entry.

        Args:
            output_path: Path to write the template CSV
            include_foods: If True, include all foods without prices

        Returns:
            Number of foods in template
        """
        if include_foods:
            query = """
                SELECT f.fdc_id, f.description, '' as price_per_100g,
                       '' as price_source, '' as notes
                FROM foods f
                LEFT JOIN prices p ON f.fdc_id = p.fdc_id
                WHERE p.fdc_id IS NULL AND f.is_active = TRUE
                ORDER BY f.description
            """
        else:
            query = """
                SELECT '' as fdc_id, '' as description, '' as price_per_100g,
                       '' as price_source, '' as notes
                LIMIT 1
            """

        df = pd.read_sql_query(query, self.conn)
        df.to_csv(output_path, index=False)
        return len(df)


def load_prices_from_csv(
    csv_path: Path, conn: sqlite3.Connection
) -> dict[str, int]:
    """Convenience function to load prices from CSV.

    Args:
        csv_path: Path to prices CSV
        conn: Database connection

    Returns:
        Dict with load statistics
    """
    loader = PriceLoader(conn)
    return loader.load_from_csv(csv_path)
