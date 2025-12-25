"""Load USDA FoodData Central CSV files into SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from mealplan.data.nutrient_ids import MACRO_NUTRIENT_IDS


class USDALoader:
    """Handles loading USDA CSV data into the database."""

    REQUIRED_FILES = ["food.csv", "food_nutrient.csv", "nutrient.csv"]
    OPTIONAL_FILES = ["food_portion.csv"]

    def __init__(self, usda_path: Path, conn: sqlite3.Connection):
        """Initialize the USDA loader.

        Args:
            usda_path: Path to directory containing USDA CSV files
            conn: SQLite database connection
        """
        self.usda_path = usda_path
        self.conn = conn
        self._validate_path()

    def _validate_path(self) -> None:
        """Ensure required USDA files exist."""
        for filename in self.REQUIRED_FILES:
            filepath = self.usda_path / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required file '{filename}' not found in {self.usda_path}. "
                    f"Please download the USDA FoodData Central CSV files from "
                    f"https://fdc.nal.usda.gov/download-datasets.html"
                )

    def load_all(
        self,
        data_types: list[str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, int]:
        """Load all USDA data into the database.

        Args:
            data_types: List of data types to include (e.g., ['foundation_food', 'sr_legacy_food'])
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with counts of loaded records
        """
        counts: dict[str, int] = {}

        # Load nutrients first (no filtering needed)
        counts["nutrients"] = self._load_nutrients()
        if progress_callback:
            progress_callback(f"Loaded {counts['nutrients']} nutrients")

        # Load foods with filtering
        counts["foods"] = self._load_foods(data_types)
        if progress_callback:
            progress_callback(f"Loaded {counts['foods']} foods")

        # Load food_nutrients (only for loaded foods)
        counts["food_nutrients"] = self._load_food_nutrients()
        if progress_callback:
            progress_callback(f"Loaded {counts['food_nutrients']} food-nutrient records")

        # Load servings if available
        servings_path = self.usda_path / "food_portion.csv"
        if servings_path.exists():
            counts["servings"] = self._load_servings()
            if progress_callback:
                progress_callback(f"Loaded {counts['servings']} serving sizes")
        else:
            counts["servings"] = 0

        return counts

    def _load_nutrients(self) -> int:
        """Load nutrient definitions from nutrient.csv.

        Returns:
            Number of nutrients loaded
        """
        df = pd.read_csv(self.usda_path / "nutrient.csv")

        records = []
        for _, row in df.iterrows():
            nutrient_id = int(row["id"])
            records.append(
                (
                    nutrient_id,
                    row["name"],
                    row["unit_name"],
                    row["name"],  # display_name defaults to name
                    nutrient_id in MACRO_NUTRIENT_IDS,
                )
            )

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO nutrients
            (nutrient_id, name, unit, display_name, is_macro)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

        return len(records)

    def _load_foods(self, data_types: list[str]) -> int:
        """Load and filter foods from food.csv.

        Args:
            data_types: List of data types to include

        Returns:
            Number of foods loaded
        """
        df = pd.read_csv(self.usda_path / "food.csv", low_memory=False)

        # Filter by data type
        df = df[df["data_type"].isin(data_types)]

        # Handle duplicates: prefer foundation_food over sr_legacy_food
        # Sort so foundation comes before sr_legacy (alphabetically)
        df = df.sort_values("data_type")
        df = df.drop_duplicates(subset="description", keep="first")

        records = []
        for _, row in df.iterrows():
            food_category = row.get("food_category_id")
            if pd.isna(food_category):
                food_category = None

            records.append(
                (
                    int(row["fdc_id"]),
                    row["description"],
                    row["data_type"],
                    food_category,
                    True,  # is_active
                )
            )

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO foods
            (fdc_id, description, data_type, food_category, is_active)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

        return len(records)

    def _load_food_nutrients(self) -> int:
        """Load nutrient amounts from food_nutrient.csv.

        Only loads nutrients for foods that exist in the foods table.

        Returns:
            Number of food-nutrient records loaded
        """
        # Get set of loaded food IDs
        cursor = self.conn.execute("SELECT fdc_id FROM foods")
        food_ids = {row[0] for row in cursor.fetchall()}

        if not food_ids:
            return 0

        # Load in chunks to manage memory
        chunk_size = 100000
        total_loaded = 0

        for chunk in pd.read_csv(
            self.usda_path / "food_nutrient.csv",
            chunksize=chunk_size,
            low_memory=False,
        ):
            # Filter to loaded foods
            chunk = chunk[chunk["fdc_id"].isin(food_ids)]

            records = []
            for _, row in chunk.iterrows():
                amount = row.get("amount")
                if pd.notna(amount):
                    records.append(
                        (
                            int(row["fdc_id"]),
                            int(row["nutrient_id"]),
                            float(amount),
                        )
                    )

            if records:
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO food_nutrients
                    (fdc_id, nutrient_id, amount)
                    VALUES (?, ?, ?)
                    """,
                    records,
                )
                total_loaded += len(records)

        self.conn.commit()
        return total_loaded

    def _load_servings(self) -> int:
        """Load serving sizes from food_portion.csv.

        Returns:
            Number of serving records loaded
        """
        # Get set of loaded food IDs
        cursor = self.conn.execute("SELECT fdc_id FROM foods")
        food_ids = {row[0] for row in cursor.fetchall()}

        if not food_ids:
            return 0

        df = pd.read_csv(self.usda_path / "food_portion.csv", low_memory=False)

        # Filter to loaded foods
        df = df[df["fdc_id"].isin(food_ids)]

        records = []
        for _, row in df.iterrows():
            gram_weight = row.get("gram_weight")
            portion_desc = row.get("portion_description")

            # Skip if missing essential data
            if pd.isna(gram_weight) or pd.isna(portion_desc):
                continue

            # Skip if gram_weight is not a valid positive number
            try:
                gram_weight = float(gram_weight)
                if gram_weight <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            records.append(
                (
                    int(row["fdc_id"]),
                    str(portion_desc),
                    gram_weight,
                )
            )

        if records:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO servings
                (fdc_id, portion_description, gram_weight)
                VALUES (?, ?, ?)
                """,
                records,
            )

        self.conn.commit()
        return len(records)


def load_usda_data(
    usda_path: Path,
    conn: sqlite3.Connection,
    data_types: Optional[list[str]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, int]:
    """Convenience function to load USDA data.

    Args:
        usda_path: Path to USDA CSV directory
        conn: Database connection
        data_types: Data types to load (defaults to foundation_food, sr_legacy_food)
        progress_callback: Optional progress callback

    Returns:
        Dict with counts of loaded records
    """
    if data_types is None:
        data_types = ["foundation_food", "sr_legacy_food"]

    loader = USDALoader(usda_path, conn)
    return loader.load_all(data_types, progress_callback)
