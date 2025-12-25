"""Pytest fixtures for mealplan tests."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mealplan.db.connection import DatabaseConnection
from mealplan.db.schema import get_schema_sql


@pytest.fixture
def temp_db():
    """Create a temporary database with schema."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    db = DatabaseConnection(db_path)
    db.initialize_schema()

    yield db

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def sample_foods(temp_db):
    """Populate database with sample test foods."""
    with temp_db.get_connection() as conn:
        # Insert sample nutrients
        nutrients = [
            (1008, "Energy", "kcal", "Calories", True),
            (1003, "Protein", "g", "Protein", True),
            (1004, "Total lipid (fat)", "g", "Total Fat", True),
            (1005, "Carbohydrate, by difference", "g", "Carbohydrates", True),
            (1079, "Fiber, total dietary", "g", "Fiber", True),
            (1093, "Sodium, Na", "mg", "Sodium", False),
        ]
        conn.executemany(
            "INSERT INTO nutrients VALUES (?, ?, ?, ?, ?)",
            nutrients,
        )

        # Insert sample foods
        foods = [
            (1, "Chicken breast, boneless, skinless, raw", "foundation_food", None, True),
            (2, "Brown rice, medium-grain, cooked", "sr_legacy_food", None, True),
            (3, "Broccoli, raw", "foundation_food", None, True),
            (4, "Eggs, whole, raw", "foundation_food", None, True),
            (5, "Olive oil", "foundation_food", None, True),
        ]
        conn.executemany(
            "INSERT INTO foods (fdc_id, description, data_type, food_category, is_active) VALUES (?, ?, ?, ?, ?)",
            foods,
        )

        # Insert food_nutrients (per 100g)
        food_nutrients = [
            # Chicken breast: 165 kcal, 31g protein, 3.6g fat, 0g carbs, 0g fiber, 74mg sodium
            (1, 1008, 165), (1, 1003, 31), (1, 1004, 3.6), (1, 1005, 0), (1, 1079, 0), (1, 1093, 74),
            # Brown rice: 123 kcal, 2.7g protein, 1g fat, 26g carbs, 1.6g fiber, 1mg sodium
            (2, 1008, 123), (2, 1003, 2.7), (2, 1004, 1), (2, 1005, 26), (2, 1079, 1.6), (2, 1093, 1),
            # Broccoli: 34 kcal, 2.8g protein, 0.4g fat, 7g carbs, 2.6g fiber, 33mg sodium
            (3, 1008, 34), (3, 1003, 2.8), (3, 1004, 0.4), (3, 1005, 7), (3, 1079, 2.6), (3, 1093, 33),
            # Eggs: 143 kcal, 13g protein, 10g fat, 0.7g carbs, 0g fiber, 140mg sodium
            (4, 1008, 143), (4, 1003, 13), (4, 1004, 10), (4, 1005, 0.7), (4, 1079, 0), (4, 1093, 140),
            # Olive oil: 884 kcal, 0g protein, 100g fat, 0g carbs, 0g fiber, 2mg sodium
            (5, 1008, 884), (5, 1003, 0), (5, 1004, 100), (5, 1005, 0), (5, 1079, 0), (5, 1093, 2),
        ]
        conn.executemany(
            "INSERT INTO food_nutrients VALUES (?, ?, ?)",
            food_nutrients,
        )

        # Insert prices (per 100g)
        prices = [
            (1, 0.80, "test", None),   # Chicken: $0.80/100g
            (2, 0.15, "test", None),   # Rice: $0.15/100g
            (3, 0.30, "test", None),   # Broccoli: $0.30/100g
            (4, 0.40, "test", None),   # Eggs: $0.40/100g
            (5, 0.50, "test", None),   # Olive oil: $0.50/100g
        ]
        conn.executemany(
            "INSERT INTO prices (fdc_id, price_per_100g, price_source, notes) VALUES (?, ?, ?, ?)",
            prices,
        )

    return temp_db


@pytest.fixture
def sample_db_path(sample_foods):
    """Return path to the sample database."""
    return sample_foods.db_path
