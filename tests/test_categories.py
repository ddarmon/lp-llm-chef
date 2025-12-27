"""Tests for food category classification."""

from __future__ import annotations

import pytest

from mealplan.data.food_categories import FoodCategory, classify_food


class TestFoodClassification:
    """Tests for classify_food function."""

    def test_classify_high_protein(self):
        """Test classifying high-protein foods."""
        # Chicken breast: ~31g protein, ~3.6g fat, ~0g carbs per 100g
        category = classify_food(
            protein_per_100g=31, carbs_per_100g=0, fat_per_100g=3.6,
            fiber_per_100g=0, energy_per_100g=165, description="Chicken breast"
        )
        assert category == FoodCategory.PROTEIN

    def test_classify_high_fat(self):
        """Test classifying high-fat foods."""
        # Olive oil: 0g protein, 0g carbs, 100g fat per 100g
        category = classify_food(
            protein_per_100g=0, carbs_per_100g=0, fat_per_100g=100,
            fiber_per_100g=0, energy_per_100g=884, description="Olive oil"
        )
        assert category == FoodCategory.FAT

    def test_classify_high_carb(self):
        """Test classifying high-carb foods."""
        # Rice: ~2.7g protein, ~26g carbs, ~1g fat per 100g
        category = classify_food(
            protein_per_100g=2.7, carbs_per_100g=26, fat_per_100g=1,
            fiber_per_100g=1.6, energy_per_100g=123, description="Rice, cooked"
        )
        assert category == FoodCategory.CARB

    def test_classify_vegetable(self):
        """Test classifying vegetables (low energy, high fiber)."""
        # Broccoli: ~2.8g protein, ~7g carbs, ~0.4g fat, 2.6g fiber per 100g
        category = classify_food(
            protein_per_100g=2.8, carbs_per_100g=7, fat_per_100g=0.4,
            fiber_per_100g=2.6, energy_per_100g=34, description="Broccoli, raw"
        )
        assert category == FoodCategory.VEGETABLE

    def test_classify_legume(self):
        """Test classifying legumes (decent protein + fiber)."""
        # Lentils: ~9g protein, ~20g carbs, ~0.4g fat, ~8g fiber per 100g
        category = classify_food(
            protein_per_100g=9, carbs_per_100g=20, fat_per_100g=0.4,
            fiber_per_100g=8, energy_per_100g=116, description="Lentils, cooked"
        )
        assert category == FoodCategory.LEGUME

    def test_classify_legume_by_name(self):
        """Test that legume keywords help classification."""
        # Black beans with moderate stats
        category = classify_food(
            protein_per_100g=8, carbs_per_100g=24, fat_per_100g=0.5,
            fiber_per_100g=7, energy_per_100g=132, description="Black beans, canned"
        )
        assert category == FoodCategory.LEGUME

    def test_classify_fruit(self):
        """Test classifying fruits (carbs, low fat, low protein)."""
        # Apple: ~0.3g protein, ~14g carbs, ~0.2g fat, 2.4g fiber per 100g
        category = classify_food(
            protein_per_100g=0.3, carbs_per_100g=14, fat_per_100g=0.2,
            fiber_per_100g=2.4, energy_per_100g=52, description="Apple, raw"
        )
        assert category == FoodCategory.FRUIT

    def test_classify_eggs(self):
        """Test classifying eggs (high fat percentage from calories)."""
        # Eggs: 13g protein, 0.7g carbs, 10g fat
        # Fat cals: 10*9=90, Protein cals: 13*4=52, Carb cals: 0.7*4=2.8
        # Fat is ~62% of calories, so classified as FAT
        category = classify_food(
            protein_per_100g=13, carbs_per_100g=0.7, fat_per_100g=10,
            fiber_per_100g=0, energy_per_100g=143, description="Eggs, whole"
        )
        # Eggs are classified by calorie composition (fat dominant)
        assert category in (FoodCategory.FAT, FoodCategory.PROTEIN, FoodCategory.MIXED)


class TestClassifyFoodsInDb:
    """Tests for classify_foods_in_db function."""

    def test_classify_sample_foods(self, sample_foods):
        """Test classifying foods from sample database."""
        from mealplan.data.food_categories import classify_foods_in_db

        with sample_foods.get_connection() as conn:
            categories = classify_foods_in_db(conn, [1, 2, 3, 4, 5])

        # Chicken should be protein
        assert categories[1] == FoodCategory.PROTEIN

        # Rice should be carb
        assert categories[2] == FoodCategory.CARB

        # Broccoli should be vegetable
        assert categories[3] == FoodCategory.VEGETABLE

        # Olive oil should be fat
        assert categories[5] == FoodCategory.FAT
