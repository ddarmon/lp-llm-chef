"""Tests for meal slot allocation."""

from __future__ import annotations

from llmn.export.meal_allocator import (
    allocate_to_meals,
    get_meal_affinity,
)
from llmn.optimizer.models import FoodResult


class TestMealAffinity:
    """Tests for get_meal_affinity function."""

    def test_egg_is_breakfast(self):
        """Test that eggs are associated with breakfast."""
        affinity = get_meal_affinity("Eggs, whole, raw")
        assert "breakfast" in affinity

    def test_chicken_is_lunch_dinner(self):
        """Test that chicken is associated with lunch and dinner."""
        affinity = get_meal_affinity("Chicken breast, boneless")
        assert "lunch" in affinity
        assert "dinner" in affinity

    def test_fish_is_lunch_dinner(self):
        """Test that fish is associated with lunch and dinner."""
        affinity = get_meal_affinity("Salmon, Atlantic, raw")
        assert "lunch" in affinity
        assert "dinner" in affinity

    def test_beans_are_lunch_dinner(self):
        """Test that beans are associated with lunch and dinner."""
        affinity = get_meal_affinity("Black beans, canned")
        assert "lunch" in affinity
        assert "dinner" in affinity

    def test_almonds_are_snack(self):
        """Test that nuts are associated with snack."""
        affinity = get_meal_affinity("Almonds, raw")
        assert "snack" in affinity

    def test_unknown_food_defaults(self):
        """Test that unknown foods can go in any meal."""
        affinity = get_meal_affinity("Some unknown food item")
        assert "breakfast" in affinity
        assert "lunch" in affinity
        assert "dinner" in affinity


class TestAllocateToMeals:
    """Tests for allocate_to_meals function."""

    def test_allocate_simple_foods(self):
        """Test allocating a simple set of foods."""
        foods = [
            FoodResult(
                fdc_id=1,
                description="Eggs, whole, raw",
                grams=200,
                cost=0.80,
                nutrients={1008: 143, 1003: 13},  # per 100g
            ),
            FoodResult(
                fdc_id=2,
                description="Chicken breast, raw",
                grams=300,
                cost=2.40,
                nutrients={1008: 165, 1003: 31},
            ),
            FoodResult(
                fdc_id=3,
                description="Almonds, raw",
                grams=50,
                cost=0.75,
                nutrients={1008: 579, 1003: 21},
            ),
        ]

        meals = allocate_to_meals(foods, total_calories=2000)

        # Check we got all meal slots
        meal_names = [m.name for m in meals]
        assert "breakfast" in meal_names
        assert "lunch" in meal_names
        assert "dinner" in meal_names
        assert "snack" in meal_names

        # Eggs should be in breakfast
        breakfast = next(m for m in meals if m.name == "breakfast")
        breakfast_foods = [f["description"] for f in breakfast.foods]
        assert any("Eggs" in desc for desc in breakfast_foods)

        # Almonds should be in snack
        snack = next(m for m in meals if m.name == "snack")
        snack_foods = [f["description"] for f in snack.foods]
        assert any("Almonds" in desc for desc in snack_foods)

    def test_empty_foods(self):
        """Test allocating with no foods."""
        meals = allocate_to_meals([], total_calories=2000)

        # Should still have meal slots, just empty
        assert len(meals) == 4
        for meal in meals:
            assert len(meal.foods) == 0

    def test_meal_calorie_targets(self):
        """Test that meal calorie targets are set correctly."""
        meals = allocate_to_meals([], total_calories=2000)

        # Default structure is 25/35/35/5
        breakfast = next(m for m in meals if m.name == "breakfast")
        assert breakfast.target_calories == 500  # 25% of 2000

        lunch = next(m for m in meals if m.name == "lunch")
        assert lunch.target_calories == 700  # 35% of 2000

        dinner = next(m for m in meals if m.name == "dinner")
        assert dinner.target_calories == 700  # 35% of 2000

        snack = next(m for m in meals if m.name == "snack")
        assert snack.target_calories == 100  # 5% of 2000
