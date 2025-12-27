"""Tests for multi-period (per-meal) optimization."""

from __future__ import annotations

import pytest

from mealplan.optimizer.models import NutrientConstraint
from mealplan.optimizer.multiperiod_constraints import MultiPeriodConstraintBuilder
from mealplan.optimizer.multiperiod_models import (
    DEFAULT_CALORIE_SPLITS,
    DEFAULT_TYPICAL_PORTIONS,
    EquiCalorieConstraint,
    FoodMealAffinity,
    MealCalorieTarget,
    MealConfig,
    MealNutrientConstraint,
    MealType,
    MultiPeriodRequest,
    derive_default_meal_configs,
)
from mealplan.optimizer.multiperiod_solver import solve_multiperiod_diet


class TestMealType:
    """Tests for MealType enum."""

    def test_meal_types(self):
        """Verify all expected meal types exist."""
        assert MealType.BREAKFAST.value == "breakfast"
        assert MealType.LUNCH.value == "lunch"
        assert MealType.DINNER.value == "dinner"
        assert MealType.SNACK.value == "snack"


class TestDeriveDefaultMealConfigs:
    """Tests for default meal configuration derivation."""

    def test_derives_four_meals(self):
        """Should create configs for all four meal types."""
        configs = derive_default_meal_configs(
            daily_cal_range=(2000, 2200),
            daily_nutrients=[],
        )
        assert len(configs) == 4
        meal_types = {c.meal_type for c in configs}
        assert meal_types == {
            MealType.BREAKFAST,
            MealType.LUNCH,
            MealType.DINNER,
            MealType.SNACK,
        }

    def test_calorie_splits(self):
        """Should split calories using 25/35/35/5 percentages."""
        configs = derive_default_meal_configs(
            daily_cal_range=(2000, 2000),  # Exact value for easy math
            daily_nutrients=[],
        )

        # Find each meal config
        breakfast = next(c for c in configs if c.meal_type == MealType.BREAKFAST)
        lunch = next(c for c in configs if c.meal_type == MealType.LUNCH)
        dinner = next(c for c in configs if c.meal_type == MealType.DINNER)
        snack = next(c for c in configs if c.meal_type == MealType.SNACK)

        assert breakfast.calorie_target.min_calories == pytest.approx(500)  # 25%
        assert lunch.calorie_target.min_calories == pytest.approx(700)  # 35%
        assert dinner.calorie_target.min_calories == pytest.approx(700)  # 35%
        assert snack.calorie_target.min_calories == pytest.approx(100)  # 5%

    def test_nutrient_splits(self):
        """Should split nutrient constraints proportionally."""
        configs = derive_default_meal_configs(
            daily_cal_range=(2000, 2200),
            daily_nutrients=[
                NutrientConstraint(nutrient_id=1003, min_value=150, max_value=200),
            ],
        )

        breakfast = next(c for c in configs if c.meal_type == MealType.BREAKFAST)
        # 25% of 150 = 37.5
        prot_constraint = breakfast.nutrient_constraints[0]
        assert prot_constraint.min_value == pytest.approx(37.5)
        assert prot_constraint.max_value == pytest.approx(50.0)

    def test_typical_portions(self):
        """Should set smaller typical portions for snacks."""
        configs = derive_default_meal_configs(
            daily_cal_range=(2000, 2200),
            daily_nutrients=[],
        )

        breakfast = next(c for c in configs if c.meal_type == MealType.BREAKFAST)
        snack = next(c for c in configs if c.meal_type == MealType.SNACK)

        assert breakfast.typical_portion == DEFAULT_TYPICAL_PORTIONS[MealType.BREAKFAST]
        assert snack.typical_portion == DEFAULT_TYPICAL_PORTIONS[MealType.SNACK]
        assert snack.typical_portion < breakfast.typical_portion


class TestMultiPeriodConstraintBuilder:
    """Tests for multi-period constraint matrix construction."""

    def test_variable_indexing(self, sample_foods):
        """Verify variable index mapping."""
        request = MultiPeriodRequest(
            meals=[
                MealConfig(meal_type=MealType.BREAKFAST),
                MealConfig(meal_type=MealType.LUNCH),
                MealConfig(meal_type=MealType.DINNER),
                MealConfig(meal_type=MealType.SNACK),
            ],
            explicit_food_ids=[1, 2, 3],  # Use sample foods
        )
        with sample_foods.get_connection() as conn:
            builder = MultiPeriodConstraintBuilder(conn, request)
            builder.build()

            # var_index(food_0, meal_0) = 0
            assert builder.var_index(0, 0) == 0
            # var_index(food_0, meal_1) = 1
            assert builder.var_index(0, 1) == 1
            # var_index(food_0, meal_3) = 3
            assert builder.var_index(0, 3) == 3
            # var_index(food_1, meal_0) = 4
            assert builder.var_index(1, 0) == 4
            # var_index(food_2, meal_2) = 10
            assert builder.var_index(2, 2) == 10

    def test_reverse_variable_indexing(self, sample_foods):
        """Verify reverse variable index mapping."""
        request = MultiPeriodRequest(
            meals=[
                MealConfig(meal_type=MealType.BREAKFAST),
                MealConfig(meal_type=MealType.LUNCH),
            ],
            explicit_food_ids=[1, 2],
        )
        with sample_foods.get_connection() as conn:
            builder = MultiPeriodConstraintBuilder(conn, request)
            builder.build()

            # Test round-trip
            for food_idx in range(2):
                for meal_idx in range(2):
                    var_idx = builder.var_index(food_idx, meal_idx)
                    recovered = builder.reverse_var_index(var_idx)
                    assert recovered == (food_idx, meal_idx)

    def test_builds_correct_dimensions(self, sample_foods):
        """Verify output dimensions match expected structure."""
        request = MultiPeriodRequest(
            meals=[
                MealConfig(meal_type=MealType.BREAKFAST),
                MealConfig(meal_type=MealType.LUNCH),
                MealConfig(meal_type=MealType.DINNER),
            ],
            explicit_food_ids=[1, 2, 3, 4, 5],  # 5 foods
        )
        with sample_foods.get_connection() as conn:
            builder = MultiPeriodConstraintBuilder(conn, request)
            data = builder.build()

            # 5 foods * 3 meals = 15 variables
            assert data["n_foods"] == 5
            assert data["n_meals"] == 3
            assert data["n_vars"] == 15

            # Check matrix shapes
            assert data["P"].shape == (15, 15)
            assert data["q"].shape == (15,)
            assert data["lb"].shape == (15,)
            assert data["ub"].shape == (15,)

    def test_per_meal_calorie_constraints(self, sample_foods):
        """Test that per-meal calorie constraints are built correctly."""
        request = MultiPeriodRequest(
            meals=[
                MealConfig(
                    meal_type=MealType.SNACK,
                    calorie_target=MealCalorieTarget(
                        meal=MealType.SNACK,
                        min_calories=50,
                        max_calories=200,
                    ),
                ),
            ],
            explicit_food_ids=[1, 2],
        )
        with sample_foods.get_connection() as conn:
            builder = MultiPeriodConstraintBuilder(conn, request)
            data = builder.build()

            # Should have constraints for snack min and max calories
            constraint_names = [c[0] for c in data["constraint_info"]]
            assert "snack_cal_min" in constraint_names
            assert "snack_cal_max" in constraint_names

    def test_daily_linking_constraints(self, sample_foods):
        """Test that daily linking constraints are built."""
        request = MultiPeriodRequest(
            daily_calorie_range=(1500, 2000),
            meals=[
                MealConfig(meal_type=MealType.BREAKFAST),
                MealConfig(meal_type=MealType.LUNCH),
            ],
            explicit_food_ids=[1, 2],
        )
        with sample_foods.get_connection() as conn:
            builder = MultiPeriodConstraintBuilder(conn, request)
            data = builder.build()

            constraint_names = [c[0] for c in data["constraint_info"]]
            assert "daily_cal_min" in constraint_names
            assert "daily_cal_max" in constraint_names

    def test_food_meal_affinity_bounds(self, sample_foods):
        """Test that food-meal affinity is enforced via upper bounds."""
        request = MultiPeriodRequest(
            meals=[
                MealConfig(meal_type=MealType.BREAKFAST),
                MealConfig(meal_type=MealType.SNACK),
            ],
            food_meal_affinities=[
                FoodMealAffinity(fdc_id=4, allowed_meals=[MealType.BREAKFAST]),  # Eggs only at breakfast
            ],
            explicit_food_ids=[1, 4],  # Chicken (1) and Eggs (4)
        )
        with sample_foods.get_connection() as conn:
            builder = MultiPeriodConstraintBuilder(conn, request)
            data = builder.build()

            # Find index for eggs (fdc_id=4)
            egg_idx = data["food_ids"].index(4)
            snack_idx = builder.meal_indices[MealType.SNACK]
            breakfast_idx = builder.meal_indices[MealType.BREAKFAST]

            # Eggs at snack should have ub=0
            eggs_snack_var = builder.var_index(egg_idx, snack_idx)
            assert data["ub"][eggs_snack_var] == 0.0

            # Eggs at breakfast should have normal ub
            eggs_breakfast_var = builder.var_index(egg_idx, breakfast_idx)
            assert data["ub"][eggs_breakfast_var] > 0

    def test_equicalorie_constraints(self, sample_foods):
        """Test that equi-calorie constraints are built correctly."""
        request = MultiPeriodRequest(
            meals=[
                MealConfig(meal_type=MealType.LUNCH),
                MealConfig(meal_type=MealType.DINNER),
            ],
            equicalorie_constraints=[
                EquiCalorieConstraint(
                    meal_a=MealType.LUNCH,
                    meal_b=MealType.DINNER,
                    tolerance=100,
                ),
            ],
            explicit_food_ids=[1, 2],
        )
        with sample_foods.get_connection() as conn:
            builder = MultiPeriodConstraintBuilder(conn, request)
            data = builder.build()

            constraint_names = [c[0] for c in data["constraint_info"]]
            # Should have both directions of the linearized constraint
            assert "equical_lunch_dinner_pos" in constraint_names
            assert "equical_lunch_dinner_neg" in constraint_names


class TestMultiPeriodSolver:
    """Integration tests for multi-period solver."""

    def test_basic_solve(self, sample_foods):
        """Test basic multi-meal optimization."""
        request = MultiPeriodRequest(
            daily_calorie_range=(800, 1500),
            meals=[
                MealConfig(
                    meal_type=MealType.BREAKFAST,
                    calorie_target=MealCalorieTarget(
                        meal=MealType.BREAKFAST,
                        min_calories=200,
                        max_calories=400,
                    ),
                ),
                MealConfig(
                    meal_type=MealType.LUNCH,
                    calorie_target=MealCalorieTarget(
                        meal=MealType.LUNCH,
                        min_calories=300,
                        max_calories=500,
                    ),
                ),
                MealConfig(
                    meal_type=MealType.DINNER,
                    calorie_target=MealCalorieTarget(
                        meal=MealType.DINNER,
                        min_calories=300,
                        max_calories=500,
                    ),
                ),
            ],
            explicit_food_ids=[1, 2, 3, 4, 5],
        )

        with sample_foods.get_connection() as conn:
            result = solve_multiperiod_diet(request, conn)

        assert result.success
        assert result.status == "optimal"
        assert len(result.meals) == 3

    def test_snack_calorie_limit_enforced(self, sample_foods):
        """Test that snack calorie limit is properly enforced."""
        request = MultiPeriodRequest(
            daily_calorie_range=(500, 1500),
            meals=[
                MealConfig(
                    meal_type=MealType.BREAKFAST,
                    calorie_target=MealCalorieTarget(
                        meal=MealType.BREAKFAST,
                        min_calories=200,
                        max_calories=600,
                    ),
                ),
                MealConfig(
                    meal_type=MealType.SNACK,
                    calorie_target=MealCalorieTarget(
                        meal=MealType.SNACK,
                        min_calories=0,
                        max_calories=200,  # This is the key constraint
                    ),
                    typical_portion=30.0,
                ),
            ],
            explicit_food_ids=[1, 2, 3, 4, 5],
        )

        with sample_foods.get_connection() as conn:
            result = solve_multiperiod_diet(request, conn)

        assert result.success
        snack = next(m for m in result.meals if m.meal_type == MealType.SNACK)
        # Allow small tolerance for numerical precision
        assert snack.total_calories <= 200 + 1

    def test_daily_linking_constraint_satisfied(self, sample_foods):
        """Test that daily totals satisfy linking constraints."""
        request = MultiPeriodRequest(
            daily_calorie_range=(800, 1000),
            meals=[
                MealConfig(meal_type=MealType.BREAKFAST),
                MealConfig(meal_type=MealType.LUNCH),
                MealConfig(meal_type=MealType.DINNER),
            ],
            explicit_food_ids=[1, 2, 3, 4, 5],
        )

        with sample_foods.get_connection() as conn:
            result = solve_multiperiod_diet(request, conn)

        assert result.success
        # Allow small tolerance
        assert result.daily_totals["calories"] >= 800 - 1
        assert result.daily_totals["calories"] <= 1000 + 1

    def test_food_meal_affinity_respected(self, sample_foods):
        """Test that food-meal affinity constraints are respected."""
        request = MultiPeriodRequest(
            daily_calorie_range=(500, 1500),
            meals=[
                MealConfig(
                    meal_type=MealType.BREAKFAST,
                    calorie_target=MealCalorieTarget(
                        meal=MealType.BREAKFAST,
                        min_calories=100,
                        max_calories=500,
                    ),
                ),
                MealConfig(
                    meal_type=MealType.LUNCH,
                    calorie_target=MealCalorieTarget(
                        meal=MealType.LUNCH,
                        min_calories=100,
                        max_calories=500,
                    ),
                ),
            ],
            food_meal_affinities=[
                # Eggs only allowed at breakfast
                FoodMealAffinity(fdc_id=4, allowed_meals=[MealType.BREAKFAST]),
            ],
            explicit_food_ids=[1, 2, 3, 4, 5],
        )

        with sample_foods.get_connection() as conn:
            result = solve_multiperiod_diet(request, conn)

        assert result.success

        # Check that eggs don't appear in lunch
        lunch = next(m for m in result.meals if m.meal_type == MealType.LUNCH)
        egg_fdc_ids = [f.fdc_id for f in lunch.foods if f.fdc_id == 4]
        assert len(egg_fdc_ids) == 0

    def test_infeasible_returns_diagnosis(self, sample_foods):
        """Test that infeasible problems return diagnosis."""
        # Create impossible constraints
        request = MultiPeriodRequest(
            daily_calorie_range=(5000, 5500),  # Way too high for sample foods
            meals=[
                MealConfig(
                    meal_type=MealType.BREAKFAST,
                    calorie_target=MealCalorieTarget(
                        meal=MealType.BREAKFAST,
                        min_calories=2000,
                        max_calories=2500,
                    ),
                ),
            ],
            explicit_food_ids=[1, 2, 3],  # Limited foods
        )

        with sample_foods.get_connection() as conn:
            result = solve_multiperiod_diet(request, conn)

        assert not result.success
        assert result.status == "infeasible"
        assert result.infeasibility_diagnosis is not None
        assert len(result.infeasibility_diagnosis.suggested_relaxations) > 0

    def test_no_foods_returns_error(self, temp_db):
        """Test that empty food pool returns error."""
        request = MultiPeriodRequest(
            include_tags=["nonexistent_tag"],  # No foods will match
            meals=[MealConfig(meal_type=MealType.BREAKFAST)],
        )

        with temp_db.get_connection() as conn:
            result = solve_multiperiod_diet(request, conn)

        assert not result.success
        assert result.status == "error"
        assert "No eligible foods" in result.message


class TestMealNutrientConstraint:
    """Tests for MealNutrientConstraint validation."""

    def test_requires_min_or_max(self):
        """Should raise if neither min nor max is set."""
        with pytest.raises(ValueError):
            MealNutrientConstraint(
                meal=MealType.BREAKFAST,
                nutrient_id=1003,
                min_value=None,
                max_value=None,
            )

    def test_accepts_min_only(self):
        """Should accept constraint with only min value."""
        nc = MealNutrientConstraint(
            meal=MealType.BREAKFAST,
            nutrient_id=1003,
            min_value=25,
        )
        assert nc.min_value == 25
        assert nc.max_value is None

    def test_accepts_max_only(self):
        """Should accept constraint with only max value."""
        nc = MealNutrientConstraint(
            meal=MealType.BREAKFAST,
            nutrient_id=1003,
            max_value=50,
        )
        assert nc.min_value is None
        assert nc.max_value == 50
