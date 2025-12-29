"""Tests for the optimization solver."""

from __future__ import annotations

import numpy as np
import pytest

from llmn.optimizer.models import NutrientConstraint, OptimizationRequest
from llmn.optimizer.solver import solve_diet_problem, solve_lp, solve_qp


class TestSolveLP:
    """Tests for linear programming solver."""

    def test_simple_optimization(self):
        """Basic case: minimize cost with calorie constraint only."""
        # Two foods: food 1 is expensive but calorie-dense, food 2 is cheap
        # Values are per gram (not per 100g)
        costs = np.array([0.01, 0.002])  # $0.01/g and $0.002/g
        nutrient_matrix = np.array([
            [1.65, 0.31],  # food 1: 165 kcal/100g, 31g protein/100g (like chicken)
            [1.30, 0.03],  # food 2: 130 kcal/100g, 3g protein/100g (like rice)
        ])  # Already per gram (kcal/g and g-protein/g)

        nutrient_mins = np.array([1800, -np.inf])  # 1800 kcal min, no protein min
        nutrient_maxs = np.array([2200, np.inf])   # 2200 kcal max
        food_bounds = [(0, 1000), (0, 1000)]  # Allow up to 1kg each

        result = solve_lp(costs, nutrient_matrix, nutrient_mins, nutrient_maxs, food_bounds)

        assert result["success"], f"Optimization failed: {result['message']}"
        assert result["x"] is not None

        # Verify calorie constraint satisfied
        total_calories = nutrient_matrix[:, 0] @ result["x"]
        assert total_calories >= 1800 - 0.1
        assert total_calories <= 2200 + 0.1

    def test_infeasible_constraints(self):
        """Test that infeasible constraints are detected."""
        costs = np.array([0.01])
        # Food with only 10 kcal per 100g - impossible to hit 2000 kcal in 500g
        nutrient_matrix = np.array([[0.1]])  # 10 kcal per 100g = 0.1 per gram
        nutrient_mins = np.array([2000.0])
        nutrient_maxs = np.array([np.inf])
        food_bounds = [(0, 500)]

        result = solve_lp(costs, nutrient_matrix, nutrient_mins, nutrient_maxs, food_bounds)

        assert not result["success"]


class TestSolveQP:
    """Tests for quadratic programming solver."""

    def test_qp_with_deviation_penalty(self):
        """Test that QP produces a solution."""
        costs = np.array([0.01, 0.002])
        nutrient_matrix = np.array([
            [1.65, 0.31],  # 165 kcal/100g (like chicken)
            [1.30, 0.03],  # 130 kcal/100g (like rice)
        ])  # Already per gram

        nutrient_mins = np.array([1800, -np.inf])
        nutrient_maxs = np.array([2200, np.inf])
        food_bounds = [(0, 1000), (0, 1000)]  # Allow up to 1kg each
        typical = np.array([300, 500])  # Typical consumption

        result = solve_qp(
            costs, nutrient_matrix, nutrient_mins, nutrient_maxs,
            food_bounds, typical, lambda_cost=1.0, lambda_deviation=0.001
        )

        assert result["success"], f"Optimization failed: {result['message']}"
        assert result["x"] is not None

        # Verify calorie constraint satisfied
        total_calories = nutrient_matrix[:, 0] @ result["x"]
        assert total_calories >= 1800 - 1
        assert total_calories <= 2200 + 1


class TestSolveDietProblem:
    """Integration tests for the full solver."""

    def test_with_sample_foods(self, sample_foods):
        """Test full optimization with sample database."""
        request = OptimizationRequest(
            calorie_range=(1800, 2200),
            nutrient_constraints=[
                NutrientConstraint(nutrient_id=1003, min_value=100),  # 100g protein min
            ],
            use_quadratic_penalty=False,  # Use LP for deterministic test
        )

        with sample_foods.get_connection() as conn:
            result = solve_diet_problem(request, conn)

        assert result.success
        assert result.status == "optimal"
        assert result.total_cost is not None
        assert len(result.foods) > 0

        # Verify calorie constraint
        calories = next(
            (n for n in result.nutrients if n.nutrient_id == 1008), None
        )
        assert calories is not None
        assert calories.amount >= 1800 - 1
        assert calories.amount <= 2200 + 1

    def test_no_eligible_foods(self, temp_db):
        """Test error when no foods have prices."""
        # temp_db has schema but no data
        request = OptimizationRequest()

        with temp_db.get_connection() as conn:
            result = solve_diet_problem(request, conn)

        assert not result.success
        assert result.status == "error"
        assert "No eligible foods" in result.message
