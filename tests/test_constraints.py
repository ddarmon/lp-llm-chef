"""Tests for constraint building."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from llmn.optimizer.constraints import ConstraintBuilder, load_profile_from_yaml
from llmn.optimizer.models import NutrientConstraint, OptimizationRequest


class TestConstraintBuilder:
    """Tests for ConstraintBuilder class."""

    def test_build_with_sample_foods(self, sample_foods):
        """Test building constraints from sample database."""
        request = OptimizationRequest(
            calorie_range=(1800, 2200),
            nutrient_constraints=[
                NutrientConstraint(nutrient_id=1003, min_value=100),
            ],
        )

        with sample_foods.get_connection() as conn:
            builder = ConstraintBuilder(conn, request)
            data = builder.build()

        # Should have 5 foods
        assert len(data["food_ids"]) == 5
        assert len(data["food_descriptions"]) == 5

        # Cost vector should be positive
        assert all(c > 0 for c in data["costs"])

        # Nutrient matrix should have correct shape
        assert data["nutrient_matrix"].shape[0] == 5  # 5 foods
        assert data["nutrient_matrix"].shape[1] == 2  # energy + protein

        # Check bounds
        assert len(data["food_bounds"]) == 5
        for min_g, max_g in data["food_bounds"]:
            assert min_g == 0
            assert max_g == 500  # default max

    def test_exclude_tags(self, sample_foods):
        """Test that foods with excluded tags are filtered out."""
        with sample_foods.get_connection() as conn:
            # Add exclude tag to one food
            conn.execute("INSERT INTO food_tags VALUES (1, 'exclude')")

            request = OptimizationRequest(
                exclude_tags=["exclude"],
            )
            builder = ConstraintBuilder(conn, request)
            data = builder.build()

        # Should have 4 foods (one excluded)
        assert len(data["food_ids"]) == 4
        assert 1 not in data["food_ids"]


class TestLoadProfileFromYAML:
    """Tests for YAML profile loading."""

    def test_load_valid_profile(self):
        """Test loading a valid YAML profile."""
        yaml_content = """
calories:
  min: 1800
  max: 2200

nutrients:
  protein:
    min: 150
  fiber:
    min: 30
  sodium:
    max: 2300

exclude_tags:
  - junk_food

options:
  max_grams_per_food: 400
  use_quadratic_penalty: true
  lambda_cost: 1.0
  lambda_deviation: 0.0005
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            request = load_profile_from_yaml(yaml_path)

            assert request.calorie_range == (1800, 2200)
            assert len(request.nutrient_constraints) == 3
            assert request.exclude_tags == ["junk_food"]
            assert request.max_grams_per_food == 400
            assert request.use_quadratic_penalty is True
            assert request.lambda_deviation == 0.0005

            # Check protein constraint
            protein = next(
                (nc for nc in request.nutrient_constraints if nc.nutrient_id == 1003),
                None,
            )
            assert protein is not None
            assert protein.min_value == 150
        finally:
            yaml_path.unlink()

    def test_load_profile_with_unknown_nutrient(self):
        """Test that unknown nutrient names raise an error."""
        yaml_content = """
nutrients:
  unknown_nutrient:
    min: 100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(KeyError):
                load_profile_from_yaml(yaml_path)
        finally:
            yaml_path.unlink()
