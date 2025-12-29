"""Tests for request serialization."""

from __future__ import annotations

from llmn.optimizer.models import NutrientConstraint, OptimizationRequest
from llmn.optimizer.serialization import deserialize_request, serialize_request


class TestRequestSerialization:
    """Tests for OptimizationRequest round-trip serialization."""

    def test_serialize_default_request(self):
        """Test serializing a default request."""
        request = OptimizationRequest()
        data = serialize_request(request)

        # Check calories are present (format may vary)
        assert "calories" in data
        assert data["calories"]["min"] == 1800
        assert data["calories"]["max"] == 2200
        assert data["options"]["mode"] == "feasibility"

    def test_roundtrip_with_constraints(self):
        """Test round-trip serialization preserves constraints."""
        request = OptimizationRequest(
            mode="feasibility",
            calorie_range=(1600, 1900),
            nutrient_constraints=[
                NutrientConstraint(nutrient_id=1003, min_value=150),
                NutrientConstraint(nutrient_id=1093, max_value=2300),
            ],
            include_tags=["staple", "protein"],
            exclude_tags=["junk_food"],
            max_grams_per_food=400,
            lambda_deviation=0.002,
        )

        data = serialize_request(request)
        restored = deserialize_request(data)

        assert restored.calorie_range == (1600, 1900)
        assert restored.mode == "feasibility"
        assert len(restored.nutrient_constraints) == 2
        assert restored.include_tags == ["staple", "protein"]
        assert restored.exclude_tags == ["junk_food"]
        assert restored.max_grams_per_food == 400
        assert restored.lambda_deviation == 0.002

    def test_roundtrip_with_explicit_foods(self):
        """Test round-trip with explicit food IDs."""
        request = OptimizationRequest(
            explicit_food_ids=[175167, 171287, 172421],
        )

        data = serialize_request(request)
        restored = deserialize_request(data)

        assert restored.explicit_food_ids == [175167, 171287, 172421]

    def test_deserialize_empty_dict(self):
        """Test deserializing an empty dict returns defaults."""
        restored = deserialize_request({})

        # Should get default values
        assert restored.calorie_range == (1800, 2200)
        assert restored.mode == "feasibility"
        assert len(restored.nutrient_constraints) == 0
