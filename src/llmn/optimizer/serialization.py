"""Serialization utilities for OptimizationRequest round-trip.

These functions ensure that an OptimizationRequest can be serialized to JSON
and deserialized back to an equivalent OptimizationRequest, preserving all
fields including tags, nutrient constraints, and options.
"""

from __future__ import annotations

from typing import Any

from llmn.data.nutrient_ids import NUTRIENT_NAMES, get_nutrient_id
from llmn.optimizer.models import (
    FoodConstraint,
    NutrientConstraint,
    OptimizationRequest,
)


def serialize_request(request: OptimizationRequest) -> dict[str, Any]:
    """Convert OptimizationRequest to a JSON-serializable dict.

    The output format matches the YAML profile format and is compatible
    with deserialize_request() for round-trip serialization.

    Args:
        request: The optimization request to serialize

    Returns:
        Dictionary that can be JSON-serialized and later deserialized
    """
    data: dict[str, Any] = {}

    # Serialize calorie range
    data["calories"] = {
        "min": request.calorie_range[0],
        "max": request.calorie_range[1],
    }

    # Serialize nutrient constraints using friendly names
    if request.nutrient_constraints:
        nutrients: dict[str, dict[str, float]] = {}
        for nc in request.nutrient_constraints:
            # Convert nutrient_id to name, fallback to id string if unknown
            name = NUTRIENT_NAMES.get(nc.nutrient_id, str(nc.nutrient_id))
            constraint: dict[str, float] = {}
            if nc.min_value is not None:
                constraint["min"] = nc.min_value
            if nc.max_value is not None:
                constraint["max"] = nc.max_value
            nutrients[name] = constraint
        data["nutrients"] = nutrients

    # Serialize tags
    if request.exclude_tags:
        data["exclude_tags"] = list(request.exclude_tags)
    if request.include_tags:
        data["include_tags"] = list(request.include_tags)

    # Serialize per-food limits
    if request.food_constraints:
        per_food_limits: dict[str, float] = {}
        for fc in request.food_constraints:
            if fc.max_grams is not None:
                per_food_limits[str(fc.fdc_id)] = fc.max_grams
        if per_food_limits:
            data["per_food_limits"] = per_food_limits

    # Serialize options
    options: dict[str, Any] = {
        "mode": request.mode,
        "max_grams_per_food": request.max_grams_per_food,
        "max_foods": request.max_foods,
        "use_quadratic_penalty": request.use_quadratic_penalty,
        "lambda_cost": request.lambda_cost,
        "lambda_deviation": request.lambda_deviation,
    }

    # Include sparse solver options if set
    if request.max_foods_in_solution is not None:
        options["max_foods_in_solution"] = request.max_foods_in_solution
    if request.min_grams_if_included != 50.0:  # Only if non-default
        options["min_grams_if_included"] = request.min_grams_if_included

    data["options"] = options

    # Include explicit food IDs if set
    if request.explicit_food_ids:
        data["explicit_food_ids"] = list(request.explicit_food_ids)

    return data


def deserialize_request(data: dict[str, Any]) -> OptimizationRequest:
    """Convert a JSON/dict back into an OptimizationRequest.

    This handles the same format as serialize_request() produces,
    as well as the YAML profile format used by load_profile_from_yaml().

    Args:
        data: Dictionary containing the serialized request

    Returns:
        OptimizationRequest reconstructed from the data
    """
    request = OptimizationRequest()

    # Parse calorie range
    if "calories" in data:
        cal_data = data["calories"]
        request.calorie_range = (
            float(cal_data.get("min", 0)),
            float(cal_data.get("max", 10000)),
        )

    # Parse nutrient constraints
    if "nutrients" in data:
        for nutrient_key, bounds in data["nutrients"].items():
            # Handle both string names and numeric IDs
            if isinstance(nutrient_key, int):
                nutrient_id = nutrient_key
            elif nutrient_key.isdigit():
                nutrient_id = int(nutrient_key)
            else:
                nutrient_id = get_nutrient_id(nutrient_key)

            min_val = bounds.get("min")
            max_val = bounds.get("max")

            # Convert to float if present
            if min_val is not None:
                min_val = float(min_val)
            if max_val is not None:
                max_val = float(max_val)

            request.nutrient_constraints.append(
                NutrientConstraint(
                    nutrient_id=nutrient_id,
                    min_value=min_val,
                    max_value=max_val,
                )
            )

    # Parse tags
    request.exclude_tags = list(data.get("exclude_tags", []))
    request.include_tags = list(data.get("include_tags", []))

    # Parse per-food limits
    if "per_food_limits" in data:
        for fdc_id_str, max_grams in data["per_food_limits"].items():
            request.food_constraints.append(
                FoodConstraint(
                    fdc_id=int(fdc_id_str),
                    max_grams=float(max_grams),
                )
            )

    # Parse options
    options = data.get("options", {})
    if "max_grams_per_food" in options:
        request.max_grams_per_food = float(options["max_grams_per_food"])
    if "use_quadratic_penalty" in options:
        request.use_quadratic_penalty = bool(options["use_quadratic_penalty"])
    if "lambda_cost" in options:
        request.lambda_cost = float(options["lambda_cost"])
    if "lambda_deviation" in options:
        request.lambda_deviation = float(options["lambda_deviation"])
    if "mode" in options:
        request.mode = str(options["mode"])
    if "max_foods" in options:
        request.max_foods = int(options["max_foods"])
    if "max_foods_in_solution" in options:
        request.max_foods_in_solution = int(options["max_foods_in_solution"])
    if "min_grams_if_included" in options:
        request.min_grams_if_included = float(options["min_grams_if_included"])

    # Parse explicit food IDs
    if "explicit_food_ids" in data:
        request.explicit_food_ids = [int(fid) for fid in data["explicit_food_ids"]]

    return request
