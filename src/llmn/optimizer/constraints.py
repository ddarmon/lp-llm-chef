"""Build constraint matrices for the optimizer."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from llmn.data.nutrient_ids import get_nutrient_id
from llmn.data.quality import detect_incomplete_foods, format_quality_warnings
from llmn.optimizer.models import (
    FoodConstraint,
    NutrientConstraint,
    OptimizationRequest,
)


class ConstraintBuilder:
    """Builds the constraint matrices for scipy.optimize."""

    def __init__(self, conn: sqlite3.Connection, request: OptimizationRequest):
        """Initialize the constraint builder.

        Args:
            conn: Database connection
            request: Optimization request specification
        """
        self.conn = conn
        self.request = request
        self._food_ids: list[int] = []
        self._nutrient_ids: list[int] = []
        self._total_eligible_foods: int = 0  # Before sampling
        self._was_sampled: bool = False

    def build(self) -> dict[str, Any]:
        """Build all matrices and vectors needed for optimization.

        Returns:
            Dict with:
                - food_ids: list[int] - ordered list of food IDs
                - food_descriptions: list[str] - descriptions for each food
                - costs: np.ndarray - cost per gram for each food
                - nutrient_matrix: np.ndarray - shape (n_foods, n_nutrients)
                - nutrient_mins: np.ndarray - min constraint per nutrient (-inf if none)
                - nutrient_maxs: np.ndarray - max constraint per nutrient (inf if none)
                - food_bounds: list[tuple] - (min, max) grams for each food
                - nutrient_ids: list[int] - ordered list of nutrient IDs
                - data_quality_warnings: list[str] - warnings about incomplete data
        """
        self._load_eligible_foods()
        self._load_required_nutrients()

        # Check for data quality issues
        quality_issues = detect_incomplete_foods(self.conn, self._food_ids)
        quality_warnings = format_quality_warnings(quality_issues)

        return {
            "food_ids": self._food_ids,
            "food_descriptions": self._get_food_descriptions(),
            "costs": self._build_cost_vector(),
            "nutrient_matrix": self._build_nutrient_matrix(),
            "nutrient_mins": self._build_nutrient_min_vector(),
            "nutrient_maxs": self._build_nutrient_max_vector(),
            "food_bounds": self._build_food_bounds(),
            "nutrient_ids": self._nutrient_ids,
            # Metadata for reporting
            "total_eligible_foods": self._total_eligible_foods,
            "was_sampled": self._was_sampled,
            "data_quality_warnings": quality_warnings,
        }

    def _load_eligible_foods(self) -> None:
        """Load food IDs that pass tag filters (and have prices if not in feasibility mode).

        If explicit_food_ids is set on the request, uses those directly (bypassing
        tag filtering but still applying exclude_tags and food_constraints).
        """
        # If explicit food IDs are provided, use those directly
        if self.request.explicit_food_ids:
            self._food_ids = list(self.request.explicit_food_ids)

            # Still apply exclude_tags if specified
            if self.request.exclude_tags:
                placeholders = ",".join("?" * len(self.request.exclude_tags))
                query = f"""
                    SELECT fdc_id FROM food_tags
                    WHERE tag IN ({placeholders})
                """
                cursor = self.conn.execute(query, self.request.exclude_tags)
                excluded_by_tag = {row[0] for row in cursor.fetchall()}
                self._food_ids = [fid for fid in self._food_ids if fid not in excluded_by_tag]

            # Apply food-specific exclusions
            excluded = {
                fc.fdc_id
                for fc in self.request.food_constraints
                if fc.max_grams == 0
            }
            self._food_ids = [fid for fid in self._food_ids if fid not in excluded]

            self._total_eligible_foods = len(self._food_ids)
            self._was_sampled = False
            return

        # Standard flow: load from database with tag filters
        # In feasibility mode, we don't need prices
        # In cost minimization mode, we require prices
        if self.request.mode == "feasibility":
            query = """
                SELECT DISTINCT f.fdc_id
                FROM foods f
                WHERE f.is_active = TRUE
            """
        else:
            query = """
                SELECT DISTINCT f.fdc_id
                FROM foods f
                INNER JOIN prices p ON f.fdc_id = p.fdc_id
                WHERE f.is_active = TRUE
            """
        params: list[Any] = []

        # Apply exclude_tags filter
        if self.request.exclude_tags:
            placeholders = ",".join("?" * len(self.request.exclude_tags))
            query += f"""
                AND f.fdc_id NOT IN (
                    SELECT fdc_id FROM food_tags WHERE tag IN ({placeholders})
                )
            """
            params.extend(self.request.exclude_tags)

        # Apply include_tags filter (food must have at least one of these tags)
        if self.request.include_tags:
            placeholders = ",".join("?" * len(self.request.include_tags))
            query += f"""
                AND f.fdc_id IN (
                    SELECT fdc_id FROM food_tags WHERE tag IN ({placeholders})
                )
            """
            params.extend(self.request.include_tags)

        cursor = self.conn.execute(query, params)
        self._food_ids = [row[0] for row in cursor.fetchall()]

        # Apply food-specific exclusions (max_grams = 0 or explicit exclusion)
        excluded = {
            fc.fdc_id
            for fc in self.request.food_constraints
            if fc.max_grams == 0
            or (fc.max_grams is None and fc.min_grams is None)
        }
        self._food_ids = [fid for fid in self._food_ids if fid not in excluded]

        # Track total before sampling
        self._total_eligible_foods = len(self._food_ids)

        # Sample if over the limit
        if len(self._food_ids) > self.request.max_foods:
            self._was_sampled = True
            self._food_ids = random.sample(self._food_ids, self.request.max_foods)

    def _load_required_nutrients(self) -> None:
        """Determine which nutrients we need to track."""
        # Always include energy (calories)
        required: set[int] = {1008}

        # Add nutrients from constraints
        for nc in self.request.nutrient_constraints:
            required.add(nc.nutrient_id)

        self._nutrient_ids = sorted(required)

    def _get_food_descriptions(self) -> list[str]:
        """Get descriptions for all food IDs.

        Returns:
            List of descriptions in same order as food_ids
        """
        if not self._food_ids:
            return []

        placeholders = ",".join("?" * len(self._food_ids))
        query = f"SELECT fdc_id, description FROM foods WHERE fdc_id IN ({placeholders})"
        cursor = self.conn.execute(query, self._food_ids)
        desc_map = {row[0]: row[1] for row in cursor.fetchall()}
        return [desc_map.get(fid, f"Food {fid}") for fid in self._food_ids]

    def _build_cost_vector(self) -> np.ndarray:
        """Build cost per gram vector.

        Returns:
            Array of shape (n_foods,) with cost per gram
        """
        if not self._food_ids:
            return np.array([])

        placeholders = ",".join("?" * len(self._food_ids))
        query = f"""
            SELECT fdc_id, price_per_100g / 100.0 as price_per_gram
            FROM prices
            WHERE fdc_id IN ({placeholders})
        """
        cursor = self.conn.execute(query, self._food_ids)
        price_map = {row[0]: row[1] for row in cursor.fetchall()}

        return np.array([price_map.get(fid, 0.0) for fid in self._food_ids])

    def _build_nutrient_matrix(self) -> np.ndarray:
        """Build nutrient matrix A where A[i,j] = nutrient j per gram of food i.

        USDA stores values per 100g, so we divide by 100.

        Returns:
            Array of shape (n_foods, n_nutrients)
        """
        n_foods = len(self._food_ids)
        n_nutrients = len(self._nutrient_ids)

        if n_foods == 0 or n_nutrients == 0:
            return np.zeros((n_foods, n_nutrients))

        matrix = np.zeros((n_foods, n_nutrients))

        # Build index lookups
        food_idx = {fid: i for i, fid in enumerate(self._food_ids)}
        nutrient_idx = {nid: j for j, nid in enumerate(self._nutrient_ids)}

        # Query all relevant food_nutrients
        food_placeholders = ",".join("?" * n_foods)
        nutrient_placeholders = ",".join("?" * n_nutrients)
        query = f"""
            SELECT fdc_id, nutrient_id, amount / 100.0 as amount_per_gram
            FROM food_nutrients
            WHERE fdc_id IN ({food_placeholders})
              AND nutrient_id IN ({nutrient_placeholders})
        """
        params = list(self._food_ids) + list(self._nutrient_ids)
        cursor = self.conn.execute(query, params)

        for row in cursor.fetchall():
            i = food_idx.get(row[0])
            j = nutrient_idx.get(row[1])
            if i is not None and j is not None:
                matrix[i, j] = row[2]

        return matrix

    def _build_nutrient_min_vector(self) -> np.ndarray:
        """Build vector of minimum constraints per nutrient.

        Returns:
            Array of shape (n_nutrients,) with -inf for no constraint
        """
        mins = np.full(len(self._nutrient_ids), -np.inf)

        # Handle calorie minimum from calorie_range
        if 1008 in self._nutrient_ids:
            energy_idx = self._nutrient_ids.index(1008)
            mins[energy_idx] = self.request.calorie_range[0]

        # Handle other nutrient constraints
        nutrient_idx = {nid: j for j, nid in enumerate(self._nutrient_ids)}
        for nc in self.request.nutrient_constraints:
            if nc.min_value is not None and nc.nutrient_id in nutrient_idx:
                mins[nutrient_idx[nc.nutrient_id]] = nc.min_value

        return mins

    def _build_nutrient_max_vector(self) -> np.ndarray:
        """Build vector of maximum constraints per nutrient.

        Returns:
            Array of shape (n_nutrients,) with inf for no constraint
        """
        maxs = np.full(len(self._nutrient_ids), np.inf)

        # Handle calorie maximum from calorie_range
        if 1008 in self._nutrient_ids:
            energy_idx = self._nutrient_ids.index(1008)
            maxs[energy_idx] = self.request.calorie_range[1]

        # Handle other nutrient constraints
        nutrient_idx = {nid: j for j, nid in enumerate(self._nutrient_ids)}
        for nc in self.request.nutrient_constraints:
            if nc.max_value is not None and nc.nutrient_id in nutrient_idx:
                maxs[nutrient_idx[nc.nutrient_id]] = nc.max_value

        return maxs

    def _build_food_bounds(self) -> list[tuple[float, float]]:
        """Build (min, max) bounds for each food.

        Returns:
            List of (min_grams, max_grams) tuples
        """
        default_max = self.request.max_grams_per_food

        # Build lookup from food constraints
        food_constraint_map = {fc.fdc_id: fc for fc in self.request.food_constraints}

        bounds = []
        for fid in self._food_ids:
            fc = food_constraint_map.get(fid)
            if fc:
                min_g = fc.min_grams if fc.min_grams is not None else 0.0
                max_g = fc.max_grams if fc.max_grams is not None else default_max
            else:
                min_g = 0.0
                max_g = default_max
            bounds.append((min_g, max_g))

        return bounds


def load_profile_from_yaml(yaml_path: Path) -> OptimizationRequest:
    """Parse a YAML constraint profile into an OptimizationRequest.

    Args:
        yaml_path: Path to the YAML profile file

    Returns:
        OptimizationRequest configured from the YAML

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If nutrient name is unknown
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

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
        for nutrient_name, bounds in data["nutrients"].items():
            nutrient_id = get_nutrient_id(nutrient_name)
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
    request.exclude_tags = data.get("exclude_tags", [])
    request.include_tags = data.get("include_tags", [])

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

    return request


def deserialize_profile_json(data: dict) -> OptimizationRequest:
    """Convert a JSON/dict profile into an OptimizationRequest.

    This is the dict-based equivalent of load_profile_from_yaml(),
    used when loading profiles from the database.

    Args:
        data: Profile data as a dictionary (parsed from JSON)

    Returns:
        OptimizationRequest configured from the data
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
        for nutrient_name, bounds in data["nutrients"].items():
            nutrient_id = get_nutrient_id(nutrient_name)
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
    request.exclude_tags = data.get("exclude_tags", [])
    request.include_tags = data.get("include_tags", [])

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

    return request


def has_meals_section(yaml_path: Path) -> bool:
    """Check if a YAML profile contains a meals section (multi-period).

    Args:
        yaml_path: Path to the YAML profile file

    Returns:
        True if profile has a 'meals' dict section
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return "meals" in data and isinstance(data.get("meals"), dict)


def load_multiperiod_profile_from_yaml(yaml_path: Path):
    """Parse a YAML profile with meals section into MultiPeriodRequest.

    Example YAML format:
        calories:
          min: 1800
          max: 2000

        nutrients:
          protein: {min: 150}

        meals:
          breakfast:
            calories: {min: 400, max: 550}
            nutrients:
              protein: {min: 25}
          snack:
            calories: {min: 0, max: 200}

        equicalorie:
          meals: [lunch, dinner]
          tolerance: 100

        food_meal_rules:
          170567: [snack]  # Explicit FDC ID -> allowed meals

        options:
          max_grams_per_food_per_meal: 300

    Args:
        yaml_path: Path to the YAML profile file

    Returns:
        MultiPeriodRequest configured from the YAML

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If nutrient name is unknown
    """
    from llmn.optimizer.multiperiod_models import (
        EquiCalorieConstraint,
        FoodMealAffinity,
        MealCalorieTarget,
        MealConfig,
        MealNutrientConstraint,
        MealType,
        MultiPeriodRequest,
        derive_default_meal_configs,
    )

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Parse daily calorie range
    daily_cal_range = (
        float(data.get("calories", {}).get("min", 0)),
        float(data.get("calories", {}).get("max", 10000)),
    )

    # Parse daily nutrient constraints
    daily_nutrients = []
    for nutrient_name, bounds in data.get("nutrients", {}).items():
        nutrient_id = get_nutrient_id(nutrient_name)
        min_val = bounds.get("min")
        max_val = bounds.get("max")
        if min_val is not None:
            min_val = float(min_val)
        if max_val is not None:
            max_val = float(max_val)
        daily_nutrients.append(
            NutrientConstraint(
                nutrient_id=nutrient_id,
                min_value=min_val,
                max_value=max_val,
            )
        )

    # Parse meals section
    meal_configs = []
    meals_data = data.get("meals", {})

    for meal_name, meal_spec in meals_data.items():
        try:
            meal_type = MealType(meal_name.lower())
        except ValueError:
            # Skip unknown meal types
            continue

        # Per-meal calories
        cal_target = None
        if "calories" in meal_spec:
            cal_spec = meal_spec["calories"]
            cal_target = MealCalorieTarget(
                meal=meal_type,
                min_calories=float(cal_spec.get("min", 0)),
                max_calories=float(cal_spec.get("max", float("inf"))),
            )

        # Per-meal nutrients
        meal_nutrients = []
        for nutrient_name, bounds in meal_spec.get("nutrients", {}).items():
            nutrient_id = get_nutrient_id(nutrient_name)
            min_val = bounds.get("min")
            max_val = bounds.get("max")
            if min_val is not None:
                min_val = float(min_val)
            if max_val is not None:
                max_val = float(max_val)
            meal_nutrients.append(
                MealNutrientConstraint(
                    meal=meal_type,
                    nutrient_id=nutrient_id,
                    min_value=min_val,
                    max_value=max_val,
                )
            )

        # Typical portion for this meal type
        typical = 100.0
        if meal_type == MealType.SNACK:
            typical = 30.0
        typical = float(meal_spec.get("typical_portion", typical))

        meal_configs.append(
            MealConfig(
                meal_type=meal_type,
                calorie_target=cal_target,
                nutrient_constraints=meal_nutrients,
                typical_portion=typical,
            )
        )

    # If no meals specified, derive defaults from daily constraints
    if not meal_configs:
        meal_configs = derive_default_meal_configs(daily_cal_range, daily_nutrients)

    # Parse equicalorie constraints
    # Supports both list format (from TRY_ME.md) and dict format
    equicalorie_constraints = []
    if "equicalorie" in data:
        eq_data = data["equicalorie"]
        # Handle list format: equicalorie: [{ meals: [...], tolerance: 100 }]
        if isinstance(eq_data, list):
            for eq_item in eq_data:
                eq_meals = eq_item.get("meals", [])
                if len(eq_meals) == 2:
                    try:
                        equicalorie_constraints.append(
                            EquiCalorieConstraint(
                                meal_a=MealType(eq_meals[0].lower()),
                                meal_b=MealType(eq_meals[1].lower()),
                                tolerance=float(eq_item.get("tolerance", 100)),
                            )
                        )
                    except ValueError:
                        pass  # Skip invalid meal types
        # Handle dict format: equicalorie: { meals: [...], tolerance: 100 }
        else:
            eq_meals = eq_data.get("meals", [])
            if len(eq_meals) == 2:
                try:
                    equicalorie_constraints.append(
                        EquiCalorieConstraint(
                            meal_a=MealType(eq_meals[0].lower()),
                            meal_b=MealType(eq_meals[1].lower()),
                            tolerance=float(eq_data.get("tolerance", 100)),
                        )
                    )
                except ValueError:
                    pass  # Skip invalid meal types

    # Parse food-meal affinities (explicit FDC IDs only)
    food_meal_affinities = []
    for fdc_id_str, allowed_meals in data.get("food_meal_rules", {}).items():
        try:
            fdc_id = int(fdc_id_str)
            meal_types = [MealType(m.lower()) for m in allowed_meals]
            food_meal_affinities.append(
                FoodMealAffinity(fdc_id=fdc_id, allowed_meals=meal_types)
            )
        except (ValueError, KeyError):
            pass  # Skip invalid entries

    # Parse options
    options = data.get("options", {})

    return MultiPeriodRequest(
        daily_calorie_range=daily_cal_range,
        daily_nutrient_constraints=daily_nutrients,
        meals=meal_configs,
        equicalorie_constraints=equicalorie_constraints,
        food_meal_affinities=food_meal_affinities,
        mode=str(options.get("mode", "feasibility")),
        exclude_tags=data.get("exclude_tags", []),
        include_tags=data.get("include_tags", []),
        max_grams_per_food_per_meal=float(
            options.get("max_grams_per_food_per_meal", 300)
        ),
        max_grams_per_food_daily=float(
            options.get("max_grams_per_food_daily", 500)
        ),
        max_foods=int(options.get("max_foods", 300)),
        lambda_cost=float(options.get("lambda_cost", 1.0)),
        lambda_deviation=float(options.get("lambda_deviation", 0.001)),
    )
