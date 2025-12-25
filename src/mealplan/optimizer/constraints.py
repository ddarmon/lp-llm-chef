"""Build constraint matrices for the optimizer."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from mealplan.data.nutrient_ids import get_nutrient_id
from mealplan.optimizer.models import (
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
        """
        self._load_eligible_foods()
        self._load_required_nutrients()

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
        }

    def _load_eligible_foods(self) -> None:
        """Load food IDs that pass tag filters (and have prices if not in feasibility mode)."""
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
