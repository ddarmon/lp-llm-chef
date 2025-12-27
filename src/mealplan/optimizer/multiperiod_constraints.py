"""Build constraint matrices for multi-period (per-meal) optimization.

This module constructs the expanded constraint matrices needed for optimizing
food allocation across multiple meals (breakfast, lunch, dinner, snack).

Decision variables: x_{i,m} = grams of food i in meal m
Variable indexing: var_index(food_i, meal_m) = i * n_meals + m
Total variables: n_foods * n_meals
"""

from __future__ import annotations

import random
import sqlite3
from typing import Any, Optional

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

from mealplan.data.quality import detect_incomplete_foods, format_quality_warnings
from mealplan.optimizer.multiperiod_models import (
    MealConfig,
    MealType,
    MultiPeriodRequest,
    derive_default_meal_configs,
)


class MultiPeriodConstraintBuilder:
    """Builds constraint matrices for multi-period (per-meal) optimization.

    This builder creates the expanded QP problem where each food can appear
    in each meal independently, with per-meal and daily constraints.

    Variable layout: [food_0_meal_0, food_0_meal_1, ..., food_0_meal_M,
                      food_1_meal_0, food_1_meal_1, ..., food_1_meal_M,
                      ...]

    Attributes:
        conn: Database connection
        request: Multi-period optimization request
        n_meals: Number of meal slots
        meal_indices: Map from MealType to index
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        request: MultiPeriodRequest,
    ):
        """Initialize the multi-period constraint builder.

        Args:
            conn: Database connection
            request: Multi-period optimization request
        """
        self.conn = conn
        self.request = request

        # Ensure we have meal configs (derive defaults if needed)
        if not request.meals:
            request.meals = derive_default_meal_configs(
                request.daily_calorie_range,
                request.daily_nutrient_constraints,
            )

        self.n_meals = len(request.meals)
        self.meal_indices: dict[MealType, int] = {
            config.meal_type: idx for idx, config in enumerate(request.meals)
        }

        # These will be populated by build()
        self._food_ids: list[int] = []
        self._food_descriptions: list[str] = []
        self._nutrient_ids: list[int] = []
        self._nutrient_matrix: np.ndarray = np.array([])
        self._costs: np.ndarray = np.array([])
        self._n_foods: int = 0
        self._n_vars: int = 0
        self._total_eligible_foods: int = 0
        self._was_sampled: bool = False

        # Constraint tracking for debugging and KKT analysis
        self._constraint_info: list[tuple[str, float]] = []

    def var_index(self, food_idx: int, meal_idx: int) -> int:
        """Map (food, meal) indices to flat variable index.

        Args:
            food_idx: Index into _food_ids (0 to n_foods-1)
            meal_idx: Index into meals (0 to n_meals-1)

        Returns:
            Flat index into the solution vector
        """
        return food_idx * self.n_meals + meal_idx

    def reverse_var_index(self, var_idx: int) -> tuple[int, int]:
        """Map flat variable index back to (food, meal) indices.

        Args:
            var_idx: Flat index into the solution vector

        Returns:
            Tuple of (food_idx, meal_idx)
        """
        food_idx = var_idx // self.n_meals
        meal_idx = var_idx % self.n_meals
        return food_idx, meal_idx

    def build(self) -> dict[str, Any]:
        """Build all matrices and vectors needed for multi-period optimization.

        Returns:
            Dict with:
                - food_ids: list[int] - ordered list of food IDs
                - food_descriptions: list[str] - descriptions for each food
                - n_foods: int - number of foods
                - n_meals: int - number of meals
                - n_vars: int - total variables (n_foods * n_meals)
                - P: sparse CSC matrix - QP Hessian
                - q: ndarray - QP linear term
                - G: sparse CSC matrix - inequality constraints
                - h: ndarray - inequality RHS
                - lb: ndarray - lower bounds
                - ub: ndarray - upper bounds
                - nutrient_ids: list[int] - ordered list of nutrient IDs
                - nutrient_matrix: ndarray - (n_foods, n_nutrients) per-gram values
                - costs: ndarray - cost per gram for each food
                - constraint_info: list of (name, bound) for each constraint row
                - meal_configs: list of MealConfig objects
                - data_quality_warnings: list[str]
        """
        self._load_foods_and_nutrients()

        self._n_vars = self._n_foods * self.n_meals

        # Handle edge case of no foods
        if self._n_foods == 0:
            return {
                "food_ids": [],
                "food_descriptions": [],
                "n_foods": 0,
                "n_meals": self.n_meals,
                "n_vars": 0,
                "P": csc_matrix((0, 0)),
                "q": np.array([]),
                "G": csc_matrix((0, 0)),
                "h": np.array([]),
                "lb": np.array([]),
                "ub": np.array([]),
                "nutrient_ids": self._nutrient_ids,
                "nutrient_matrix": np.array([]).reshape(0, len(self._nutrient_ids)),
                "costs": np.array([]),
                "constraint_info": [],
                "meal_configs": self.request.meals,
                "total_eligible_foods": 0,
                "was_sampled": False,
                "data_quality_warnings": [],
            }

        # Check for data quality issues
        quality_issues = detect_incomplete_foods(self.conn, self._food_ids)
        quality_warnings = format_quality_warnings(quality_issues)

        # Build matrices
        G, h = self._build_inequality_constraints()

        return {
            "food_ids": self._food_ids,
            "food_descriptions": self._food_descriptions,
            "n_foods": self._n_foods,
            "n_meals": self.n_meals,
            "n_vars": self._n_vars,
            "P": self._build_hessian(),
            "q": self._build_linear_term(),
            "G": G,
            "h": h,
            "lb": self._build_lower_bounds(),
            "ub": self._build_upper_bounds(),
            "nutrient_ids": self._nutrient_ids,
            "nutrient_matrix": self._nutrient_matrix,
            "costs": self._costs,
            "constraint_info": self._constraint_info,
            "meal_configs": self.request.meals,
            "total_eligible_foods": self._total_eligible_foods,
            "was_sampled": self._was_sampled,
            "data_quality_warnings": quality_warnings,
        }

    def _load_foods_and_nutrients(self) -> None:
        """Load eligible foods and build the nutrient matrix.

        This reuses the food loading logic from the single-period builder,
        adapted for multi-period requests.
        """
        # Determine required nutrients
        required_nutrients: set[int] = {1008}  # Always need energy (calories)

        # Add nutrients from daily constraints
        for nc in self.request.daily_nutrient_constraints:
            required_nutrients.add(nc.nutrient_id)

        # Add nutrients from per-meal constraints
        for meal_config in self.request.meals:
            for mnc in meal_config.nutrient_constraints:
                required_nutrients.add(mnc.nutrient_id)

        self._nutrient_ids = sorted(required_nutrients)

        # Load eligible foods
        if self.request.explicit_food_ids:
            # Use explicit food IDs directly
            self._food_ids = list(self.request.explicit_food_ids)

            # Still apply exclude_tags if specified
            if self.request.exclude_tags:
                placeholders = ",".join("?" * len(self.request.exclude_tags))
                query = f"""
                    SELECT fdc_id FROM food_tags
                    WHERE tag IN ({placeholders})
                """
                cursor = self.conn.execute(query, self.request.exclude_tags)
                excluded = {row[0] for row in cursor.fetchall()}
                self._food_ids = [fid for fid in self._food_ids if fid not in excluded]

            # Filter out foods missing calorie data
            if self._food_ids:
                placeholders = ",".join("?" * len(self._food_ids))
                query = f"""
                    SELECT DISTINCT fdc_id FROM food_nutrients
                    WHERE fdc_id IN ({placeholders})
                      AND nutrient_id = 1008
                      AND amount > 0
                """
                cursor = self.conn.execute(query, self._food_ids)
                foods_with_calories = {row[0] for row in cursor.fetchall()}
                self._food_ids = [fid for fid in self._food_ids if fid in foods_with_calories]

            self._total_eligible_foods = len(self._food_ids)
            self._was_sampled = False
        else:
            # Standard flow: load from database with tag filters
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

            # Apply include_tags filter
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

            # Filter out foods missing calorie data (nutrient_id 1008)
            # These would break optimization by appearing to have 0 calories
            if self._food_ids:
                placeholders = ",".join("?" * len(self._food_ids))
                query = f"""
                    SELECT DISTINCT fdc_id FROM food_nutrients
                    WHERE fdc_id IN ({placeholders})
                      AND nutrient_id = 1008
                      AND amount > 0
                """
                cursor = self.conn.execute(query, self._food_ids)
                foods_with_calories = {row[0] for row in cursor.fetchall()}
                self._food_ids = [fid for fid in self._food_ids if fid in foods_with_calories]

            self._total_eligible_foods = len(self._food_ids)

            # Sample if over the limit
            if len(self._food_ids) > self.request.max_foods:
                self._was_sampled = True
                self._food_ids = random.sample(self._food_ids, self.request.max_foods)

        self._n_foods = len(self._food_ids)

        # Load food descriptions
        if self._food_ids:
            placeholders = ",".join("?" * len(self._food_ids))
            query = f"SELECT fdc_id, description FROM foods WHERE fdc_id IN ({placeholders})"
            cursor = self.conn.execute(query, self._food_ids)
            desc_map = {row[0]: row[1] for row in cursor.fetchall()}
            self._food_descriptions = [
                desc_map.get(fid, f"Food {fid}") for fid in self._food_ids
            ]
        else:
            self._food_descriptions = []

        # Build nutrient matrix (n_foods x n_nutrients), values per gram
        self._nutrient_matrix = self._build_nutrient_matrix()

        # Load costs
        self._costs = self._build_cost_vector()

    def _build_nutrient_matrix(self) -> np.ndarray:
        """Build nutrient matrix where A[i,j] = nutrient j per gram of food i.

        Returns:
            Array of shape (n_foods, n_nutrients)
        """
        n_nutrients = len(self._nutrient_ids)

        if self._n_foods == 0 or n_nutrients == 0:
            return np.zeros((self._n_foods, n_nutrients))

        matrix = np.zeros((self._n_foods, n_nutrients))

        food_idx = {fid: i for i, fid in enumerate(self._food_ids)}
        nutrient_idx = {nid: j for j, nid in enumerate(self._nutrient_ids)}

        food_placeholders = ",".join("?" * self._n_foods)
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

    def _build_cost_vector(self) -> np.ndarray:
        """Build cost per gram vector.

        Returns:
            Array of shape (n_foods,) with cost per gram
        """
        if self._n_foods == 0:
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

    def _build_hessian(self) -> csc_matrix:
        """Build QP Hessian P for objective 0.5 * x'Px.

        For multi-period, the diagonal varies by meal type to encourage
        smaller portions for snacks vs main meals.

        Returns:
            Sparse diagonal matrix of shape (n_vars, n_vars)
        """
        P = lil_matrix((self._n_vars, self._n_vars))

        for i in range(self._n_foods):
            for m, meal_config in enumerate(self.request.meals):
                var_idx = self.var_index(i, m)
                # Scale by 2 for standard QP form (0.5 * x'Px)
                P[var_idx, var_idx] = 2 * self.request.lambda_deviation

        return P.tocsc()

    def _build_linear_term(self) -> np.ndarray:
        """Build QP linear term q for objective 0.5 * x'Px + q'x.

        q = lambda_cost * c - 2 * lambda_deviation * x_bar
        where x_bar varies by meal type (smaller for snacks).

        Returns:
            Array of shape (n_vars,)
        """
        q = np.zeros(self._n_vars)

        for i in range(self._n_foods):
            cost_per_gram = self._costs[i]
            for m, meal_config in enumerate(self.request.meals):
                var_idx = self.var_index(i, m)
                typical = meal_config.typical_portion
                q[var_idx] = (
                    self.request.lambda_cost * cost_per_gram
                    - 2 * self.request.lambda_deviation * typical
                )

        return q

    def _build_inequality_constraints(self) -> tuple[csc_matrix, np.ndarray]:
        """Build G matrix and h vector for inequality constraints G @ x <= h.

        Constraint types:
        1. Per-meal calorie min/max
        2. Per-meal nutrient min/max
        3. Daily linking constraints (sum across meals)
        4. Equi-calorie constraints (linearized absolute value)

        Returns:
            Tuple of (G sparse matrix, h vector)
        """
        self._constraint_info = []
        rows: list[tuple[np.ndarray, float, str]] = []

        # Get calorie column index
        cal_idx = self._nutrient_ids.index(1008) if 1008 in self._nutrient_ids else 0

        # 1. Per-meal calorie constraints
        for m, meal_config in enumerate(self.request.meals):
            if meal_config.calorie_target:
                target = meal_config.calorie_target
                meal_name = meal_config.meal_type.value

                # Min: -sum_i(E_i * x_{i,m}) <= -min_cal
                if target.min_calories > 0:
                    row = np.zeros(self._n_vars)
                    for i in range(self._n_foods):
                        var_idx = self.var_index(i, m)
                        row[var_idx] = -self._nutrient_matrix[i, cal_idx]
                    rows.append((row, -target.min_calories, f"{meal_name}_cal_min"))

                # Max: sum_i(E_i * x_{i,m}) <= max_cal
                if target.max_calories < float("inf"):
                    row = np.zeros(self._n_vars)
                    for i in range(self._n_foods):
                        var_idx = self.var_index(i, m)
                        row[var_idx] = self._nutrient_matrix[i, cal_idx]
                    rows.append((row, target.max_calories, f"{meal_name}_cal_max"))

        # 2. Per-meal nutrient constraints
        for m, meal_config in enumerate(self.request.meals):
            meal_name = meal_config.meal_type.value
            for mnc in meal_config.nutrient_constraints:
                if mnc.nutrient_id not in self._nutrient_ids:
                    continue
                n_idx = self._nutrient_ids.index(mnc.nutrient_id)

                if mnc.min_value is not None:
                    row = np.zeros(self._n_vars)
                    for i in range(self._n_foods):
                        var_idx = self.var_index(i, m)
                        row[var_idx] = -self._nutrient_matrix[i, n_idx]
                    rows.append(
                        (row, -mnc.min_value, f"{meal_name}_n{mnc.nutrient_id}_min")
                    )

                if mnc.max_value is not None:
                    row = np.zeros(self._n_vars)
                    for i in range(self._n_foods):
                        var_idx = self.var_index(i, m)
                        row[var_idx] = self._nutrient_matrix[i, n_idx]
                    rows.append(
                        (row, mnc.max_value, f"{meal_name}_n{mnc.nutrient_id}_max")
                    )

        # 3. Daily linking constraints (sum over all meals)
        # Calories
        cal_min, cal_max = self.request.daily_calorie_range

        # Daily cal min: -sum_{i,m}(E_i * x_{i,m}) <= -cal_min
        row = np.zeros(self._n_vars)
        for i in range(self._n_foods):
            for m in range(self.n_meals):
                var_idx = self.var_index(i, m)
                row[var_idx] = -self._nutrient_matrix[i, cal_idx]
        rows.append((row, -cal_min, "daily_cal_min"))

        # Daily cal max: sum_{i,m}(E_i * x_{i,m}) <= cal_max
        row = np.zeros(self._n_vars)
        for i in range(self._n_foods):
            for m in range(self.n_meals):
                var_idx = self.var_index(i, m)
                row[var_idx] = self._nutrient_matrix[i, cal_idx]
        rows.append((row, cal_max, "daily_cal_max"))

        # Daily nutrient constraints
        for nc in self.request.daily_nutrient_constraints:
            if nc.nutrient_id not in self._nutrient_ids:
                continue
            n_idx = self._nutrient_ids.index(nc.nutrient_id)

            if nc.min_value is not None:
                row = np.zeros(self._n_vars)
                for i in range(self._n_foods):
                    for m in range(self.n_meals):
                        var_idx = self.var_index(i, m)
                        row[var_idx] = -self._nutrient_matrix[i, n_idx]
                rows.append((row, -nc.min_value, f"daily_n{nc.nutrient_id}_min"))

            if nc.max_value is not None:
                row = np.zeros(self._n_vars)
                for i in range(self._n_foods):
                    for m in range(self.n_meals):
                        var_idx = self.var_index(i, m)
                        row[var_idx] = self._nutrient_matrix[i, n_idx]
                rows.append((row, nc.max_value, f"daily_n{nc.nutrient_id}_max"))

        # 4. Equi-calorie constraints (linearized)
        for eq in self.request.equicalorie_constraints:
            m_a = self.meal_indices.get(eq.meal_a)
            m_b = self.meal_indices.get(eq.meal_b)
            if m_a is None or m_b is None:
                continue

            # cal_a - cal_b <= tolerance
            row = np.zeros(self._n_vars)
            for i in range(self._n_foods):
                e = self._nutrient_matrix[i, cal_idx]
                row[self.var_index(i, m_a)] = e
                row[self.var_index(i, m_b)] = -e
            rows.append(
                (row, eq.tolerance, f"equical_{eq.meal_a.value}_{eq.meal_b.value}_pos")
            )

            # -(cal_a - cal_b) <= tolerance  =>  cal_b - cal_a <= tolerance
            row = np.zeros(self._n_vars)
            for i in range(self._n_foods):
                e = self._nutrient_matrix[i, cal_idx]
                row[self.var_index(i, m_a)] = -e
                row[self.var_index(i, m_b)] = e
            rows.append(
                (row, eq.tolerance, f"equical_{eq.meal_a.value}_{eq.meal_b.value}_neg")
            )

        # Build sparse matrix
        if not rows:
            return csc_matrix((0, self._n_vars)), np.array([])

        G_data = np.vstack([r[0] for r in rows])
        h = np.array([r[1] for r in rows])
        self._constraint_info = [(r[2], r[1]) for r in rows]

        return csc_matrix(G_data), h

    def _build_lower_bounds(self) -> np.ndarray:
        """Build lower bounds (all zeros).

        Returns:
            Array of shape (n_vars,) with all zeros
        """
        return np.zeros(self._n_vars)

    def _build_upper_bounds(self) -> np.ndarray:
        """Build upper bounds with food-meal affinity enforcement.

        Foods not allowed in a meal get ub=0 for that meal slot.

        Returns:
            Array of shape (n_vars,) with upper bounds
        """
        default_ub = self.request.max_grams_per_food_per_meal
        ub = np.full(self._n_vars, default_ub)

        # Build affinity lookup: fdc_id -> set of allowed MealTypes
        affinity_map: dict[int, set[MealType]] = {}
        for aff in self.request.food_meal_affinities:
            affinity_map[aff.fdc_id] = set(aff.allowed_meals)

        # Apply affinity constraints
        for i, fdc_id in enumerate(self._food_ids):
            if fdc_id in affinity_map:
                allowed = affinity_map[fdc_id]
                for m, meal_config in enumerate(self.request.meals):
                    if meal_config.meal_type not in allowed:
                        var_idx = self.var_index(i, m)
                        ub[var_idx] = 0.0

        return ub

    def get_nutrient_index(self, nutrient_id: int) -> Optional[int]:
        """Get the index of a nutrient in the nutrient matrix.

        Args:
            nutrient_id: USDA nutrient ID

        Returns:
            Index into nutrient dimension, or None if not tracked
        """
        try:
            return self._nutrient_ids.index(nutrient_id)
        except ValueError:
            return None
