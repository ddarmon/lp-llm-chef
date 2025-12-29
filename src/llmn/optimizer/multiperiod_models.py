"""Data models for multi-period (per-meal) optimization.

This module extends the single-period optimization with meal-level structure,
allowing constraints like "snack calories <= 200" to be enforced at optimization
time rather than through post-hoc allocation.

Decision variables: x_{i,m} = grams of food i in meal m
Variable indexing: var_index(food_i, meal_m) = i * n_meals + m
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from llmn.optimizer.models import NutrientConstraint


class MealType(Enum):
    """Standard meal types for multi-period optimization."""

    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


# Default calorie split percentages for each meal type
DEFAULT_CALORIE_SPLITS: dict[MealType, float] = {
    MealType.BREAKFAST: 0.25,
    MealType.LUNCH: 0.35,
    MealType.DINNER: 0.35,
    MealType.SNACK: 0.05,
}

# Default typical portion sizes (grams) for deviation penalty
DEFAULT_TYPICAL_PORTIONS: dict[MealType, float] = {
    MealType.BREAKFAST: 100.0,
    MealType.LUNCH: 100.0,
    MealType.DINNER: 100.0,
    MealType.SNACK: 30.0,  # Smaller portions for snacks
}


@dataclass
class MealNutrientConstraint:
    """Nutrient constraint for a specific meal.

    Example: breakfast protein min 25g, max 50g
    """

    meal: MealType
    nutrient_id: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_value is None and self.max_value is None:
            raise ValueError("At least one of min_value or max_value must be set")


@dataclass
class MealCalorieTarget:
    """Calorie range for a specific meal.

    Example: snack calories 0-200
    """

    meal: MealType
    min_calories: float = 0.0
    max_calories: float = float("inf")


@dataclass
class EquiCalorieConstraint:
    """Constraint that two meals should have similar calories.

    |cal_meal_a - cal_meal_b| <= tolerance

    Linearized as two constraints:
      cal_a - cal_b <= tolerance
      cal_b - cal_a <= tolerance
    """

    meal_a: MealType
    meal_b: MealType
    tolerance: float = 100.0  # kcal


@dataclass
class FoodMealAffinity:
    """Specifies which meals a food is allowed in.

    Foods not in allowed_meals get upper bound = 0 for those meals.
    This is used to enforce rules like "almonds only in snacks".

    Note: Affinities use explicit FDC IDs, not keyword matching.
    """

    fdc_id: int
    allowed_meals: list[MealType] = field(default_factory=list)


@dataclass
class MealConfig:
    """Configuration for a single meal slot.

    Defines the calorie target, nutrient constraints, and typical portion
    size for a meal. The typical_portion affects the quadratic deviation
    penalty - smaller values encourage smaller portions (useful for snacks).
    """

    meal_type: MealType
    calorie_target: Optional[MealCalorieTarget] = None
    nutrient_constraints: list[MealNutrientConstraint] = field(default_factory=list)
    typical_portion: float = 100.0  # Target grams for deviation penalty

    def __post_init__(self) -> None:
        # Set default typical portion based on meal type
        if self.typical_portion == 100.0 and self.meal_type == MealType.SNACK:
            self.typical_portion = DEFAULT_TYPICAL_PORTIONS[MealType.SNACK]


@dataclass
class MultiPeriodRequest:
    """Complete specification for multi-period optimization.

    Extends OptimizationRequest with meal-level structure. Daily constraints
    are linking constraints that ensure total intake across all meals satisfies
    daily requirements.

    Example YAML profile structure:
        calories: {min: 1800, max: 2000}
        nutrients: {protein: {min: 150}}
        meals:
          breakfast: {calories: {min: 400, max: 550}}
          snack: {calories: {min: 0, max: 200}}
    """

    # Daily linking constraints (totals across all meals)
    daily_calorie_range: tuple[float, float] = (1800, 2200)
    daily_nutrient_constraints: list[NutrientConstraint] = field(default_factory=list)

    # Per-meal structure
    meals: list[MealConfig] = field(default_factory=list)

    # Optional constraints
    equicalorie_constraints: list[EquiCalorieConstraint] = field(default_factory=list)
    food_meal_affinities: list[FoodMealAffinity] = field(default_factory=list)

    # Standard options (from OptimizationRequest)
    mode: str = "feasibility"  # "feasibility" | "minimize_cost"
    exclude_tags: list[str] = field(default_factory=list)
    include_tags: list[str] = field(default_factory=list)
    max_grams_per_food_per_meal: float = 300.0
    max_grams_per_food_daily: float = 500.0
    max_foods: int = 300
    lambda_cost: float = 1.0
    lambda_deviation: float = 0.001

    # Explicit food pool (bypasses tag filtering when set)
    explicit_food_ids: Optional[list[int]] = None


@dataclass
class MealFoodResult:
    """A single food allocated to a meal in the result."""

    fdc_id: int
    description: str
    grams: float
    cost: float
    nutrients: dict[int, float]  # nutrient_id -> amount for this food


@dataclass
class MealResult:
    """Result for a single meal."""

    meal_type: MealType
    foods: list[MealFoodResult]
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    total_cost: float


@dataclass
class InfeasibilityDiagnosis:
    """Diagnosis of why multi-period optimization failed.

    Uses IIS-like (Irreducible Infeasible Subset) analysis to identify
    the minimal set of constraints that conflict.
    """

    infeasible: bool
    conflicting_constraints: list[str]  # Names of constraints in conflict
    suggested_relaxations: list[str]  # Actionable suggestions for user
    analysis_method: str  # "constraint_deletion" or "iis"


@dataclass
class MultiPeriodResult:
    """Complete result from multi-period optimization."""

    success: bool
    status: str  # 'optimal', 'infeasible', 'unbounded', 'error'
    message: str
    meals: list[MealResult]
    daily_totals: dict[str, float]  # calories, protein, carbs, fat, cost
    solver_info: dict  # elapsed_seconds, n_vars, n_meals, etc.
    infeasibility_diagnosis: Optional[InfeasibilityDiagnosis] = None


def derive_default_meal_configs(
    daily_cal_range: tuple[float, float],
    daily_nutrients: list[NutrientConstraint],
) -> list[MealConfig]:
    """Auto-derive per-meal targets from daily constraints.

    Uses 25/35/35/5% splits for breakfast/lunch/dinner/snack.
    Nutrient constraints are also proportionally distributed.

    Args:
        daily_cal_range: Daily calorie min/max
        daily_nutrients: Daily nutrient constraints

    Returns:
        List of MealConfig objects for all four meals
    """
    configs = []

    for meal_type in [MealType.BREAKFAST, MealType.LUNCH, MealType.DINNER, MealType.SNACK]:
        split = DEFAULT_CALORIE_SPLITS[meal_type]

        # Scale calorie range by split percentage
        cal_min = daily_cal_range[0] * split
        cal_max = daily_cal_range[1] * split

        calorie_target = MealCalorieTarget(
            meal=meal_type,
            min_calories=cal_min,
            max_calories=cal_max,
        )

        # Scale nutrient constraints by split percentage
        meal_nutrients = []
        for nc in daily_nutrients:
            meal_nc = MealNutrientConstraint(
                meal=meal_type,
                nutrient_id=nc.nutrient_id,
                min_value=nc.min_value * split if nc.min_value is not None else None,
                max_value=nc.max_value * split if nc.max_value is not None else None,
            )
            meal_nutrients.append(meal_nc)

        configs.append(
            MealConfig(
                meal_type=meal_type,
                calorie_target=calorie_target,
                nutrient_constraints=meal_nutrients,
                typical_portion=DEFAULT_TYPICAL_PORTIONS[meal_type],
            )
        )

    return configs
