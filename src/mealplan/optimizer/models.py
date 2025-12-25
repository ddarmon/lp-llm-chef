"""Data models for optimization requests and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ObjectiveType(Enum):
    """Optimization objective types."""

    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_CALORIES = "minimize_calories"
    MAXIMIZE_PROTEIN = "maximize_protein"


@dataclass
class NutrientConstraint:
    """A constraint on a single nutrient.

    At least one of min_value or max_value must be set.
    """

    nutrient_id: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_value is None and self.max_value is None:
            raise ValueError("At least one of min_value or max_value must be set")


@dataclass
class FoodConstraint:
    """A constraint on a specific food.

    Use max_grams=0 to exclude a food entirely.
    """

    fdc_id: int
    min_grams: Optional[float] = None
    max_grams: Optional[float] = None


@dataclass
class OptimizationRequest:
    """Input specification for the optimizer."""

    mode: str = "feasibility"  # "feasibility" | "minimize_cost"
    objective: ObjectiveType = ObjectiveType.MINIMIZE_COST
    calorie_range: tuple[float, float] = (1800, 2200)
    nutrient_constraints: list[NutrientConstraint] = field(default_factory=list)
    food_constraints: list[FoodConstraint] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    include_tags: list[str] = field(default_factory=list)
    max_grams_per_food: float = 500.0
    max_foods: int = 300  # Max foods to consider (randomly sampled if exceeded)
    planning_days: int = 1
    use_quadratic_penalty: bool = True
    lambda_cost: float = 1.0
    lambda_deviation: float = 0.001


@dataclass
class FoodResult:
    """A single food in the optimization result."""

    fdc_id: int
    description: str
    grams: float
    cost: float
    nutrients: dict[int, float]  # nutrient_id -> amount for this food


@dataclass
class NutrientResult:
    """Summary of a nutrient in the result."""

    nutrient_id: int
    name: str
    unit: str
    amount: float
    min_constraint: Optional[float]
    max_constraint: Optional[float]
    satisfied: bool


@dataclass
class ConstraintKKT:
    """KKT analysis for a single constraint."""

    name: str  # Human-readable name, e.g., "Protein (min)"
    constraint_type: str  # "nutrient_min", "nutrient_max", "food_lower", "food_upper"
    bound: float  # The constraint bound value
    value: float  # Actual value at solution
    slack: float  # Distance from bound (0 = binding)
    multiplier: Optional[float]  # Lagrange multiplier (shadow price)
    is_binding: bool  # True if |slack| < tolerance


@dataclass
class KKTAnalysis:
    """Complete KKT conditions analysis for verifying optimality."""

    solver_type: str  # "lp_highs", "qp_slsqp", "qp_feasibility"
    primal_feasible: bool  # All constraints satisfied
    dual_feasible: bool  # All multipliers >= 0 for inequality constraints
    complementary_slackness_satisfied: bool  # slack * multiplier ≈ 0
    stationarity_residual: float  # ||∇L(x*)|| at optimum
    nutrient_constraints: list[ConstraintKKT]  # All nutrient constraint analyses
    binding_food_bounds: list[ConstraintKKT]  # Foods at their limits


@dataclass
class OptimizationResult:
    """Complete output from the optimizer."""

    success: bool
    status: str  # 'optimal', 'infeasible', 'unbounded', 'error'
    message: str
    foods: list[FoodResult]
    total_cost: Optional[float]
    nutrients: list[NutrientResult]
    solver_info: dict  # iterations, time, etc.
    kkt_analysis: Optional[KKTAnalysis] = None  # Populated when --verbose is used


# Custom exceptions


class MealPlanError(Exception):
    """Base exception for mealplan errors."""

    pass


class InfeasibleDietError(MealPlanError):
    """Raised when no feasible diet exists for given constraints."""

    def __init__(
        self, message: str, constraint_analysis: Optional[dict] = None
    ):
        super().__init__(message)
        self.constraint_analysis = constraint_analysis or {}


class MissingDataError(MealPlanError):
    """Raised when required data is missing."""

    pass


class InvalidProfileError(MealPlanError):
    """Raised when constraint profile is invalid."""

    pass
