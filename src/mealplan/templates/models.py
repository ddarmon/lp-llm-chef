"""Data models for template-based meal composition.

These models define the structure of meal templates and the results of
template-based optimization. The key insight is that meal planning should
be a two-phase process:

1. Selection: Choose one food per slot (discrete)
2. Optimization: Find optimal quantities (continuous)

This produces meals that look like what humans would actually eat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from mealplan.optimizer.multiperiod_models import MealType


class DiversityRule(Enum):
    """Rules for food diversity across meals."""

    NO_REPEAT = "no_repeat"  # Never repeat a food across any meal
    NO_REPEAT_ADJACENT = "no_repeat_adjacent"  # Don't repeat in consecutive meals
    ALLOW_REPEAT = "allow_repeat"  # Foods can appear in multiple meals


class SelectionStrategy(Enum):
    """Strategies for selecting foods within slots."""

    RANDOM = "random"  # Random selection (default)
    HIGHEST_PROTEIN = "highest_protein"  # Pick highest protein density
    HIGHEST_FIBER = "highest_fiber"  # Pick highest fiber density
    LOWEST_CALORIE = "lowest_calorie"  # Pick lowest calorie density


@dataclass
class SlotDefinition:
    """A slot in a meal template representing a food category.

    Slots define what type of food should fill a position in a meal.
    For example, a "protein" slot might be filled by eggs at breakfast
    or salmon at lunch.

    Attributes:
        name: Slot name (e.g., "protein", "legume", "vegetable")
        sources: List of source names referencing staple lists (e.g., ["eggs", "fish"])
        target_grams: Ideal portion size for this slot
        min_grams: Minimum allowed grams (default 0)
        max_grams: Maximum allowed grams (default 500)
        count: Number of foods to select for this slot (vegetables might be 2)
        required: If False, slot can be empty if constraints allow
    """

    name: str
    sources: list[str]
    target_grams: float
    min_grams: float = 0.0
    max_grams: float = 500.0
    count: int = 1
    required: bool = True


@dataclass
class MealTemplate:
    """Template defining the structure of a single meal.

    Attributes:
        meal_type: Which meal this template is for
        slots: List of slot definitions for this meal
        calorie_range: Optional (min, max) calorie target for this meal
        protein_min: Optional minimum protein for this meal
    """

    meal_type: MealType
    slots: list[SlotDefinition]
    calorie_range: Optional[tuple[float, float]] = None
    protein_min: Optional[float] = None


@dataclass
class DietTemplate:
    """Full diet template for a dietary pattern.

    A diet template defines meal structure for an entire day,
    including which food sources are appropriate for each slot
    and how diversity should be enforced.

    Attributes:
        name: Template identifier (e.g., "pescatarian_slowcarb")
        description: Human-readable description
        meals: List of meal templates
        diversity_rule: How to enforce food variety
        daily_calorie_range: Target daily calorie range
        daily_protein_min: Minimum daily protein target
    """

    name: str
    description: str
    meals: list[MealTemplate]
    diversity_rule: DiversityRule = DiversityRule.NO_REPEAT
    daily_calorie_range: Optional[tuple[float, float]] = None
    daily_protein_min: Optional[float] = None


@dataclass
class SelectedFood:
    """A food selected to fill a slot.

    Attributes:
        slot: The slot this food fills
        fdc_id: FDC ID of the selected food
        description: Food description from database
        target_grams: Target portion from slot definition
    """

    slot: SlotDefinition
    fdc_id: int
    description: str
    target_grams: float


@dataclass
class SelectedMeal:
    """Result of food selection for one meal.

    Attributes:
        meal_type: Which meal this is
        selections: List of selected foods for each slot
    """

    meal_type: MealType
    selections: list[SelectedFood]


@dataclass
class TemplateOptimizationRequest:
    """Request for template-based optimization.

    Attributes:
        template: The diet template to use
        selections: Pre-selected foods (or None for auto-select)
        daily_calories: (min, max) daily calorie target
        daily_protein: (min, max) daily protein target
        selection_strategy: How to select foods within slots
        seed: Random seed for reproducibility
        max_retries: Max re-selections on infeasibility
        excluded_foods: FDC IDs to exclude from selection
        preferred_foods: Dict mapping slot names to preferred FDC IDs
    """

    template: DietTemplate
    selections: Optional[list[SelectedMeal]] = None
    daily_calories: tuple[float, float] = (1800, 2200)
    daily_protein: tuple[float, float] = (150, 200)
    selection_strategy: SelectionStrategy = SelectionStrategy.RANDOM
    seed: Optional[int] = None
    max_retries: int = 5
    excluded_foods: set[int] = field(default_factory=set)
    preferred_foods: dict[str, int] = field(default_factory=dict)


@dataclass
class OptimizedFood:
    """A food with optimized quantity.

    Attributes:
        fdc_id: FDC ID
        description: Food description
        grams: Optimized quantity in grams
        calories: Calories for this quantity
        protein: Protein in grams
        carbs: Carbs in grams
        fat: Fat in grams
        slot_name: Which slot this food fills
    """

    fdc_id: int
    description: str
    grams: float
    calories: float
    protein: float
    carbs: float
    fat: float
    slot_name: str


@dataclass
class OptimizedMeal:
    """A meal with optimized food quantities.

    Attributes:
        meal_type: Which meal
        foods: List of foods with optimized quantities
        total_calories: Total calories for this meal
        total_protein: Total protein for this meal
        total_carbs: Total carbs for this meal
        total_fat: Total fat for this meal
    """

    meal_type: MealType
    foods: list[OptimizedFood]

    @property
    def total_calories(self) -> float:
        return sum(f.calories for f in self.foods)

    @property
    def total_protein(self) -> float:
        return sum(f.protein for f in self.foods)

    @property
    def total_carbs(self) -> float:
        return sum(f.carbs for f in self.foods)

    @property
    def total_fat(self) -> float:
        return sum(f.fat for f in self.foods)


@dataclass
class TemplateResult:
    """Result of template-based optimization.

    Attributes:
        success: Whether optimization succeeded
        meals: List of optimized meals
        template_name: Name of template used
        selection_attempts: How many selection attempts were made
        message: Status message or error description
    """

    success: bool
    meals: list[OptimizedMeal]
    template_name: str
    selection_attempts: int = 1
    message: str = ""

    @property
    def daily_calories(self) -> float:
        return sum(m.total_calories for m in self.meals)

    @property
    def daily_protein(self) -> float:
        return sum(m.total_protein for m in self.meals)

    @property
    def daily_carbs(self) -> float:
        return sum(m.total_carbs for m in self.meals)

    @property
    def daily_fat(self) -> float:
        return sum(m.total_fat for m in self.meals)
