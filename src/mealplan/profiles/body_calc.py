"""Body composition calculator for calorie and macro targets.

Calculates TDEE (Total Daily Energy Expenditure) and macronutrient targets
based on body metrics and goals (fat loss, maintenance, muscle gain).

Uses Mifflin-St Jeor equation for BMR as it's widely validated for
calculating resting metabolic rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Sex(Enum):
    """Biological sex for BMR calculation."""
    MALE = "male"
    FEMALE = "female"


class ActivityLevel(Enum):
    """Activity level multipliers for TDEE calculation."""
    SEDENTARY = "sedentary"          # Little or no exercise
    LIGHT = "light"                  # Light exercise 1-3 days/week
    MODERATE = "moderate"            # Moderate exercise 3-5 days/week
    ACTIVE = "active"                # Hard exercise 6-7 days/week
    VERY_ACTIVE = "very_active"      # Very hard exercise, physical job


class Goal(Enum):
    """Body composition goal."""
    FAT_LOSS = "fat_loss"            # 500-750 cal deficit
    AGGRESSIVE_FAT_LOSS = "aggressive_fat_loss"  # 750-1000 cal deficit
    MAINTENANCE = "maintenance"       # TDEE
    LEAN_GAIN = "lean_gain"          # 200-300 cal surplus
    MUSCLE_GAIN = "muscle_gain"      # 400-500 cal surplus


# Activity level multipliers (Harris-Benedict activity factors)
ACTIVITY_MULTIPLIERS = {
    ActivityLevel.SEDENTARY: 1.2,
    ActivityLevel.LIGHT: 1.375,
    ActivityLevel.MODERATE: 1.55,
    ActivityLevel.ACTIVE: 1.725,
    ActivityLevel.VERY_ACTIVE: 1.9,
}

# Calorie adjustments by goal (deficit or surplus from TDEE)
GOAL_ADJUSTMENTS = {
    Goal.FAT_LOSS: (-750, -500),           # 500-750 cal deficit
    Goal.AGGRESSIVE_FAT_LOSS: (-1000, -750),  # 750-1000 cal deficit
    Goal.MAINTENANCE: (0, 0),               # No adjustment
    Goal.LEAN_GAIN: (200, 300),            # 200-300 cal surplus
    Goal.MUSCLE_GAIN: (400, 500),          # 400-500 cal surplus
}

# Protein recommendations by goal (grams per pound body weight)
PROTEIN_RECOMMENDATIONS = {
    Goal.FAT_LOSS: (0.8, 1.0),              # Higher to preserve muscle
    Goal.AGGRESSIVE_FAT_LOSS: (1.0, 1.2),   # Even higher for aggressive cuts
    Goal.MAINTENANCE: (0.7, 0.9),           # Moderate
    Goal.LEAN_GAIN: (0.8, 1.0),             # Moderate-high
    Goal.MUSCLE_GAIN: (0.9, 1.1),           # High for muscle building
}


@dataclass
class NutritionTargets:
    """Calculated nutrition targets based on body composition goals."""

    calories_min: int
    calories_max: int
    protein_min: int
    protein_max: int
    fiber_min: int
    sodium_max: int

    # Reference values
    bmr: int                    # Basal Metabolic Rate
    tdee: int                   # Total Daily Energy Expenditure
    deficit_or_surplus: int     # Calories above/below TDEE

    # Goal context
    goal: str
    weight_lbs: float
    target_weight_lbs: Optional[float]
    projected_weeks: Optional[int]  # Weeks to reach target weight

    def to_yaml_constraints(self) -> dict:
        """Convert to YAML constraint format."""
        return {
            "calories": {
                "min": self.calories_min,
                "max": self.calories_max,
            },
            "nutrients": {
                "protein": {
                    "min": self.protein_min,
                    "max": self.protein_max,
                },
                "fiber": {
                    "min": self.fiber_min,
                },
                "sodium": {
                    "max": self.sodium_max,
                },
            },
        }

    def summary(self) -> str:
        """Human-readable summary of targets."""
        lines = [
            f"Goal: {self.goal}",
            f"BMR: {self.bmr} kcal/day",
            f"TDEE: {self.tdee} kcal/day",
            f"Target: {self.calories_min}-{self.calories_max} kcal/day "
            f"({self.deficit_or_surplus:+d} from TDEE)",
            f"Protein: {self.protein_min}-{self.protein_max}g/day",
        ]

        if self.target_weight_lbs and self.projected_weeks:
            weight_to_lose = self.weight_lbs - self.target_weight_lbs
            lines.append(
                f"Projected: {weight_to_lose:.0f} lbs in {self.projected_weeks} weeks"
            )

        return "\n".join(lines)


def calculate_bmr(
    age: int,
    sex: Sex,
    height_inches: float,
    weight_lbs: float,
) -> float:
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation.

    Args:
        age: Age in years
        sex: Biological sex
        height_inches: Height in inches
        weight_lbs: Weight in pounds

    Returns:
        BMR in calories per day
    """
    # Convert to metric
    weight_kg = weight_lbs * 0.453592
    height_cm = height_inches * 2.54

    # Mifflin-St Jeor equation
    if sex == Sex.MALE:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

    return bmr


def calculate_tdee(
    bmr: float,
    activity_level: ActivityLevel,
) -> float:
    """Calculate Total Daily Energy Expenditure.

    Args:
        bmr: Basal Metabolic Rate
        activity_level: Activity level

    Returns:
        TDEE in calories per day
    """
    multiplier = ACTIVITY_MULTIPLIERS[activity_level]
    return bmr * multiplier


def calculate_targets(
    age: int,
    sex: str,
    height_inches: float,
    weight_lbs: float,
    goal: str,
    target_weight_lbs: Optional[float] = None,
    activity_level: str = "moderate",
) -> NutritionTargets:
    """Calculate calorie and macro targets based on body composition goals.

    Args:
        age: Age in years
        sex: "male" or "female"
        height_inches: Height in inches
        weight_lbs: Current weight in pounds
        goal: "fat_loss", "aggressive_fat_loss", "maintenance", "lean_gain", "muscle_gain"
        target_weight_lbs: Target weight for projected timeline
        activity_level: "sedentary", "light", "moderate", "active", "very_active"

    Returns:
        NutritionTargets with calorie and macro recommendations
    """
    # Parse enums
    sex_enum = Sex(sex.lower())
    goal_enum = Goal(goal.lower())
    activity_enum = ActivityLevel(activity_level.lower())

    # Calculate BMR and TDEE
    bmr = calculate_bmr(age, sex_enum, height_inches, weight_lbs)
    tdee = calculate_tdee(bmr, activity_enum)

    # Calculate calorie targets
    adjustment_min, adjustment_max = GOAL_ADJUSTMENTS[goal_enum]
    calories_min = int(tdee + adjustment_min)
    calories_max = int(tdee + adjustment_max)

    # Ensure minimum calories (never below 1200 for women, 1500 for men)
    min_safe = 1500 if sex_enum == Sex.MALE else 1200
    calories_min = max(calories_min, min_safe)
    calories_max = max(calories_max, min_safe + 200)

    # Calculate protein targets
    protein_min_ratio, protein_max_ratio = PROTEIN_RECOMMENDATIONS[goal_enum]
    protein_min = int(weight_lbs * protein_min_ratio)
    protein_max = int(weight_lbs * protein_max_ratio)

    # Standard fiber recommendation
    fiber_min = 25 if sex_enum == Sex.FEMALE else 30

    # Standard sodium limit
    sodium_max = 2300

    # Calculate projected timeline if target weight specified
    projected_weeks = None
    if target_weight_lbs and target_weight_lbs != weight_lbs:
        weight_diff = weight_lbs - target_weight_lbs
        avg_deficit = -(adjustment_min + adjustment_max) / 2
        if avg_deficit > 0:
            # 3500 calories = 1 lb fat
            weekly_loss = (avg_deficit * 7) / 3500
            if weekly_loss > 0:
                projected_weeks = int(abs(weight_diff) / weekly_loss)

    return NutritionTargets(
        calories_min=calories_min,
        calories_max=calories_max,
        protein_min=protein_min,
        protein_max=protein_max,
        fiber_min=fiber_min,
        sodium_max=sodium_max,
        bmr=int(bmr),
        tdee=int(tdee),
        deficit_or_surplus=int((adjustment_min + adjustment_max) / 2),
        goal=goal,
        weight_lbs=weight_lbs,
        target_weight_lbs=target_weight_lbs,
        projected_weeks=projected_weeks,
    )


def parse_goal_string(goal_string: str) -> dict:
    """Parse a goal string like 'fat_loss:185lbs:165lbs' into components.

    Args:
        goal_string: Format "goal:current_weight:target_weight" or just "goal"

    Returns:
        Dict with goal, weight_lbs, target_weight_lbs (if provided)
    """
    parts = goal_string.split(":")

    result = {
        "goal": parts[0],
        "weight_lbs": None,
        "target_weight_lbs": None,
    }

    if len(parts) >= 2:
        # Parse current weight (remove 'lbs' suffix if present)
        weight_str = parts[1].lower().replace("lbs", "").replace("lb", "").strip()
        result["weight_lbs"] = float(weight_str)

    if len(parts) >= 3:
        # Parse target weight
        target_str = parts[2].lower().replace("lbs", "").replace("lb", "").strip()
        result["target_weight_lbs"] = float(target_str)

    return result


def targets_to_dict(targets: NutritionTargets) -> dict:
    """Convert NutritionTargets to dict for JSON output."""
    return {
        "calories": {
            "min": targets.calories_min,
            "max": targets.calories_max,
        },
        "protein": {
            "min": targets.protein_min,
            "max": targets.protein_max,
        },
        "fiber": {
            "min": targets.fiber_min,
        },
        "sodium": {
            "max": targets.sodium_max,
        },
        "reference": {
            "bmr": targets.bmr,
            "tdee": targets.tdee,
            "deficit_or_surplus": targets.deficit_or_surplus,
        },
        "projection": {
            "goal": targets.goal,
            "current_weight_lbs": targets.weight_lbs,
            "target_weight_lbs": targets.target_weight_lbs,
            "weeks_to_goal": targets.projected_weeks,
        },
    }
