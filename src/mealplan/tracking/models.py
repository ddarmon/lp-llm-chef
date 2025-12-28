"""Data models for weight tracking and TDEE learning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class UserProfile:
    """User profile for weight tracking and TDEE calculations."""

    user_id: Optional[int]
    age: int
    sex: str  # 'male' or 'female'
    height_inches: float
    weight_lbs: float
    activity_level: str  # 'sedentary', 'lightly_active', 'moderate', 'active', 'very_active'
    goal: Optional[str] = None  # e.g., 'fat_loss:185:165'
    target_weight_lbs: Optional[float] = None
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.sex not in ("male", "female"):
            raise ValueError(f"sex must be 'male' or 'female', got '{self.sex}'")
        valid_levels = (
            "sedentary",
            "lightly_active",
            "moderate",
            "active",
            "very_active",
        )
        if self.activity_level not in valid_levels:
            raise ValueError(
                f"activity_level must be one of {valid_levels}, got '{self.activity_level}'"
            )


@dataclass
class WeightEntry:
    """A single weight log entry with EMA trend."""

    log_id: Optional[int]
    user_id: int
    weight_lbs: float
    trend_lbs: float
    measured_at: date
    notes: Optional[str] = None


@dataclass
class CalorieEntry:
    """A single calorie log entry."""

    log_id: Optional[int]
    user_id: int
    date: date
    planned_calories: float
    notes: Optional[str] = None


@dataclass
class TDEEEstimate:
    """TDEE estimate from Kalman filter."""

    estimate_id: Optional[int]
    user_id: int
    estimated_at: date
    mifflin_tdee: float
    tdee_bias: float
    variance: float
    adjusted_tdee: float

    @property
    def uncertainty_95ci(self) -> float:
        """Return 95% confidence interval half-width."""
        import math

        return 1.96 * math.sqrt(self.variance)
