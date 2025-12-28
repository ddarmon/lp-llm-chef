"""Weight tracking and TDEE learning module.

This module implements Hacker's Diet-style exponentially smoothed moving
average (EMA) for weight trend tracking, plus a simple Kalman filter for
learning personalized TDEE from observed weight changes.

Key components:
- EMA trend calculation (10% smoothing, ~10 day time constant)
- TDEE Kalman filter (learns TDEE bias from weekly trend changes)
- User profile and logging queries
"""

from __future__ import annotations

from mealplan.tracking.ema import update_trend
from mealplan.tracking.models import (
    CalorieEntry,
    TDEEEstimate,
    UserProfile,
    WeightEntry,
)
from mealplan.tracking.tdee_filter import TDEEFilter

__all__ = [
    "CalorieEntry",
    "TDEEEstimate",
    "TDEEFilter",
    "UserProfile",
    "WeightEntry",
    "update_trend",
]
