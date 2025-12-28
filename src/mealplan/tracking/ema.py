"""Exponentially smoothed moving average for weight tracking.

This implements the Hacker's Diet trend calculation:
    T_n = T_{n-1} + smoothing × (W_n - T_{n-1})

With smoothing=0.1 (10%), this is equivalent to a low-pass filter with
approximately 10-day time constant. It effectively removes daily noise
from water retention, gut contents, and scale measurement error while
tracking the underlying weight trend.

Reference: https://www.fourmilab.ch/hackdiet/
"""

from __future__ import annotations

# Default smoothing factor (10% = 0.1)
# This is the classic Hacker's Diet value, chosen because:
# 1. Easy mental math (shift decimal point)
# 2. ~10 day time constant balances responsiveness vs noise rejection
DEFAULT_SMOOTHING = 0.1


def update_trend(
    prev_trend: float, today_weight: float, smoothing: float = DEFAULT_SMOOTHING
) -> float:
    """
    Calculate new trend value using exponentially smoothed moving average.

    The formula is:
        T_n = T_{n-1} + smoothing × (W_n - T_{n-1})

    Equivalently:
        T_n = smoothing × W_n + (1 - smoothing) × T_{n-1}

    Args:
        prev_trend: Previous day's trend value (T_{n-1})
        today_weight: Today's scale weight (W_n)
        smoothing: Smoothing factor, default 0.1 (10%)
                   Higher values = more responsive, more noise
                   Lower values = smoother, more lag

    Returns:
        Today's trend value (T_n)

    Example:
        >>> update_trend(173.2, 171.5)
        173.03
        >>> # Pencil method: 171.5 - 173.2 = -1.7, shift decimal = -0.17
        >>> # Round to -0.2, add to 173.2 = 173.0
    """
    return prev_trend + smoothing * (today_weight - prev_trend)


def calculate_trend_from_scratch(
    weights: list[float], smoothing: float = DEFAULT_SMOOTHING
) -> list[float]:
    """
    Calculate trend values for a series of weights.

    The first weight is used as the initial trend value.

    Args:
        weights: List of weight measurements in chronological order
        smoothing: Smoothing factor, default 0.1

    Returns:
        List of trend values, same length as weights

    Example:
        >>> weights = [172.5, 171.5, 172.0, 171.5]
        >>> calculate_trend_from_scratch(weights)
        [172.5, 172.4, 172.36, 172.274]
    """
    if not weights:
        return []

    trends = [weights[0]]  # First trend = first weight
    for weight in weights[1:]:
        trends.append(update_trend(trends[-1], weight, smoothing))

    return trends


def estimate_weekly_change(trend_start: float, trend_end: float, days: int = 7) -> float:
    """
    Estimate weekly weight change from trend values.

    Args:
        trend_start: Trend value at start of period
        trend_end: Trend value at end of period
        days: Number of days in period (default 7)

    Returns:
        Estimated weekly change in lbs (negative = losing)
    """
    daily_change = (trend_end - trend_start) / days
    return daily_change * 7


def estimate_daily_calorie_balance(weekly_change_lbs: float) -> float:
    """
    Estimate daily calorie surplus/deficit from weekly weight change.

    Uses the standard approximation: 3500 calories = 1 lb of body weight.

    Args:
        weekly_change_lbs: Weekly weight change in lbs (negative = loss)

    Returns:
        Daily calorie balance (negative = deficit, positive = surplus)
    """
    # 3500 cal per lb, 7 days per week
    return (weekly_change_lbs * 3500) / 7
