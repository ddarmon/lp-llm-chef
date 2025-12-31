"""Exponentially smoothed moving average for weight tracking.

This implements the Hacker's Diet trend calculation:
    T_n = T_{n-1} + smoothing × (W_n - T_{n-1})

With smoothing=0.1 (10%), this is equivalent to a low-pass filter with
approximately 10-day time constant. It effectively removes daily noise
from water retention, gut contents, and scale measurement error while
tracking the underlying weight trend.

For non-daily measurements, we use time-scaled smoothing:
    α_adjusted = 1 - (1 - α)^t
where t is days since last measurement. This is theoretically grounded
in treating discrete EWMA as an approximation of continuous exponential
smoothing.

Reference: https://www.fourmilab.ch/hackdiet/
"""

from __future__ import annotations

from datetime import date

# Default smoothing factor (10% = 0.1)
# This is the classic Hacker's Diet value, chosen because:
# 1. Easy mental math (shift decimal point)
# 2. ~10 day time constant balances responsiveness vs noise rejection
DEFAULT_SMOOTHING = 0.1


def time_scaled_alpha(base_alpha: float, days_elapsed: int) -> float:
    """
    Adjust smoothing factor for non-daily measurements.

    When measurements are not daily, we adjust α to account for elapsed time.
    This is derived from treating EWMA as a discrete approximation of continuous
    exponential smoothing.

    Args:
        base_alpha: Base smoothing factor (typically 0.1)
        days_elapsed: Days since last measurement

    Returns:
        Adjusted smoothing factor

    Example:
        >>> time_scaled_alpha(0.1, 1)  # Daily: unchanged
        0.1
        >>> time_scaled_alpha(0.1, 3)  # 3 days: trust new measurement more
        0.271  # = 1 - 0.9^3
        >>> time_scaled_alpha(0.1, 7)  # Weekly
        0.522  # = 1 - 0.9^7
    """
    if days_elapsed <= 0:
        days_elapsed = 1
    return 1 - (1 - base_alpha) ** days_elapsed


def update_trend(
    prev_trend: float,
    today_weight: float,
    smoothing: float = DEFAULT_SMOOTHING,
    days_elapsed: int = 1,
) -> float:
    """
    Calculate new trend value using exponentially smoothed moving average.

    The formula is:
        T_n = T_{n-1} + α × (W_n - T_{n-1})

    For non-daily measurements, α is adjusted:
        α_adjusted = 1 - (1 - α)^t

    Args:
        prev_trend: Previous trend value (T_{n-1})
        today_weight: Today's scale weight (W_n)
        smoothing: Base smoothing factor, default 0.1 (10%)
                   Higher values = more responsive, more noise
                   Lower values = smoother, more lag
        days_elapsed: Days since last measurement (default 1)
                      Used to adjust smoothing for irregular logging

    Returns:
        Today's trend value (T_n)

    Example:
        >>> update_trend(173.2, 171.5)  # Daily measurement
        173.03
        >>> update_trend(173.2, 171.5, days_elapsed=3)  # After 3-day gap
        172.74  # More weight given to new measurement
    """
    adjusted_alpha = time_scaled_alpha(smoothing, days_elapsed)
    return prev_trend + adjusted_alpha * (today_weight - prev_trend)


def calculate_trend_from_scratch(
    weights: list[float] | list[tuple[date, float]],
    smoothing: float = DEFAULT_SMOOTHING,
) -> list[float]:
    """
    Calculate trend values for a series of weights.

    The first weight is used as the initial trend value. If date/weight
    tuples are provided, gaps between measurements are detected and the
    smoothing factor is adjusted accordingly.

    Args:
        weights: Either:
            - List of weight measurements in chronological order (assumes daily)
            - List of (date, weight) tuples for gap-aware calculation
        smoothing: Base smoothing factor, default 0.1

    Returns:
        List of trend values, same length as weights

    Example:
        >>> weights = [172.5, 171.5, 172.0, 171.5]
        >>> calculate_trend_from_scratch(weights)
        [172.5, 172.4, 172.36, 172.274]

        >>> # With dates (gap-aware)
        >>> from datetime import date
        >>> dated = [(date(2025, 1, 1), 172.5), (date(2025, 1, 4), 171.5)]
        >>> calculate_trend_from_scratch(dated)  # 3-day gap detected
        [172.5, 172.229...]  # More weight given due to gap
    """
    if not weights:
        return []

    # Check if we have date/weight tuples or just weights
    first = weights[0]
    if isinstance(first, tuple):
        # Gap-aware mode: extract dates and weights
        dated_weights: list[tuple[date, float]] = weights  # type: ignore
        trends = [dated_weights[0][1]]  # First trend = first weight

        for i in range(1, len(dated_weights)):
            prev_date, _ = dated_weights[i - 1]
            curr_date, curr_weight = dated_weights[i]
            days_elapsed = (curr_date - prev_date).days
            trends.append(update_trend(trends[-1], curr_weight, smoothing, days_elapsed))
    else:
        # Simple mode: assume daily measurements
        weight_list: list[float] = weights  # type: ignore
        trends = [weight_list[0]]  # First trend = first weight
        for weight in weight_list[1:]:
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
