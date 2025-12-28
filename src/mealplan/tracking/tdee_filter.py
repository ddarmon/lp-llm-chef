"""Kalman filter for learning personalized TDEE.

This implements a simple scalar Kalman filter that learns the deviation
between your actual TDEE and the Mifflin-St Jeor estimate. The state
is a single scalar: TDEE bias (kcal/day).

The filter is updated weekly based on:
- Observed: implied calorie deficit from weight trend change
- Expected: planned calories minus Mifflin-St Jeor TDEE
- Innovation: difference between observed and expected

If you're losing weight faster than expected given your planned intake,
the filter infers your TDEE is higher than Mifflin-St Jeor predicts
(positive bias). If slower, TDEE is lower (negative bias).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TDEEFilter:
    """
    Simple scalar Kalman filter for TDEE bias estimation.

    State: tdee_bias (kcal/day deviation from Mifflin-St Jeor)
    Process model: random walk (TDEE can drift slowly over time)
    Observation: implied deficit from weekly trend change

    Attributes:
        bias: Current TDEE bias estimate (kcal/day)
        variance: Current uncertainty (kcal²/day²)
        process_noise: Random walk variance per day (default: 25 = 5² kcal/day)
        obs_noise: Observation noise variance (default: 2500 = 50² kcal/day)
    """

    bias: float = 0.0
    variance: float = 10000.0  # Initial uncertainty: 100 kcal/day std
    process_noise: float = 25.0  # 5 kcal/day std per day
    obs_noise: float = 2500.0  # 50 kcal/day std observation noise

    def predict(self, days: int = 7) -> None:
        """
        Predict step: increase uncertainty due to process noise.

        Args:
            days: Number of days since last update (typically 7)
        """
        self.variance += self.process_noise * days

    def update(self, implied_deficit: float, expected_deficit: float) -> float:
        """
        Update step: incorporate new observation.

        Args:
            implied_deficit: Observed calorie deficit from trend change
                            = (trend_start - trend_end) × 3500 / days
            expected_deficit: Expected deficit from meal plan
                            = mifflin_tdee - planned_calories

        Returns:
            Innovation (observed - expected). Positive means losing
            faster than expected → TDEE is higher than thought.
        """
        innovation = implied_deficit - expected_deficit

        # Kalman gain
        kalman_gain = self.variance / (self.variance + self.obs_noise)

        # Update state
        self.bias += kalman_gain * innovation

        # Update variance
        self.variance *= 1 - kalman_gain

        return innovation

    def predict_and_update(
        self,
        implied_deficit: float,
        expected_deficit: float,
        days: int = 7,
    ) -> float:
        """
        Combined predict + update step for weekly observations.

        Args:
            implied_deficit: Observed deficit from trend change
            expected_deficit: Expected deficit from plan
            days: Days since last observation

        Returns:
            Innovation value
        """
        self.predict(days)
        return self.update(implied_deficit, expected_deficit)

    def get_adjusted_tdee(self, mifflin_tdee: float) -> tuple[float, float]:
        """
        Get adjusted TDEE with uncertainty.

        Args:
            mifflin_tdee: Baseline TDEE from Mifflin-St Jeor formula

        Returns:
            Tuple of (adjusted_tdee, uncertainty_95ci_half_width)
        """
        adjusted = mifflin_tdee + self.bias
        uncertainty = 1.96 * math.sqrt(self.variance)
        return adjusted, uncertainty

    def get_state(self) -> dict:
        """Return current filter state as dictionary."""
        return {
            "bias": self.bias,
            "variance": self.variance,
            "std": math.sqrt(self.variance),
        }


def run_filter_on_history(
    trend_history: list[tuple[float, float]],  # (trend_start, trend_end) per week
    calorie_history: list[float],  # average daily calories per week
    mifflin_tdee: float,
    initial_filter: Optional[TDEEFilter] = None,
) -> TDEEFilter:
    """
    Run Kalman filter on historical data.

    Args:
        trend_history: List of (trend_start, trend_end) tuples per week
        calorie_history: List of average daily calorie intakes per week
        mifflin_tdee: Baseline TDEE from Mifflin-St Jeor
        initial_filter: Optional initial filter state

    Returns:
        Updated TDEEFilter with learned bias
    """
    if initial_filter is None:
        filter = TDEEFilter()
    else:
        filter = TDEEFilter(
            bias=initial_filter.bias,
            variance=initial_filter.variance,
            process_noise=initial_filter.process_noise,
            obs_noise=initial_filter.obs_noise,
        )

    for (trend_start, trend_end), avg_calories in zip(trend_history, calorie_history):
        # Implied deficit from observed trend change
        # Positive if losing weight (trend_start > trend_end)
        implied_deficit = (trend_start - trend_end) * 3500 / 7

        # Expected deficit from meal plan
        expected_deficit = mifflin_tdee - avg_calories

        # Update filter
        filter.predict_and_update(implied_deficit, expected_deficit, days=7)

    return filter
