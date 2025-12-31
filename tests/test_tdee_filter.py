"""Tests for TDEE Kalman filter implementation."""

from __future__ import annotations

import random

import pytest

from llmn.tracking.tdee_filter import (
    MAX_VARIANCE,
    MIN_CALORIE_COVERAGE,
    TDEEFilter,
    run_filter_on_history,
)


class TestTDEEFilter:
    """Tests for TDEEFilter class."""

    def test_filter_converges_to_true_bias(self) -> None:
        """Filter should converge to ~100 when true bias is 100."""
        true_bias = 100
        tdee_filter = TDEEFilter()

        # Simulate 8 weeks of observations: z = true_bias + noise
        random.seed(42)
        for _ in range(8):
            noise = random.gauss(0, 50)
            z = true_bias + noise
            # implied_deficit - expected_deficit = z (so implied = expected + z)
            tdee_filter.predict_and_update(
                implied_deficit=500 + z,  # arbitrary base
                expected_deficit=500,
            )

        # Should be within ~50 kcal of true bias after 8 observations
        # (with new obs_noise=22500, convergence is slower but more robust)
        assert abs(tdee_filter.bias - true_bias) < 60

    def test_filter_converges_to_negative_bias(self) -> None:
        """Filter should converge to ~-75 when true bias is -75."""
        true_bias = -75
        tdee_filter = TDEEFilter()

        random.seed(123)
        for _ in range(10):
            noise = random.gauss(0, 40)
            z = true_bias + noise
            tdee_filter.predict_and_update(
                implied_deficit=300 + z,
                expected_deficit=300,
            )

        assert abs(tdee_filter.bias - true_bias) < 50

    def test_filter_stable_when_no_bias(self) -> None:
        """Filter should stay near 0 when true bias is 0."""
        true_bias = 0
        tdee_filter = TDEEFilter()

        random.seed(456)
        for _ in range(8):
            noise = random.gauss(0, 50)
            z = true_bias + noise
            tdee_filter.predict_and_update(
                implied_deficit=400 + z,
                expected_deficit=400,
            )

        # Should stay close to 0
        assert abs(tdee_filter.bias) < 50

    def test_update_returns_residual(self) -> None:
        """Update should return residual (z - x_predicted), not raw z."""
        tdee_filter = TDEEFilter()
        tdee_filter.bias = 50.0  # Set a known bias

        # z = implied - expected = 600 - 500 = 100
        # residual = z - bias = 100 - 50 = 50
        residual = tdee_filter.update(implied_deficit=600, expected_deficit=500)

        assert abs(residual - 50.0) < 0.01

    def test_variance_decreases_with_observations(self) -> None:
        """Variance should decrease as we gather observations."""
        tdee_filter = TDEEFilter()
        initial_variance = tdee_filter.variance

        for _ in range(5):
            tdee_filter.predict_and_update(
                implied_deficit=500,
                expected_deficit=500,
            )

        # Variance should be lower after observations
        assert tdee_filter.variance < initial_variance

    def test_predict_increases_variance(self) -> None:
        """Predict step should increase variance due to process noise."""
        tdee_filter = TDEEFilter()
        initial_variance = tdee_filter.variance

        tdee_filter.predict(days=7)

        expected_increase = tdee_filter.process_noise * 7
        assert abs(tdee_filter.variance - (initial_variance + expected_increase)) < 0.01

    def test_variance_capped_after_long_gap(self) -> None:
        """Variance should be capped at MAX_VARIANCE after very long gaps."""
        tdee_filter = TDEEFilter()

        # Need (40000 - 10000) / 25 = 1200+ days to hit cap
        tdee_filter.predict(days=1500)

        assert tdee_filter.variance == MAX_VARIANCE

    def test_variance_capping_preserves_normal_growth(self) -> None:
        """Normal gaps should not hit the cap."""
        tdee_filter = TDEEFilter()

        # 7 days should not hit the cap
        tdee_filter.predict(days=7)
        expected = 10000.0 + 25.0 * 7  # 10175
        assert tdee_filter.variance == pytest.approx(expected)
        assert tdee_filter.variance < MAX_VARIANCE

    def test_calorie_coverage_increases_obs_noise(self) -> None:
        """Lower calorie coverage should increase effective observation noise."""
        filter_full = TDEEFilter()
        filter_half = TDEEFilter()

        # Same measurement, different coverage
        filter_full.predict(days=7)
        filter_half.predict(days=7)

        filter_full.update(
            implied_deficit=500,
            expected_deficit=500,
            calorie_coverage=1.0,
        )
        filter_half.update(
            implied_deficit=500,
            expected_deficit=500,
            calorie_coverage=0.5,
        )

        # With lower coverage, Kalman gain should be lower (less trust in obs)
        # so variance should decrease less
        assert filter_half.variance > filter_full.variance

    def test_calorie_coverage_minimum(self) -> None:
        """Coverage below MIN_CALORIE_COVERAGE should be clamped."""
        filter_low = TDEEFilter()
        filter_min = TDEEFilter()

        filter_low.predict(days=7)
        filter_min.predict(days=7)

        # Very low coverage should be same as MIN_CALORIE_COVERAGE
        filter_low.update(
            implied_deficit=500,
            expected_deficit=500,
            calorie_coverage=0.1,  # Below minimum
        )
        filter_min.update(
            implied_deficit=500,
            expected_deficit=500,
            calorie_coverage=MIN_CALORIE_COVERAGE,
        )

        # Should be identical (clamped)
        assert filter_low.variance == pytest.approx(filter_min.variance)
        assert filter_low.bias == pytest.approx(filter_min.bias)

    def test_obs_noise_default(self) -> None:
        """Default obs_noise should be 22500 (150 kcal/day std)."""
        tdee_filter = TDEEFilter()
        assert tdee_filter.obs_noise == 22500.0


class TestRunFilterOnHistory:
    """Tests for run_filter_on_history function."""

    def test_time_varying_tdee(self) -> None:
        """run_filter_on_history should use per-week TDEE values."""
        # Simulate 3 weeks of data
        trend_history = [
            (185.0, 184.0),  # Week 1: lost 1 lb
            (184.0, 183.0),  # Week 2: lost 1 lb
            (183.0, 182.0),  # Week 3: lost 1 lb
        ]
        calorie_history = [1800.0, 1800.0, 1800.0]

        # Mifflin TDEE decreases as weight decreases (~10 kcal per lb)
        tdee_history = [2300.0, 2290.0, 2280.0]

        result = run_filter_on_history(trend_history, calorie_history, tdee_history)

        # Filter should have run on all 3 weeks
        # Variance should be lower than initial (some observations processed)
        assert result.variance < 10000.0

    def test_consistent_bias_with_static_tdee(self) -> None:
        """With static TDEE, bias should converge properly."""
        # Simulate perfect data: exactly 500 kcal deficit with true bias of 0
        # implied = (start - end) * 3500 / 7 = 1 * 500 = 500 kcal/day
        # expected = 2300 - 1800 = 500 kcal/day
        # z = implied - expected = 0 (no bias)
        trend_history = [
            (185.0, 184.0),
            (184.0, 183.0),
            (183.0, 182.0),
        ]
        calorie_history = [1800.0, 1800.0, 1800.0]
        tdee_history = [2300.0, 2300.0, 2300.0]

        result = run_filter_on_history(trend_history, calorie_history, tdee_history)

        # Bias should stay near 0
        assert abs(result.bias) < 30
