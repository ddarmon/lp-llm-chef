"""Tests for EMA weight tracking with missing day handling."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from llmn.tracking.ema import (
    DEFAULT_SMOOTHING,
    calculate_trend_from_scratch,
    time_scaled_alpha,
    update_trend,
)


class TestTimeScaledAlpha:
    """Tests for time_scaled_alpha function."""

    def test_daily_unchanged(self) -> None:
        """Alpha should be unchanged for daily measurements."""
        assert time_scaled_alpha(0.1, 1) == pytest.approx(0.1)

    def test_three_day_gap(self) -> None:
        """After 3 days, alpha should be 1 - 0.9^3 ≈ 0.271."""
        expected = 1 - (1 - 0.1) ** 3
        assert time_scaled_alpha(0.1, 3) == pytest.approx(expected)

    def test_weekly_gap(self) -> None:
        """After 7 days, alpha should be 1 - 0.9^7 ≈ 0.522."""
        expected = 1 - (1 - 0.1) ** 7
        assert time_scaled_alpha(0.1, 7) == pytest.approx(expected)

    def test_zero_days_treated_as_one(self) -> None:
        """Zero or negative days should be treated as 1."""
        assert time_scaled_alpha(0.1, 0) == pytest.approx(0.1)
        assert time_scaled_alpha(0.1, -1) == pytest.approx(0.1)

    def test_large_gap_approaches_one(self) -> None:
        """Very large gaps should result in alpha near 1."""
        # After 30 days: 1 - 0.9^30 ≈ 0.958
        assert time_scaled_alpha(0.1, 30) > 0.95

    def test_different_base_alpha(self) -> None:
        """Should work with different base alpha values."""
        # With alpha=0.2, after 3 days: 1 - 0.8^3 = 0.488
        expected = 1 - (1 - 0.2) ** 3
        assert time_scaled_alpha(0.2, 3) == pytest.approx(expected)


class TestUpdateTrend:
    """Tests for update_trend function."""

    def test_daily_update_backward_compatible(self) -> None:
        """Daily update should match original behavior."""
        # From original docstring example
        result = update_trend(173.2, 171.5)
        expected = 173.2 + 0.1 * (171.5 - 173.2)
        assert result == pytest.approx(expected, abs=0.01)

    def test_multi_day_gap_gives_more_weight(self) -> None:
        """Longer gaps should give more weight to new measurement."""
        daily = update_trend(173.2, 171.5, days_elapsed=1)
        three_day = update_trend(173.2, 171.5, days_elapsed=3)

        # With 3-day gap, trend should move more toward new measurement
        assert three_day < daily  # Closer to 171.5

    def test_weekly_gap(self) -> None:
        """Weekly gap should give significant weight to new measurement."""
        weekly = update_trend(173.2, 171.5, days_elapsed=7)
        # alpha ≈ 0.522, so trend ≈ 173.2 + 0.522 * (171.5 - 173.2) ≈ 172.31
        expected_alpha = time_scaled_alpha(0.1, 7)
        expected = 173.2 + expected_alpha * (171.5 - 173.2)
        assert weekly == pytest.approx(expected)


class TestCalculateTrendFromScratch:
    """Tests for calculate_trend_from_scratch function."""

    def test_simple_list_backward_compatible(self) -> None:
        """Simple weight list should work as before (assumes daily)."""
        weights = [172.5, 171.5, 172.0, 171.5]
        trends = calculate_trend_from_scratch(weights)

        assert len(trends) == 4
        assert trends[0] == 172.5  # First trend = first weight
        # Each subsequent trend uses standard alpha=0.1
        assert trends[1] == pytest.approx(172.5 + 0.1 * (171.5 - 172.5))

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        assert calculate_trend_from_scratch([]) == []

    def test_single_weight(self) -> None:
        """Single weight should return single trend."""
        trends = calculate_trend_from_scratch([185.0])
        assert trends == [185.0]

    def test_dated_weights_with_gaps(self) -> None:
        """Date/weight tuples should detect and handle gaps."""
        # Day 1: 172.5, Day 4: 171.5 (3-day gap)
        dated = [
            (date(2025, 1, 1), 172.5),
            (date(2025, 1, 4), 171.5),  # 3-day gap
        ]
        trends = calculate_trend_from_scratch(dated)

        assert len(trends) == 2
        assert trends[0] == 172.5

        # Second trend should use time-scaled alpha for 3 days
        expected_alpha = time_scaled_alpha(0.1, 3)
        expected = 172.5 + expected_alpha * (171.5 - 172.5)
        assert trends[1] == pytest.approx(expected)

    def test_dated_weights_daily(self) -> None:
        """Consecutive dates should behave like simple list."""
        dated = [
            (date(2025, 1, 1), 172.5),
            (date(2025, 1, 2), 171.5),
            (date(2025, 1, 3), 172.0),
        ]
        simple = [172.5, 171.5, 172.0]

        dated_trends = calculate_trend_from_scratch(dated)
        simple_trends = calculate_trend_from_scratch(simple)

        for d, s in zip(dated_trends, simple_trends):
            assert d == pytest.approx(s)

    def test_mixed_gaps(self) -> None:
        """Should handle varying gap sizes correctly."""
        dated = [
            (date(2025, 1, 1), 180.0),
            (date(2025, 1, 2), 179.5),  # 1-day gap
            (date(2025, 1, 5), 179.0),  # 3-day gap
            (date(2025, 1, 12), 178.0),  # 7-day gap
        ]
        trends = calculate_trend_from_scratch(dated)

        assert len(trends) == 4

        # Verify each trend was calculated with correct gap
        # Day 2: 1-day gap, alpha=0.1
        expected_1 = 180.0 + 0.1 * (179.5 - 180.0)
        assert trends[1] == pytest.approx(expected_1)

        # Day 5: 3-day gap
        alpha_3 = time_scaled_alpha(0.1, 3)
        expected_2 = trends[1] + alpha_3 * (179.0 - trends[1])
        assert trends[2] == pytest.approx(expected_2)

        # Day 12: 7-day gap
        alpha_7 = time_scaled_alpha(0.1, 7)
        expected_3 = trends[2] + alpha_7 * (178.0 - trends[2])
        assert trends[3] == pytest.approx(expected_3)
