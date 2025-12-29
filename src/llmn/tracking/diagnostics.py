"""Diagnostic output for weight tracking and TDEE analysis."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from llmn.profiles.body_calc import calculate_bmr, calculate_tdee
from llmn.tracking.ema import estimate_daily_calorie_balance, estimate_weekly_change
from llmn.tracking.models import TDEEEstimate, UserProfile, WeightEntry
from llmn.tracking.queries import (
    CalorieQueries,
    TDEEQueries,
    UserQueries,
    WeightQueries,
)
from llmn.tracking.tdee_filter import TDEEFilter


@dataclass
class WeightReport:
    """Summary report of weight tracking progress."""

    current_weight: float
    current_trend: float
    trend_change: float  # vs period start
    weekly_rate: float  # lbs/week (negative = losing)
    implied_daily_deficit: float  # kcal/day
    period_days: int
    start_weight: float
    start_trend: float


@dataclass
class TDEEReport:
    """Summary report of TDEE analysis."""

    mifflin_tdee: float
    tdee_bias: float
    adjusted_tdee: float
    uncertainty_95ci: float
    avg_planned_calories: Optional[float]
    expected_deficit: Optional[float]  # mifflin_tdee - planned
    implied_deficit: Optional[float]  # from trend change


@dataclass
class ProgressReport:
    """Combined progress report."""

    weight: WeightReport
    tdee: Optional[TDEEReport]
    goal_weight: Optional[float]
    remaining_lbs: Optional[float]
    weeks_to_goal: Optional[float]
    notes: list[str]


def calculate_mifflin_tdee(profile: UserProfile) -> float:
    """Calculate TDEE using Mifflin-St Jeor + activity multiplier."""
    return calculate_mifflin_tdee_at_weight(profile, profile.weight_lbs)


def calculate_mifflin_tdee_at_weight(profile: UserProfile, weight_lbs: float) -> float:
    """
    Calculate TDEE at a specific weight using Mifflin-St Jeor + activity multiplier.

    This allows time-varying TDEE calculation as weight changes, keeping the
    learned bias interpretable as personal deviation from formula.

    Args:
        profile: User profile (for height, age, sex, activity level)
        weight_lbs: Weight to use for calculation (in pounds)

    Returns:
        TDEE in kcal/day
    """
    height_inches = profile.height_inches
    age = profile.age
    sex = profile.sex

    # Calculate BMR
    bmr = calculate_bmr(age, sex, height_inches, weight_lbs)

    # Activity multipliers (Harris-Benedict)
    activity_multipliers = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }

    multiplier = activity_multipliers.get(profile.activity_level, 1.55)
    return bmr * multiplier


def generate_weight_report(
    conn: sqlite3.Connection,
    user_id: int,
    days: int = 30,
) -> Optional[WeightReport]:
    """Generate a weight tracking report."""
    history = WeightQueries.get_weight_history(conn, user_id, days=days)

    if len(history) < 2:
        return None

    oldest = history[0]
    latest = history[-1]

    period_days = (latest.measured_at - oldest.measured_at).days
    if period_days == 0:
        period_days = 1  # Avoid division by zero

    trend_change = latest.trend_lbs - oldest.trend_lbs
    weekly_rate = estimate_weekly_change(oldest.trend_lbs, latest.trend_lbs, period_days)
    # estimate_daily_calorie_balance returns negative for deficit, but we want
    # "deficit" to be positive when in deficit (losing weight), so negate it
    implied_deficit = -estimate_daily_calorie_balance(weekly_rate)

    return WeightReport(
        current_weight=latest.weight_lbs,
        current_trend=latest.trend_lbs,
        trend_change=trend_change,
        weekly_rate=weekly_rate,
        implied_daily_deficit=implied_deficit,
        period_days=period_days,
        start_weight=oldest.weight_lbs,
        start_trend=oldest.trend_lbs,
    )


def generate_tdee_report(
    conn: sqlite3.Connection,
    user_id: int,
    profile: UserProfile,
    weight_report: Optional[WeightReport] = None,
) -> TDEEReport:
    """Generate a TDEE analysis report."""
    mifflin_tdee = calculate_mifflin_tdee(profile)

    # Get latest TDEE estimate from filter
    latest_estimate = TDEEQueries.get_latest_estimate(conn, user_id)

    if latest_estimate:
        tdee_bias = latest_estimate.tdee_bias
        variance = latest_estimate.variance
        adjusted_tdee = latest_estimate.adjusted_tdee
    else:
        tdee_bias = 0.0
        variance = 10000.0  # Initial uncertainty
        adjusted_tdee = mifflin_tdee

    uncertainty_95ci = 1.96 * math.sqrt(variance)

    # Get average planned calories over period
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    avg_calories = CalorieQueries.get_average_calories(conn, user_id, start_date, end_date)

    expected_deficit = None
    if avg_calories is not None:
        expected_deficit = mifflin_tdee - avg_calories

    implied_deficit = None
    if weight_report:
        implied_deficit = weight_report.implied_daily_deficit

    return TDEEReport(
        mifflin_tdee=mifflin_tdee,
        tdee_bias=tdee_bias,
        adjusted_tdee=adjusted_tdee,
        uncertainty_95ci=uncertainty_95ci,
        avg_planned_calories=avg_calories,
        expected_deficit=expected_deficit,
        implied_deficit=implied_deficit,
    )


def generate_progress_report(
    conn: sqlite3.Connection,
    user_id: int,
    days: int = 30,
) -> Optional[ProgressReport]:
    """Generate a comprehensive progress report."""
    profile = UserQueries.get_user(conn, user_id)
    if profile is None:
        return None

    weight_report = generate_weight_report(conn, user_id, days)
    if weight_report is None:
        return None

    tdee_report = generate_tdee_report(conn, user_id, profile, weight_report)

    # Goal progress
    goal_weight = profile.target_weight_lbs
    remaining_lbs = None
    weeks_to_goal = None

    if goal_weight is not None and weight_report.weekly_rate != 0:
        remaining_lbs = weight_report.current_trend - goal_weight
        if weight_report.weekly_rate < 0:  # Losing weight
            weeks_to_goal = remaining_lbs / abs(weight_report.weekly_rate)
        elif weight_report.weekly_rate > 0 and remaining_lbs < 0:  # Gaining toward goal
            weeks_to_goal = abs(remaining_lbs) / weight_report.weekly_rate

    # Generate notes
    notes = []

    if tdee_report.expected_deficit and tdee_report.implied_deficit:
        diff = tdee_report.implied_deficit - tdee_report.expected_deficit
        if abs(diff) > 200:
            if diff > 0:
                notes.append(
                    f"Losing faster than expected by ~{abs(diff):.0f} kcal/day. "
                    "Possible causes: actual intake lower than logged, activity higher than estimated, "
                    "or metabolism faster than Mifflin-St Jeor predicts."
                )
            else:
                notes.append(
                    f"Losing slower than expected by ~{abs(diff):.0f} kcal/day. "
                    "Possible causes: actual intake higher than logged, activity lower than estimated, "
                    "or metabolism slower than Mifflin-St Jeor predicts."
                )

    if weight_report.weekly_rate < -2.0:
        notes.append(
            "Warning: Losing more than 2 lbs/week. This pace may not be sustainable."
        )
    elif weight_report.weekly_rate > 0 and goal_weight and weight_report.current_trend > goal_weight:
        notes.append("You are currently gaining weight while above your goal weight.")

    return ProgressReport(
        weight=weight_report,
        tdee=tdee_report,
        goal_weight=goal_weight,
        remaining_lbs=remaining_lbs,
        weeks_to_goal=weeks_to_goal,
        notes=notes,
    )


def format_weight_report(report: WeightReport) -> str:
    """Format weight report as text."""
    direction = "lost" if report.trend_change < 0 else "gained"
    rate_dir = "losing" if report.weekly_rate < 0 else "gaining"

    lines = [
        f"Weight Tracking Report (last {report.period_days} days)",
        "=" * 45,
        f"Current weight: {report.current_weight:.1f} lbs",
        f"Current trend:  {report.current_trend:.1f} lbs (EMA)",
        f"Trend change:   {abs(report.trend_change):.1f} lbs {direction} (from {report.start_trend:.1f})",
        f"Rate:           {abs(report.weekly_rate):.1f} lbs/week ({rate_dir})",
    ]

    return "\n".join(lines)


def format_tdee_report(report: TDEEReport) -> str:
    """Format TDEE report as text."""
    lines = [
        "",
        "Total Daily Energy Expenditure (TDEE) Analysis",
        "=" * 50,
        f"Mifflin-St Jeor baseline: {report.mifflin_tdee:.0f} kcal/day",
        f"Learned adjustment:       {report.tdee_bias:+.0f} kcal/day",
        f"Your estimated TDEE:      {report.adjusted_tdee:.0f} ± {report.uncertainty_95ci:.0f} kcal/day",
    ]

    if report.avg_planned_calories:
        lines.append("")
        lines.append(f"Average planned intake: {report.avg_planned_calories:.0f} kcal/day")

        if report.expected_deficit is not None:
            lines.append(f"  Expected deficit: {report.expected_deficit:.0f} kcal/day")

        if report.implied_deficit is not None:
            lines.append(f"  Implied deficit:  {report.implied_deficit:.0f} kcal/day (from trend)")

    return "\n".join(lines)


def format_progress_report(report: ProgressReport) -> str:
    """Format full progress report as text."""
    parts = [format_weight_report(report.weight)]

    if report.tdee:
        parts.append(format_tdee_report(report.tdee))

    if report.goal_weight is not None:
        parts.append("")
        parts.append(f"Progress toward goal ({report.goal_weight:.0f} lbs)")
        parts.append("-" * 45)

        if report.remaining_lbs is not None:
            parts.append(f"  Remaining: {report.remaining_lbs:.1f} lbs")

        if report.weeks_to_goal is not None:
            parts.append(f"  At current rate: ~{report.weeks_to_goal:.0f} weeks")

    if report.notes:
        parts.append("")
        parts.append("Notes:")
        for note in report.notes:
            # Wrap long notes
            parts.append(f"  - {note}")

    return "\n".join(parts)
