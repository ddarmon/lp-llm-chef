"""Database queries for weight tracking and TDEE learning."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Optional

from llmn.tracking.ema import update_trend
from llmn.tracking.models import (
    CalorieEntry,
    TDEEEstimate,
    UserProfile,
    WeightEntry,
)


class UserQueries:
    """Database queries for user profiles."""

    @staticmethod
    def create_user(conn: sqlite3.Connection, profile: UserProfile) -> int:
        """Create a new user profile and return the user_id."""
        cursor = conn.execute(
            """
            INSERT INTO user_profiles (age, sex, height_inches, weight_lbs,
                                       activity_level, goal, target_weight_lbs)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.age,
                profile.sex,
                profile.height_inches,
                profile.weight_lbs,
                profile.activity_level,
                profile.goal,
                profile.target_weight_lbs,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0

    @staticmethod
    def get_user(conn: sqlite3.Connection, user_id: int) -> Optional[UserProfile]:
        """Get user profile by ID."""
        row = conn.execute(
            """
            SELECT user_id, age, sex, height_inches, weight_lbs,
                   activity_level, goal, target_weight_lbs, created_at
            FROM user_profiles WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

        if row is None:
            return None

        return UserProfile(
            user_id=row[0],
            age=row[1],
            sex=row[2],
            height_inches=row[3],
            weight_lbs=row[4],
            activity_level=row[5],
            goal=row[6],
            target_weight_lbs=row[7],
            created_at=datetime.fromisoformat(row[8]) if row[8] else None,
        )

    @staticmethod
    def get_default_user(conn: sqlite3.Connection) -> Optional[UserProfile]:
        """Get the first (default) user profile."""
        row = conn.execute(
            """
            SELECT user_id, age, sex, height_inches, weight_lbs,
                   activity_level, goal, target_weight_lbs, created_at
            FROM user_profiles ORDER BY user_id LIMIT 1
            """
        ).fetchone()

        if row is None:
            return None

        return UserProfile(
            user_id=row[0],
            age=row[1],
            sex=row[2],
            height_inches=row[3],
            weight_lbs=row[4],
            activity_level=row[5],
            goal=row[6],
            target_weight_lbs=row[7],
            created_at=datetime.fromisoformat(row[8]) if row[8] else None,
        )

    @staticmethod
    def update_user(conn: sqlite3.Connection, profile: UserProfile) -> None:
        """Update an existing user profile."""
        if profile.user_id is None:
            raise ValueError("Cannot update profile without user_id")

        conn.execute(
            """
            UPDATE user_profiles
            SET age = ?, sex = ?, height_inches = ?, weight_lbs = ?,
                activity_level = ?, goal = ?, target_weight_lbs = ?
            WHERE user_id = ?
            """,
            (
                profile.age,
                profile.sex,
                profile.height_inches,
                profile.weight_lbs,
                profile.activity_level,
                profile.goal,
                profile.target_weight_lbs,
                profile.user_id,
            ),
        )
        conn.commit()


class WeightQueries:
    """Database queries for weight log entries."""

    @staticmethod
    def add_weight(
        conn: sqlite3.Connection,
        user_id: int,
        weight_lbs: float,
        measured_at: date,
        notes: Optional[str] = None,
    ) -> WeightEntry:
        """
        Add a weight entry, computing EMA trend automatically.

        If an entry already exists for this date, it will be replaced.
        """
        # Get previous trend value
        prev = conn.execute(
            """
            SELECT trend_lbs FROM weight_log
            WHERE user_id = ? AND measured_at < ?
            ORDER BY measured_at DESC LIMIT 1
            """,
            (user_id, measured_at.isoformat()),
        ).fetchone()

        if prev is None:
            # First entry: trend = weight
            trend_lbs = weight_lbs
        else:
            trend_lbs = update_trend(prev[0], weight_lbs)

        # Insert or replace
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO weight_log (user_id, weight_lbs, trend_lbs, measured_at, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, weight_lbs, trend_lbs, measured_at.isoformat(), notes),
        )
        conn.commit()

        return WeightEntry(
            log_id=cursor.lastrowid,
            user_id=user_id,
            weight_lbs=weight_lbs,
            trend_lbs=trend_lbs,
            measured_at=measured_at,
            notes=notes,
        )

    @staticmethod
    def get_latest_weight(
        conn: sqlite3.Connection, user_id: int
    ) -> Optional[WeightEntry]:
        """Get the most recent weight entry."""
        row = conn.execute(
            """
            SELECT log_id, user_id, weight_lbs, trend_lbs, measured_at, notes
            FROM weight_log
            WHERE user_id = ?
            ORDER BY measured_at DESC LIMIT 1
            """,
            (user_id,),
        ).fetchone()

        if row is None:
            return None

        return WeightEntry(
            log_id=row[0],
            user_id=row[1],
            weight_lbs=row[2],
            trend_lbs=row[3],
            measured_at=date.fromisoformat(row[4]),
            notes=row[5],
        )

    @staticmethod
    def get_weight_history(
        conn: sqlite3.Connection,
        user_id: int,
        days: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[WeightEntry]:
        """
        Get weight history for a user.

        Args:
            user_id: User ID
            days: If set, return last N days of entries
            start_date: If set, return entries on or after this date
            end_date: If set, return entries on or before this date
        """
        query = """
            SELECT log_id, user_id, weight_lbs, trend_lbs, measured_at, notes
            FROM weight_log
            WHERE user_id = ?
        """
        params: list = [user_id]

        if start_date:
            query += " AND measured_at >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND measured_at <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY measured_at DESC"

        if days:
            query += " LIMIT ?"
            params.append(days)

        rows = conn.execute(query, params).fetchall()

        return [
            WeightEntry(
                log_id=row[0],
                user_id=row[1],
                weight_lbs=row[2],
                trend_lbs=row[3],
                measured_at=date.fromisoformat(row[4]),
                notes=row[5],
            )
            for row in reversed(rows)  # Return in chronological order
        ]

    @staticmethod
    def get_trend_at_date(
        conn: sqlite3.Connection, user_id: int, at_date: date
    ) -> Optional[float]:
        """Get the trend value at or before a specific date."""
        row = conn.execute(
            """
            SELECT trend_lbs FROM weight_log
            WHERE user_id = ? AND measured_at <= ?
            ORDER BY measured_at DESC LIMIT 1
            """,
            (user_id, at_date.isoformat()),
        ).fetchone()

        return row[0] if row else None


class CalorieQueries:
    """Database queries for calorie log entries."""

    @staticmethod
    def log_calories(
        conn: sqlite3.Connection,
        user_id: int,
        log_date: date,
        planned_calories: float,
        notes: Optional[str] = None,
    ) -> CalorieEntry:
        """Log planned calories for a day (insert or replace)."""
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO calorie_log (user_id, date, planned_calories, notes)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, log_date.isoformat(), planned_calories, notes),
        )
        conn.commit()

        return CalorieEntry(
            log_id=cursor.lastrowid,
            user_id=user_id,
            date=log_date,
            planned_calories=planned_calories,
            notes=notes,
        )

    @staticmethod
    def get_calorie_history(
        conn: sqlite3.Connection,
        user_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[CalorieEntry]:
        """Get calorie log history."""
        query = """
            SELECT log_id, user_id, date, planned_calories, notes
            FROM calorie_log
            WHERE user_id = ?
        """
        params: list = [user_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY date"

        rows = conn.execute(query, params).fetchall()

        return [
            CalorieEntry(
                log_id=row[0],
                user_id=row[1],
                date=date.fromisoformat(row[2]),
                planned_calories=row[3],
                notes=row[4],
            )
            for row in rows
        ]

    @staticmethod
    def get_average_calories(
        conn: sqlite3.Connection,
        user_id: int,
        start_date: date,
        end_date: date,
    ) -> Optional[float]:
        """Get average daily planned calories over a period."""
        row = conn.execute(
            """
            SELECT AVG(planned_calories) FROM calorie_log
            WHERE user_id = ? AND date >= ? AND date <= ?
            """,
            (user_id, start_date.isoformat(), end_date.isoformat()),
        ).fetchone()

        return row[0] if row and row[0] is not None else None


class TDEEQueries:
    """Database queries for TDEE estimates."""

    @staticmethod
    def save_estimate(conn: sqlite3.Connection, estimate: TDEEEstimate) -> int:
        """Save a TDEE estimate (insert or replace)."""
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO tdee_estimates
            (user_id, estimated_at, mifflin_tdee, tdee_bias, variance, adjusted_tdee)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                estimate.user_id,
                estimate.estimated_at.isoformat(),
                estimate.mifflin_tdee,
                estimate.tdee_bias,
                estimate.variance,
                estimate.adjusted_tdee,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0

    @staticmethod
    def get_latest_estimate(
        conn: sqlite3.Connection, user_id: int
    ) -> Optional[TDEEEstimate]:
        """Get the most recent TDEE estimate."""
        row = conn.execute(
            """
            SELECT estimate_id, user_id, estimated_at, mifflin_tdee,
                   tdee_bias, variance, adjusted_tdee
            FROM tdee_estimates
            WHERE user_id = ?
            ORDER BY estimated_at DESC LIMIT 1
            """,
            (user_id,),
        ).fetchone()

        if row is None:
            return None

        return TDEEEstimate(
            estimate_id=row[0],
            user_id=row[1],
            estimated_at=date.fromisoformat(row[2]),
            mifflin_tdee=row[3],
            tdee_bias=row[4],
            variance=row[5],
            adjusted_tdee=row[6],
        )

    @staticmethod
    def get_estimate_history(
        conn: sqlite3.Connection, user_id: int, limit: int = 10
    ) -> list[TDEEEstimate]:
        """Get TDEE estimate history."""
        rows = conn.execute(
            """
            SELECT estimate_id, user_id, estimated_at, mifflin_tdee,
                   tdee_bias, variance, adjusted_tdee
            FROM tdee_estimates
            WHERE user_id = ?
            ORDER BY estimated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

        return [
            TDEEEstimate(
                estimate_id=row[0],
                user_id=row[1],
                estimated_at=date.fromisoformat(row[2]),
                mifflin_tdee=row[3],
                tdee_bias=row[4],
                variance=row[5],
                adjusted_tdee=row[6],
            )
            for row in reversed(rows)  # Return in chronological order
        ]
