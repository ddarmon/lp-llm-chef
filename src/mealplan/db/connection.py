"""Database connection management using raw sqlite3."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from mealplan.db.schema import get_schema_sql


class DatabaseConnection:
    """Manages SQLite database connections."""

    def __init__(self, db_path: Path):
        """Initialize database connection manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create parent directories if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections.

        Yields:
            sqlite3.Connection with Row factory enabled

        Example:
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM foods")
                rows = cursor.fetchall()
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create all tables if they don't exist."""
        with self.get_connection() as conn:
            conn.executescript(get_schema_sql())

    def execute_query(
        self, query: str, params: tuple = ()
    ) -> list[sqlite3.Row]:
        """Execute a query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of Row objects
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()

    def execute_many(self, query: str, params_list: list[tuple]) -> int:
        """Execute a query with multiple parameter sets.

        Args:
            query: SQL query string with placeholders
            params_list: List of parameter tuples

        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.executemany(query, params_list)
            return cursor.rowcount

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists
        """
        query = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (table_name,))
            return cursor.fetchone() is not None

    def get_table_count(self, table_name: str) -> int:
        """Get the number of rows in a table.

        Args:
            table_name: Name of the table

        Returns:
            Row count
        """
        # Note: table_name is validated by checking it exists first
        if not self.table_exists(table_name):
            return 0
        with self.get_connection() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = cursor.fetchone()
            return result[0] if result else 0


# Global database instance (lazy loaded)
_db: Optional[DatabaseConnection] = None


def get_db() -> DatabaseConnection:
    """Get the global database instance.

    Lazily initializes the database connection using settings.

    Returns:
        DatabaseConnection instance
    """
    global _db
    if _db is None:
        from mealplan.config import get_settings

        settings = get_settings()
        _db = DatabaseConnection(settings.database.path)
    return _db


def set_db(db: DatabaseConnection) -> None:
    """Set the global database instance.

    Useful for testing with a custom database.

    Args:
        db: DatabaseConnection instance to use
    """
    global _db
    _db = db
