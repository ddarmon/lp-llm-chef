"""Compare optimization runs for iterative diet refinement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Any, Optional


@dataclass
class RunSummary:
    """Summary of a single optimization run."""

    run_id: int
    timestamp: str
    status: str
    success: bool
    profile: Optional[str]
    total_cost: Optional[float]
    food_count: int
    calories: float
    protein: float
    foods: list[dict[str, Any]]


@dataclass
class RunComparison:
    """Comparison between two optimization runs."""

    run_a: RunSummary
    run_b: RunSummary
    cost_difference: Optional[float]
    calorie_difference: float
    protein_difference: float
    foods_only_in_a: list[str]
    foods_only_in_b: list[str]
    foods_in_both: list[str]


def load_run(conn: Connection, run_id: int) -> Optional[RunSummary]:
    """Load a run from the database and extract summary.

    Args:
        conn: Database connection
        run_id: Run ID to load

    Returns:
        RunSummary or None if not found
    """
    query = """
        SELECT run_id, created_at, status, total_cost, result_json
        FROM optimization_runs
        WHERE run_id = ?
    """
    row = conn.execute(query, [run_id]).fetchone()

    if not row:
        return None

    # Parse the result JSON
    result_data = json.loads(row["result_json"]) if row["result_json"] else {}

    # Extract foods
    foods = []
    if "solution" in result_data:
        foods = result_data["solution"].get("foods", [])
    elif "foods" in result_data:
        foods = result_data.get("foods", [])

    # Extract nutrients
    nutrients = {}
    if "solution" in result_data:
        nutrients = result_data["solution"].get("nutrients", {})
    elif "nutrients" in result_data:
        nutrients = result_data.get("nutrients", {})

    # Find calories and protein
    calories = 0.0
    protein = 0.0

    for nid, ndata in nutrients.items():
        if str(nid) == "1008":  # Energy
            calories = ndata.get("amount", 0)
        elif str(nid) == "1003":  # Protein
            protein = ndata.get("amount", 0)

    return RunSummary(
        run_id=row["run_id"],
        timestamp=row["created_at"],
        status=row["status"],
        success=row["status"] == "optimal",
        profile=result_data.get("profile"),
        total_cost=row["total_cost"],
        food_count=len(foods),
        calories=calories,
        protein=protein,
        foods=foods,
    )


def compare_runs(
    conn: Connection,
    run_id_a: int,
    run_id_b: int,
) -> Optional[RunComparison]:
    """Compare two optimization runs.

    Args:
        conn: Database connection
        run_id_a: First run ID
        run_id_b: Second run ID

    Returns:
        RunComparison or None if runs not found
    """
    run_a = load_run(conn, run_id_a)
    run_b = load_run(conn, run_id_b)

    if not run_a or not run_b:
        return None

    # Calculate differences
    cost_diff = None
    if run_a.total_cost is not None and run_b.total_cost is not None:
        cost_diff = run_b.total_cost - run_a.total_cost

    calorie_diff = run_b.calories - run_a.calories
    protein_diff = run_b.protein - run_a.protein

    # Compare food sets
    foods_a = {f.get("description", "") for f in run_a.foods}
    foods_b = {f.get("description", "") for f in run_b.foods}

    only_in_a = sorted(foods_a - foods_b)
    only_in_b = sorted(foods_b - foods_a)
    in_both = sorted(foods_a & foods_b)

    return RunComparison(
        run_a=run_a,
        run_b=run_b,
        cost_difference=cost_diff,
        calorie_difference=calorie_diff,
        protein_difference=protein_diff,
        foods_only_in_a=only_in_a,
        foods_only_in_b=only_in_b,
        foods_in_both=in_both,
    )


def format_run_comparison(comparison: RunComparison) -> dict[str, Any]:
    """Format run comparison for JSON output.

    Args:
        comparison: RunComparison object

    Returns:
        Formatted dictionary for JSON
    """
    return {
        "run_a": {
            "run_id": comparison.run_a.run_id,
            "timestamp": comparison.run_a.timestamp,
            "status": comparison.run_a.status,
            "profile": comparison.run_a.profile,
            "total_cost": comparison.run_a.total_cost,
            "food_count": comparison.run_a.food_count,
            "calories": round(comparison.run_a.calories, 0),
            "protein": round(comparison.run_a.protein, 1),
        },
        "run_b": {
            "run_id": comparison.run_b.run_id,
            "timestamp": comparison.run_b.timestamp,
            "status": comparison.run_b.status,
            "profile": comparison.run_b.profile,
            "total_cost": comparison.run_b.total_cost,
            "food_count": comparison.run_b.food_count,
            "calories": round(comparison.run_b.calories, 0),
            "protein": round(comparison.run_b.protein, 1),
        },
        "differences": {
            "cost": round(comparison.cost_difference, 2) if comparison.cost_difference else None,
            "calories": round(comparison.calorie_difference, 0),
            "protein": round(comparison.protein_difference, 1),
        },
        "foods": {
            "only_in_a": comparison.foods_only_in_a,
            "only_in_b": comparison.foods_only_in_b,
            "in_both": comparison.foods_in_both,
            "overlap_count": len(comparison.foods_in_both),
        },
    }


def list_runs(
    conn: Connection,
    limit: int = 10,
    profile: Optional[str] = None,
) -> list[RunSummary]:
    """List recent optimization runs.

    Args:
        conn: Database connection
        limit: Maximum runs to return
        profile: Optional profile filter

    Returns:
        List of RunSummary objects (most recent first)
    """
    params: list[Any] = []
    profile_filter = ""

    if profile:
        profile_filter = "AND result_json LIKE ?"
        params.append(f'%"profile": "{profile}"%')

    params.append(limit)

    query = f"""
        SELECT run_id, created_at, status, total_cost, result_json
        FROM optimization_runs
        WHERE 1=1
        {profile_filter}
        ORDER BY run_id DESC
        LIMIT ?
    """

    rows = conn.execute(query, params).fetchall()
    summaries = []

    for row in rows:
        summary = load_run(conn, row["run_id"])
        if summary:
            summaries.append(summary)

    return summaries


def format_run_list(runs: list[RunSummary]) -> dict[str, Any]:
    """Format run list for JSON output.

    Args:
        runs: List of RunSummary objects

    Returns:
        Formatted dictionary for JSON
    """
    return {
        "runs": [
            {
                "run_id": r.run_id,
                "timestamp": r.timestamp,
                "status": r.status,
                "profile": r.profile,
                "total_cost": r.total_cost,
                "food_count": r.food_count,
                "calories": round(r.calories, 0),
                "protein": round(r.protein, 1),
            }
            for r in runs
        ],
        "count": len(runs),
    }
