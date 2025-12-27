"""Batch optimization across multiple food pools.

Enables LLM meta-optimization by running multiple optimizations with different
food pools in a single call, then comparing results to find the best option.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from mealplan.optimizer.models import NutrientConstraint, OptimizationRequest
from mealplan.optimizer.solver import solve_diet_problem


@dataclass
class BatchPool:
    """A named food pool for batch optimization."""

    name: str
    food_ids: list[int]


@dataclass
class BatchRequest:
    """Request for batch optimization across multiple food pools."""

    # Base constraints applied to all pools
    calorie_range: tuple[float, float] = (1800, 2200)
    nutrient_constraints: list[NutrientConstraint] = field(default_factory=list)
    max_grams_per_food: float = 500.0
    lambda_deviation: float = 0.001

    # Pools to optimize
    pools: list[BatchPool] = field(default_factory=list)


@dataclass
class BatchResult:
    """Result of a single pool optimization within a batch."""

    pool_name: str
    status: str  # "optimal", "infeasible", "error"
    success: bool
    run_id: Optional[int] = None
    summary: Optional[dict[str, Any]] = None  # calories, protein, cost, food_count
    diagnosis: Optional[str] = None  # If infeasible


def run_batch_optimization(
    conn: sqlite3.Connection,
    batch_request: BatchRequest,
    save_runs: bool = False,
) -> list[BatchResult]:
    """Run optimization on each pool and return results.

    Args:
        conn: Database connection
        batch_request: Batch request with constraints and pools
        save_runs: Whether to save each run to the database

    Returns:
        List of BatchResult objects, one per pool
    """
    results: list[BatchResult] = []

    for pool in batch_request.pools:
        # Build request for this pool
        request = OptimizationRequest(
            mode="feasibility",
            calorie_range=batch_request.calorie_range,
            nutrient_constraints=list(batch_request.nutrient_constraints),
            explicit_food_ids=pool.food_ids,
            max_grams_per_food=batch_request.max_grams_per_food,
            lambda_deviation=batch_request.lambda_deviation,
            use_quadratic_penalty=True,
        )

        # Run optimization
        opt_result = solve_diet_problem(request, conn, verbose=False)

        # Build batch result
        if opt_result.success:
            # Extract summary
            total_calories = sum(
                n.amount for n in opt_result.nutrients if n.nutrient_id == 1008
            )
            total_protein = sum(
                n.amount for n in opt_result.nutrients if n.nutrient_id == 1003
            )

            summary = {
                "total_calories": round(total_calories, 0),
                "total_protein": round(total_protein, 1),
                "total_cost": round(opt_result.total_cost, 2) if opt_result.total_cost else None,
                "food_count": len(opt_result.foods),
                "foods": [
                    {"fdc_id": f.fdc_id, "description": f.description, "grams": round(f.grams, 1)}
                    for f in opt_result.foods
                ],
            }

            batch_result = BatchResult(
                pool_name=pool.name,
                status="optimal",
                success=True,
                summary=summary,
            )
        else:
            batch_result = BatchResult(
                pool_name=pool.name,
                status=opt_result.status,
                success=False,
                diagnosis=opt_result.message,
            )

        results.append(batch_result)

    return results


def compare_batch_results(results: list[BatchResult]) -> dict[str, Any]:
    """Generate comparison summary of batch results.

    Args:
        results: List of batch results

    Returns:
        Comparison data including best pool and pairwise comparisons
    """
    successful = [r for r in results if r.success and r.summary]

    if not successful:
        return {
            "best": None,
            "successful_count": 0,
            "failed_count": len(results),
            "comparison": {},
        }

    # Find best by protein (primary) and cost (secondary)
    def score_pool(r: BatchResult) -> tuple[float, float]:
        summary = r.summary or {}
        protein = summary.get("total_protein", 0)
        cost = summary.get("total_cost", float("inf")) or float("inf")
        return (protein, -cost)  # Higher protein, lower cost is better

    best = max(successful, key=score_pool)

    # Generate pairwise comparisons
    comparison = {}
    for i, r1 in enumerate(successful):
        for r2 in successful[i + 1:]:
            key = f"{r1.pool_name}_vs_{r2.pool_name}"
            s1 = r1.summary or {}
            s2 = r2.summary or {}

            comparison[key] = {
                "calories_diff": (s1.get("total_calories", 0) or 0) - (s2.get("total_calories", 0) or 0),
                "protein_diff": (s1.get("total_protein", 0) or 0) - (s2.get("total_protein", 0) or 0),
                "cost_diff": (s1.get("total_cost") or 0) - (s2.get("total_cost") or 0),
            }

    return {
        "best": best.pool_name,
        "successful_count": len(successful),
        "failed_count": len(results) - len(successful),
        "comparison": comparison,
    }


def parse_batch_request_json(data: dict[str, Any]) -> BatchRequest:
    """Parse a batch request from JSON data.

    Expected format:
    {
        "base_constraints": {
            "calories": [1700, 1900],
            "nutrients": {"protein": {"min": 150}, "fiber": {"min": 30}}
        },
        "pools": [
            {"name": "pool_a", "foods": [175167, 171287, ...]},
            {"name": "pool_b", "foods": [171955, 175139, ...]}
        ]
    }

    Args:
        data: JSON-parsed dict

    Returns:
        BatchRequest object
    """
    from mealplan.data.nutrient_ids import get_nutrient_id

    request = BatchRequest()

    # Parse base constraints
    base = data.get("base_constraints", {})

    if "calories" in base:
        cal = base["calories"]
        if isinstance(cal, list) and len(cal) >= 2:
            request.calorie_range = (float(cal[0]), float(cal[1]))
        elif isinstance(cal, dict):
            request.calorie_range = (
                float(cal.get("min", 1800)),
                float(cal.get("max", 2200)),
            )

    if "nutrients" in base:
        for nutrient_name, bounds in base["nutrients"].items():
            nutrient_id = get_nutrient_id(nutrient_name)
            min_val = bounds.get("min")
            max_val = bounds.get("max")

            if min_val is not None:
                min_val = float(min_val)
            if max_val is not None:
                max_val = float(max_val)

            request.nutrient_constraints.append(
                NutrientConstraint(
                    nutrient_id=nutrient_id,
                    min_value=min_val,
                    max_value=max_val,
                )
            )

    # Parse options
    if "max_grams_per_food" in base:
        request.max_grams_per_food = float(base["max_grams_per_food"])
    if "lambda_deviation" in base:
        request.lambda_deviation = float(base["lambda_deviation"])

    # Parse pools
    for pool_data in data.get("pools", []):
        request.pools.append(
            BatchPool(
                name=pool_data.get("name", f"pool_{len(request.pools)}"),
                food_ids=[int(fid) for fid in pool_data.get("foods", [])],
            )
        )

    return request


def format_batch_response(
    results: list[BatchResult],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Format batch optimization results for JSON output.

    Args:
        results: List of batch results
        comparison: Comparison data from compare_batch_results

    Returns:
        Formatted response dict
    """
    return {
        "results": [
            {
                "pool": r.pool_name,
                "status": r.status,
                "success": r.success,
                **({
                    "total_calories": r.summary.get("total_calories"),
                    "protein": r.summary.get("total_protein"),
                    "cost": r.summary.get("total_cost"),
                    "food_count": r.summary.get("food_count"),
                } if r.summary else {}),
                **({"diagnosis": r.diagnosis} if r.diagnosis else {}),
            }
            for r in results
        ],
        "best": comparison.get("best"),
        "comparison": comparison.get("comparison", {}),
    }
