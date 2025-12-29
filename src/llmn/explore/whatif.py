"""What-if analysis for iterative optimization exploration."""

from __future__ import annotations

import json
from copy import deepcopy
from sqlite3 import Connection
from typing import Any, Optional

from llmn.data.nutrient_ids import get_nutrient_id
from llmn.db.queries import OptimizationRunQueries
from llmn.optimizer.constraints import deserialize_profile_json
from llmn.optimizer.models import NutrientConstraint, OptimizationRequest
from llmn.optimizer.solver import solve_diet_problem


def get_run_request(conn: Connection, run_id: int) -> Optional[dict[str, Any]]:
    """Get the constraints used in a previous optimization run.

    Args:
        conn: Database connection
        run_id: Run ID to retrieve

    Returns:
        Constraints dict or None if not found
    """
    if run_id == -1:
        # "latest" marker
        run = OptimizationRunQueries.get_latest_run(conn)
    else:
        run = OptimizationRunQueries.get_run_by_id(conn, run_id)

    if not run:
        return None

    result_data = json.loads(run["result_json"])
    return result_data.get("constraints_used")


def parse_constraint_modifier(
    modifier: str,
) -> tuple[str, str, Optional[float]]:
    """Parse a constraint modifier string like 'protein:min:150'.

    Args:
        modifier: String in format 'nutrient:bound:value' or 'nutrient:remove'

    Returns:
        Tuple of (nutrient_name, action, value)
        action is 'min', 'max', or 'remove'
    """
    parts = modifier.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid modifier format: {modifier}")

    nutrient = parts[0]
    action = parts[1].lower()

    if action == "remove":
        return (nutrient, "remove", None)
    elif action in ("min", "max"):
        if len(parts) < 3:
            raise ValueError(f"Missing value for {action} constraint: {modifier}")
        try:
            value = float(parts[2])
        except ValueError:
            raise ValueError(f"Invalid value in modifier: {modifier}")
        return (nutrient, action, value)
    else:
        raise ValueError(f"Unknown action '{action}' in modifier: {modifier}")


def apply_constraint_modifiers(
    request: OptimizationRequest,
    add_constraints: Optional[list[str]] = None,
    remove_nutrients: Optional[list[str]] = None,
    new_calorie_range: Optional[tuple[float, float]] = None,
) -> OptimizationRequest:
    """Apply modifications to an optimization request.

    Args:
        request: Base optimization request
        add_constraints: List of constraint modifiers to add (e.g., 'protein:min:150')
        remove_nutrients: List of nutrient names to remove constraints for
        new_calorie_range: New calorie range (min, max)

    Returns:
        Modified optimization request (new instance)
    """
    # Deep copy to avoid modifying original
    modified = OptimizationRequest(
        mode=request.mode,
        calorie_range=request.calorie_range,
        nutrient_constraints=list(request.nutrient_constraints),
        food_constraints=list(request.food_constraints),
        exclude_tags=list(request.exclude_tags),
        include_tags=list(request.include_tags),
        max_grams_per_food=request.max_grams_per_food,
        max_foods=request.max_foods,
        planning_days=request.planning_days,
        use_quadratic_penalty=request.use_quadratic_penalty,
        lambda_cost=request.lambda_cost,
        lambda_deviation=request.lambda_deviation,
    )

    # Update calorie range
    if new_calorie_range:
        modified.calorie_range = new_calorie_range

    # Remove nutrients
    if remove_nutrients:
        nutrient_ids_to_remove = set()
        for name in remove_nutrients:
            try:
                nutrient_ids_to_remove.add(get_nutrient_id(name))
            except KeyError:
                pass  # Ignore unknown nutrients

        modified.nutrient_constraints = [
            nc
            for nc in modified.nutrient_constraints
            if nc.nutrient_id not in nutrient_ids_to_remove
        ]

    # Add/modify constraints
    if add_constraints:
        for modifier in add_constraints:
            nutrient, action, value = parse_constraint_modifier(modifier)

            try:
                nutrient_id = get_nutrient_id(nutrient)
            except KeyError:
                raise ValueError(f"Unknown nutrient: {nutrient}")

            if action == "remove":
                # Remove existing constraint for this nutrient
                modified.nutrient_constraints = [
                    nc
                    for nc in modified.nutrient_constraints
                    if nc.nutrient_id != nutrient_id
                ]
            else:
                # Find existing constraint or create new one
                existing = None
                for nc in modified.nutrient_constraints:
                    if nc.nutrient_id == nutrient_id:
                        existing = nc
                        break

                if existing:
                    if action == "min":
                        existing.min_value = value
                    else:
                        existing.max_value = value
                else:
                    new_constraint = NutrientConstraint(
                        nutrient_id=nutrient_id,
                        min_value=value if action == "min" else None,
                        max_value=value if action == "max" else None,
                    )
                    modified.nutrient_constraints.append(new_constraint)

    return modified


def run_whatif_analysis(
    conn: Connection,
    base_run_id: int,
    add_constraints: Optional[list[str]] = None,
    remove_constraints: Optional[list[str]] = None,
    new_calorie_range: Optional[tuple[float, float]] = None,
    remove_foods: Optional[list[int]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run what-if analysis by modifying constraints from a baseline run.

    Args:
        conn: Database connection
        base_run_id: Run ID to use as baseline (-1 for latest)
        add_constraints: Constraints to add (e.g., ['protein:min:180'])
        remove_constraints: Nutrient constraints to remove
        new_calorie_range: New calorie range
        remove_foods: Food IDs to exclude from the solution
        verbose: Include KKT analysis

    Returns:
        Dict with baseline info, modified constraints, and new result
    """
    from llmn.optimizer.models import FoodConstraint

    # Get baseline run
    if base_run_id == -1:
        run = OptimizationRunQueries.get_latest_run(conn)
    else:
        run = OptimizationRunQueries.get_run_by_id(conn, base_run_id)

    if not run:
        return {
            "success": False,
            "error": f"Run not found: {base_run_id}",
        }

    result_data = json.loads(run["result_json"])

    # Reconstruct base request
    # Try to use constraints_used if available, otherwise build minimal
    if "constraints_used" in result_data:
        constraints = result_data["constraints_used"]
        base_request = deserialize_profile_json(constraints)
    else:
        # Fall back to basic defaults
        base_request = OptimizationRequest()

    # Apply modifications
    modified_request = apply_constraint_modifiers(
        base_request,
        add_constraints=add_constraints,
        remove_nutrients=remove_constraints,
        new_calorie_range=new_calorie_range,
    )

    # Add food exclusions
    if remove_foods:
        for fdc_id in remove_foods:
            modified_request.food_constraints.append(
                FoodConstraint(fdc_id=fdc_id, max_grams=0)
            )

    # Run optimization with modified constraints
    result = solve_diet_problem(modified_request, conn, verbose=verbose)

    # Build response
    response: dict[str, Any] = {
        "success": True,
        "base_run_id": run["run_id"],
        "modifications": {
            "add_constraints": add_constraints,
            "remove_constraints": remove_constraints,
            "new_calorie_range": new_calorie_range,
            "remove_foods": remove_foods,
        },
        "result": {
            "success": result.success,
            "status": result.status,
            "message": result.message,
        },
    }

    if result.success:
        response["result"]["foods"] = [
            {
                "fdc_id": f.fdc_id,
                "description": f.description,
                "grams": round(f.grams, 1),
                "cost": round(f.cost, 2),
            }
            for f in result.foods
        ]
        response["result"]["nutrients"] = {
            n.nutrient_id: {
                "name": n.name,
                "amount": round(n.amount, 2),
                "unit": n.unit,
                "min": n.min_constraint,
                "max": n.max_constraint,
                "satisfied": n.satisfied,
            }
            for n in result.nutrients
        }
        response["result"]["total_cost"] = (
            round(result.total_cost, 2) if result.total_cost else None
        )

        # Compare to baseline
        baseline_foods = {
            f["fdc_id"]: f["grams"]
            for f in result_data.get("solution", {}).get("foods", [])
        }
        new_foods = {f.fdc_id: f.grams for f in result.foods}

        added = [fdc_id for fdc_id in new_foods if fdc_id not in baseline_foods]
        removed = [fdc_id for fdc_id in baseline_foods if fdc_id not in new_foods]
        changed = [
            fdc_id
            for fdc_id in new_foods
            if fdc_id in baseline_foods
            and abs(new_foods[fdc_id] - baseline_foods[fdc_id]) > 1
        ]

        response["comparison"] = {
            "foods_added": added,
            "foods_removed": removed,
            "foods_changed": changed,
        }

        # Cost comparison
        baseline_cost = result_data.get("solution", {}).get("total_cost")
        if baseline_cost and result.total_cost:
            response["comparison"]["cost_change"] = round(
                result.total_cost - baseline_cost, 2
            )

    return response


def preview_constraint_impact(
    conn: Connection,
    base_run_id: int,
    constraint: str,
) -> dict[str, Any]:
    """Preview the impact of a constraint change without full optimization.

    This is a lightweight check that returns information about how
    the constraint would affect the food pool and feasibility.

    Args:
        conn: Database connection
        base_run_id: Run ID to use as baseline
        constraint: Constraint modifier (e.g., 'protein:min:200')

    Returns:
        Preview information about the constraint impact
    """
    from llmn.optimizer.constraints import ConstraintBuilder

    # Get baseline
    if base_run_id == -1:
        run = OptimizationRunQueries.get_latest_run(conn)
    else:
        run = OptimizationRunQueries.get_run_by_id(conn, base_run_id)

    if not run:
        return {"success": False, "error": f"Run not found: {base_run_id}"}

    result_data = json.loads(run["result_json"])

    # Reconstruct request
    if "constraints_used" in result_data:
        base_request = deserialize_profile_json(result_data["constraints_used"])
    else:
        base_request = OptimizationRequest()

    # Apply modification
    nutrient, action, value = parse_constraint_modifier(constraint)

    modified_request = apply_constraint_modifiers(
        base_request, add_constraints=[constraint]
    )

    # Build constraints to see food pool impact
    builder = ConstraintBuilder(conn, modified_request)
    constraint_data = builder.build()

    return {
        "success": True,
        "constraint": constraint,
        "parsed": {"nutrient": nutrient, "action": action, "value": value},
        "food_pool": {
            "eligible_foods": constraint_data["total_eligible_foods"],
            "used_foods": len(constraint_data["food_ids"]),
            "was_sampled": constraint_data["was_sampled"],
        },
        "suggestion": "Run full what-if analysis to see solution impact",
    }
