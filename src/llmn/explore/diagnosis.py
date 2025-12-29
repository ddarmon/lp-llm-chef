"""Infeasibility diagnosis for failed optimizations."""

from __future__ import annotations

from sqlite3 import Connection
from typing import Any, Optional

from llmn.data.nutrient_ids import (
    NUTRIENT_DISPLAY_NAMES,
    NUTRIENT_IDS,
    NUTRIENT_UNITS,
    get_nutrient_name,
)
from llmn.optimizer.constraints import ConstraintBuilder
from llmn.optimizer.models import OptimizationRequest


def diagnose_infeasibility(
    conn: Connection,
    request: OptimizationRequest,
) -> dict[str, Any]:
    """Analyze why an optimization might be infeasible.

    Args:
        conn: Database connection
        request: The optimization request that failed

    Returns:
        Diagnosis dict with likely causes and suggestions
    """
    diagnosis: dict[str, Any] = {
        "likely_conflicts": [],
        "food_pool_issues": None,
        "suggestions": [],
    }

    # Check food pool
    builder = ConstraintBuilder(conn, request)
    constraint_data = builder.build()

    total_eligible = constraint_data["total_eligible_foods"]
    used_foods = len(constraint_data["food_ids"])

    if total_eligible == 0:
        diagnosis["food_pool_issues"] = {
            "foods_available": 0,
            "problem": "No eligible foods",
            "causes": [],
        }

        if request.include_tags:
            diagnosis["food_pool_issues"]["causes"].append(
                f"include_tags filter [{', '.join(request.include_tags)}] matched no foods"
            )
            diagnosis["suggestions"].append({
                "action": "remove_tag_filter",
                "detail": f"Remove or modify include_tags: {request.include_tags}",
            })

        if request.mode == "minimize_cost":
            diagnosis["food_pool_issues"]["causes"].append(
                "Cost minimization mode requires foods with prices"
            )
            diagnosis["suggestions"].append({
                "action": "switch_to_feasibility_mode",
                "detail": "Use mode: feasibility instead of minimize_cost",
            })
            diagnosis["suggestions"].append({
                "action": "add_prices",
                "detail": "Add prices to foods using: llmn prices add <fdc_id> <price>",
            })

        return diagnosis

    if total_eligible < 20:
        diagnosis["food_pool_issues"] = {
            "foods_available": total_eligible,
            "problem": "Very small food pool",
            "possibly_too_restrictive": [],
        }
        if request.include_tags:
            diagnosis["food_pool_issues"]["possibly_too_restrictive"].append(
                f"include_tags: {request.include_tags}"
            )
        diagnosis["suggestions"].append({
            "action": "expand_food_pool",
            "detail": f"Only {total_eligible} foods available. Consider tagging more foods or removing filters.",
        })

    # Analyze constraint conflicts
    cal_min, cal_max = request.calorie_range

    # Check protein-calorie conflict
    protein_constraint = None
    for nc in request.nutrient_constraints:
        if nc.nutrient_id == NUTRIENT_IDS.get("protein"):
            protein_constraint = nc
            break

    if protein_constraint and protein_constraint.min_value:
        # Protein has ~4 kcal/g, so minimum calories from protein
        min_protein_cals = protein_constraint.min_value * 4

        if min_protein_cals > cal_max * 0.5:
            diagnosis["likely_conflicts"].append({
                "constraints": [
                    f"protein >= {protein_constraint.min_value}g",
                    f"calories <= {cal_max}",
                ],
                "explanation": (
                    f"{protein_constraint.min_value}g protein requires "
                    f"~{min_protein_cals:.0f} kcal minimum ({min_protein_cals/cal_max*100:.0f}% of calorie max)"
                ),
                "suggestions": [
                    {
                        "action": "reduce_protein",
                        "to": round(cal_max * 0.3 / 4),  # 30% of cals from protein
                        "command": f"protein:min:{round(cal_max * 0.3 / 4)}",
                    },
                    {
                        "action": "increase_calories",
                        "to": round(min_protein_cals * 2.5),
                        "command": f"calories: min={cal_min}, max={round(min_protein_cals * 2.5)}",
                    },
                ],
            })

    # Check for tight nutrient bounds
    for nc in request.nutrient_constraints:
        if nc.min_value and nc.max_value:
            if nc.max_value < nc.min_value:
                name = get_nutrient_name(nc.nutrient_id)
                diagnosis["likely_conflicts"].append({
                    "constraints": [
                        f"{name} >= {nc.min_value}",
                        f"{name} <= {nc.max_value}",
                    ],
                    "explanation": f"Min ({nc.min_value}) is greater than max ({nc.max_value})",
                    "suggestions": [
                        {
                            "action": "fix_bounds",
                            "detail": f"Set max >= min for {name}",
                        }
                    ],
                })

    # Check sodium/potassium balance (common issue)
    sodium_nc = None
    potassium_nc = None
    for nc in request.nutrient_constraints:
        if nc.nutrient_id == NUTRIENT_IDS.get("sodium"):
            sodium_nc = nc
        elif nc.nutrient_id == NUTRIENT_IDS.get("potassium"):
            potassium_nc = nc

    if (
        sodium_nc
        and sodium_nc.max_value
        and sodium_nc.max_value < 1500
        and potassium_nc
        and potassium_nc.min_value
        and potassium_nc.min_value > 3500
    ):
        diagnosis["likely_conflicts"].append({
            "constraints": [
                f"sodium <= {sodium_nc.max_value}mg",
                f"potassium >= {potassium_nc.min_value}mg",
            ],
            "explanation": "Very low sodium with high potassium is hard to achieve",
            "suggestions": [
                {
                    "action": "relax_sodium",
                    "to": 2000,
                    "command": "sodium:max:2000",
                },
                {
                    "action": "relax_potassium",
                    "to": 3000,
                    "command": "potassium:min:3000",
                },
            ],
        })

    # Add general suggestions if no specific conflicts found
    if not diagnosis["likely_conflicts"] and not diagnosis["food_pool_issues"]:
        diagnosis["suggestions"].append({
            "action": "relax_constraints",
            "detail": "Try relaxing one constraint at a time to identify the bottleneck",
        })
        diagnosis["suggestions"].append({
            "action": "check_nutrient_bounds",
            "detail": "Ensure min/max bounds are achievable with your food pool",
        })

    return diagnosis


def generate_relaxation_suggestions(
    request: OptimizationRequest,
    factor: float = 0.2,
) -> list[dict[str, Any]]:
    """Generate suggestions for relaxing constraints.

    Args:
        request: The optimization request
        factor: How much to relax by (0.2 = 20%)

    Returns:
        List of relaxation suggestions
    """
    suggestions = []

    # Relax calorie max
    cal_min, cal_max = request.calorie_range
    suggestions.append({
        "constraint": "calories",
        "current": f"{cal_min}-{cal_max}",
        "suggested": f"{cal_min}-{round(cal_max * (1 + factor))}",
        "command": f"--calories {cal_min} {round(cal_max * (1 + factor))}",
    })

    # Relax min constraints down, max constraints up
    for nc in request.nutrient_constraints:
        name = get_nutrient_name(nc.nutrient_id)
        unit = NUTRIENT_UNITS.get(nc.nutrient_id, "")

        if nc.min_value:
            new_min = round(nc.min_value * (1 - factor), 1)
            suggestions.append({
                "constraint": f"{name} (min)",
                "current": f">= {nc.min_value}{unit}",
                "suggested": f">= {new_min}{unit}",
                "command": f"{name}:min:{new_min}",
            })

        if nc.max_value:
            new_max = round(nc.max_value * (1 + factor), 1)
            suggestions.append({
                "constraint": f"{name} (max)",
                "current": f"<= {nc.max_value}{unit}",
                "suggested": f"<= {new_max}{unit}",
                "command": f"{name}:max:{new_max}",
            })

    return suggestions
