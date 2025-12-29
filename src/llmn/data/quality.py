"""Data quality detection and validation utilities.

Identifies foods with incomplete or missing nutrient data in the USDA database,
and provides fallback calculations where possible.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional


# USDA nutrient IDs for critical macros
ENERGY_ID = 1008
PROTEIN_ID = 1003
FAT_ID = 1004
CARB_ID = 1005


@dataclass
class DataQualityIssue:
    """A data quality issue detected for a food."""

    fdc_id: int
    description: str
    issue_type: str  # "missing_energy", "zero_macros", "calculated_energy"
    severity: str  # "warning", "error"
    message: str
    fallback_value: Optional[float] = None


def calculate_fallback_energy(
    protein_g: float, carbs_g: float, fat_g: float
) -> float:
    """Calculate calories from macronutrients using Atwater factors.

    Uses the standard Atwater factors:
    - Protein: 4 kcal/g
    - Carbohydrate: 4 kcal/g
    - Fat: 9 kcal/g

    Args:
        protein_g: Protein in grams per 100g
        carbs_g: Carbohydrates in grams per 100g
        fat_g: Fat in grams per 100g

    Returns:
        Estimated calories per 100g
    """
    return (protein_g * 4) + (carbs_g * 4) + (fat_g * 9)


def detect_incomplete_foods(
    conn: sqlite3.Connection,
    fdc_ids: list[int],
) -> list[DataQualityIssue]:
    """Identify foods with missing or inconsistent nutrient data.

    Checks for:
    1. Foods with Energy=0 or missing but non-zero macros (missing_energy)
    2. Foods with all zero macros (zero_macros)
    3. Foods where calculated energy differs significantly from stored (inconsistent_energy)

    Args:
        conn: Database connection
        fdc_ids: List of food IDs to check

    Returns:
        List of DataQualityIssue objects for problematic foods
    """
    if not fdc_ids:
        return []

    issues: list[DataQualityIssue] = []

    # Query nutrient data for all foods
    placeholders = ",".join("?" * len(fdc_ids))
    nutrient_ids = [ENERGY_ID, PROTEIN_ID, FAT_ID, CARB_ID]
    nutrient_placeholders = ",".join("?" * len(nutrient_ids))

    query = f"""
        SELECT f.fdc_id, f.description, fn.nutrient_id, fn.amount
        FROM foods f
        LEFT JOIN food_nutrients fn ON f.fdc_id = fn.fdc_id
            AND fn.nutrient_id IN ({nutrient_placeholders})
        WHERE f.fdc_id IN ({placeholders})
    """
    params = nutrient_ids + fdc_ids

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    # Build per-food nutrient data
    food_nutrients: dict[int, dict[int, float]] = {}
    food_descriptions: dict[int, str] = {}

    for fdc_id, description, nutrient_id, amount in rows:
        if fdc_id not in food_nutrients:
            food_nutrients[fdc_id] = {}
            food_descriptions[fdc_id] = description
        if nutrient_id is not None:
            food_nutrients[fdc_id][nutrient_id] = amount or 0.0

    # Check each food
    for fdc_id in fdc_ids:
        nutrients = food_nutrients.get(fdc_id, {})
        description = food_descriptions.get(fdc_id, f"Food {fdc_id}")

        energy = nutrients.get(ENERGY_ID, 0.0)
        protein = nutrients.get(PROTEIN_ID, 0.0)
        fat = nutrients.get(FAT_ID, 0.0)
        carbs = nutrients.get(CARB_ID, 0.0)

        # Check for missing energy with non-zero macros
        has_macros = protein > 0 or fat > 0 or carbs > 0

        if energy == 0 and has_macros:
            fallback = calculate_fallback_energy(protein, carbs, fat)
            issues.append(
                DataQualityIssue(
                    fdc_id=fdc_id,
                    description=description,
                    issue_type="missing_energy",
                    severity="warning",
                    message=f"Missing energy value (calculated: {fallback:.0f} kcal)",
                    fallback_value=fallback,
                )
            )
        elif energy == 0 and not has_macros:
            # All zeros - likely incomplete data
            issues.append(
                DataQualityIssue(
                    fdc_id=fdc_id,
                    description=description,
                    issue_type="zero_macros",
                    severity="warning",
                    message="All macro nutrients are zero or missing",
                    fallback_value=None,
                )
            )
        elif has_macros:
            # Check for inconsistency between stored and calculated energy
            calculated = calculate_fallback_energy(protein, carbs, fat)
            # Allow 20% tolerance
            if energy > 0 and abs(calculated - energy) / energy > 0.20:
                issues.append(
                    DataQualityIssue(
                        fdc_id=fdc_id,
                        description=description,
                        issue_type="inconsistent_energy",
                        severity="warning",
                        message=f"Stored energy ({energy:.0f}) differs from calculated ({calculated:.0f})",
                        fallback_value=None,
                    )
                )

    return issues


def format_quality_warnings(issues: list[DataQualityIssue]) -> list[str]:
    """Format data quality issues as human-readable warning strings.

    Args:
        issues: List of detected issues

    Returns:
        List of warning message strings
    """
    warnings = []
    for issue in issues:
        if issue.issue_type == "missing_energy":
            warnings.append(
                f"{issue.description} (fdc_id={issue.fdc_id}): {issue.message}"
            )
        elif issue.issue_type == "zero_macros":
            warnings.append(
                f"{issue.description} (fdc_id={issue.fdc_id}): {issue.message}"
            )
        elif issue.issue_type == "inconsistent_energy":
            warnings.append(
                f"{issue.description} (fdc_id={issue.fdc_id}): {issue.message}"
            )
    return warnings


def get_fallback_energies(
    issues: list[DataQualityIssue],
) -> dict[int, float]:
    """Extract fallback energy values from quality issues.

    Returns a mapping from fdc_id to calculated energy for foods
    that have missing_energy issues with fallback values.

    Args:
        issues: List of detected issues

    Returns:
        Dict mapping fdc_id to fallback energy value
    """
    return {
        issue.fdc_id: issue.fallback_value
        for issue in issues
        if issue.issue_type == "missing_energy" and issue.fallback_value is not None
    }
