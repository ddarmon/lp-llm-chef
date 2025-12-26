"""Explore module for food discovery and what-if analysis."""

from __future__ import annotations

from mealplan.explore.foods import (
    compare_foods,
    explore_foods,
    get_food_nutrients,
)
from mealplan.explore.whatif import run_whatif_analysis

__all__ = [
    "compare_foods",
    "explore_foods",
    "get_food_nutrients",
    "run_whatif_analysis",
]
