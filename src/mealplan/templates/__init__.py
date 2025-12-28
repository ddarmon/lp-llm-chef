"""Template-based meal composition.

This module provides a template-based approach to meal planning that mirrors
how humans naturally compose meals: select one food from each category
(protein, legume, vegetables), then optimize quantities.

Key differences from Stigler-style optimization:
- Discrete food selection FIRST, then continuous quantity optimization
- Enforced meal structure (1 protein + 1 legume + 1-2 vegetables per meal)
- Built-in diversity (no repeating foods across meals)
- Slot-specific portion targets (not uniform 100g)
"""

from __future__ import annotations

from mealplan.templates.models import (
    DietTemplate,
    MealTemplate,
    SelectedFood,
    SelectedMeal,
    SlotDefinition,
    TemplateOptimizationRequest,
    TemplateResult,
)

__all__ = [
    "DietTemplate",
    "MealTemplate",
    "SelectedFood",
    "SelectedMeal",
    "SlotDefinition",
    "TemplateOptimizationRequest",
    "TemplateResult",
]
