"""Generate LLM-ready prompts from optimization results."""

from __future__ import annotations

import re
from pathlib import Path
from string import Template
from typing import Optional

import yaml

from mealplan.optimizer.models import OptimizationResult


def clean_food_name(description: str) -> str:
    """Clean up USDA food descriptions to be more readable.

    Examples:
        "Fish, salmon, Atlantic, farmed, raw" -> "Salmon"
        "Beans, black turtle, mature seeds, cooked, boiled, without salt" -> "Black beans"
        "Egg, whole, raw, fresh" -> "Eggs"
    """
    desc = description

    # Common patterns to simplify
    replacements = [
        # Remove common suffixes
        (r",?\s*raw$", ""),
        (r",?\s*fresh$", ""),
        (r",?\s*cooked.*$", ""),
        (r",?\s*boiled.*$", ""),
        (r",?\s*without salt.*$", ""),
        (r",?\s*with salt.*$", ""),
        (r",?\s*drained.*$", ""),
        (r",?\s*canned.*$", " (canned)"),
        (r",?\s*mature seeds$", ""),
        (r"\s*\(Includes foods.*?\)", ""),
        (r"\s*\(garbanzo beans, bengal gram\)", ""),
        (r",?\s*year round average$", ""),
        (r",?\s*all commercial varieties$", ""),
    ]

    for pattern, replacement in replacements:
        desc = re.sub(pattern, replacement, desc, flags=re.IGNORECASE)

    # Specific food name simplifications
    name_map = {
        "Fish, salmon, Atlantic, farmed": "Salmon",
        "Fish, salmon": "Salmon",
        "Fish, tuna, yellowfin": "Tuna",
        "Fish, tuna": "Tuna",
        "Fish, cod, Atlantic": "Cod",
        "Fish, cod": "Cod",
        "Fish, tilapia": "Tilapia",
        "Fish, mackerel, Atlantic": "Mackerel",
        "Fish, mackerel": "Mackerel",
        "Fish, sardine, Atlantic": "Sardines (canned)",
        "Fish, sardine": "Sardines",
        "Crustaceans, shrimp, farm raised": "Shrimp",
        "Crustaceans, shrimp": "Shrimp",
        "Egg, whole": "Eggs",
        "Beans, black turtle": "Black beans",
        "Beans, pinto": "Pinto beans",
        "Chickpeas": "Chickpeas",
        "Lentils, mature seeds": "Lentils",
        "Lentils": "Lentils",
        "Oil, olive, extra virgin": "Olive oil",
        "Oil, olive": "Olive oil",
        "Nuts, almonds": "Almonds",
        "Cottage cheese, full fat, large or small curd": "Cottage cheese",
        "Cottage cheese": "Cottage cheese",
        "Tomatoes, red, ripe": "Tomatoes",
        "Peppers, sweet, red": "Red bell peppers",
        "Peppers, sweet": "Bell peppers",
        "Garlic": "Garlic",
        "Onions": "Onions",
        "Spinach": "Spinach",
        "Broccoli": "Broccoli",
        "Kale": "Kale",
        "Cabbage": "Cabbage",
        "Cauliflower": "Cauliflower",
        "Asparagus": "Asparagus",
        "Avocados": "Avocado",
        "Sauerkraut": "Sauerkraut",
        "Hummus, commercial": "Hummus",
        "Hummus": "Hummus",
        "Anchovies": "Anchovies",
    }

    # Try to match against known names (longest match first)
    for usda_name, simple_name in sorted(name_map.items(), key=lambda x: -len(x[0])):
        if desc.lower().startswith(usda_name.lower()):
            return simple_name

    # If no match, just clean up and return
    # Remove leading category (e.g., "Fish, " or "Nuts, ")
    if ", " in desc:
        parts = desc.split(", ")
        if parts[0].lower() in ("fish", "nuts", "beans", "oil", "egg", "crustaceans"):
            desc = " ".join(parts[1:]).strip()

    return desc.strip(", ")


def clean_food_name_with_context(description: str) -> tuple[str, str]:
    """Clean up USDA food descriptions while preserving preparation state.

    Returns a tuple of (clean_name, prep_state) where prep_state indicates
    how the food should be prepared or used.

    Examples:
        "Fish, salmon, Atlantic, farmed, raw" -> ("Salmon", "raw")
        "Beans, black turtle, mature seeds, cooked, boiled" -> ("Black beans", "cooked")
        "Fish, sardine, Atlantic, canned in oil" -> ("Sardines", "canned")
        "Hummus, commercial" -> ("Hummus", "")
    """
    desc_lower = description.lower()

    # Detect preparation state from the original description
    prep_state = ""
    if "canned" in desc_lower:
        prep_state = "canned"
    elif "cooked" in desc_lower or "boiled" in desc_lower:
        prep_state = "cooked"
    elif "raw" in desc_lower or "fresh" in desc_lower:
        prep_state = "raw"

    # Get the clean name using the existing function
    clean_name = clean_food_name(description)

    # Remove any prep state that might have been added by clean_food_name
    # (like "(canned)" which we'll handle separately)
    clean_name = re.sub(r"\s*\(canned\)\s*", "", clean_name).strip()

    return clean_name, prep_state


DEFAULT_TEMPLATE = """# Meal Planning Request

## Optimized Food Allocation

$FOOD_TABLE

## My Preferences

- Cuisine styles I enjoy: $CUISINES
- Cooking skill level: $SKILL_LEVEL
- Max prep time per meal: $MAX_PREP_TIME
- Kitchen equipment: $EQUIPMENT
- Flavor preferences: $FLAVOR_NOTES

## Request

Create a $NUM_DAYS-day meal plan using EXACTLY the food quantities listed above (scaled appropriately for $NUM_DAYS days).

For each day, provide:
1. **Breakfast** - prep time, ingredients with amounts, brief instructions
2. **Lunch** - prep time, ingredients with amounts, brief instructions
3. **Dinner** - prep time, ingredients with amounts, brief instructions
4. **Snacks** (if applicable)

Requirements:
- Use ALL of the allocated foods across the meal plan
- Keep recipes simple and practical for home cooking
- Include variety across days (don't repeat the same meal)
- Suggest meal prep strategies where applicable

$ADDITIONAL_NOTES
"""


class LLMPromptGenerator:
    """Generate prompts for LLM-based recipe generation."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the generator.

        Args:
            config_dir: Directory containing templates and preferences.
                        Defaults to ~/.mealplan/
        """
        self.config_dir = config_dir or Path.home() / ".mealplan"
        self._template: Optional[str] = None
        self._preferences: Optional[dict] = None

    @property
    def template(self) -> str:
        """Load custom template or use default."""
        if self._template is None:
            template_path = self.config_dir / "llm_prompt_template.md"
            if template_path.exists():
                self._template = template_path.read_text()
            else:
                self._template = DEFAULT_TEMPLATE
        return self._template

    @property
    def preferences(self) -> dict:
        """Load user preferences from YAML."""
        if self._preferences is None:
            pref_path = self.config_dir / "preferences.yaml"
            if pref_path.exists():
                with open(pref_path) as f:
                    self._preferences = yaml.safe_load(f) or {}
            else:
                self._preferences = self._default_preferences()
        return self._preferences

    def _default_preferences(self) -> dict:
        """Return default preferences."""
        return {
            "cuisines": ["any"],
            "skill_level": "intermediate",
            "max_prep_time": "30 minutes",
            "equipment": ["stovetop", "oven"],
            "flavor_notes": "",
            "additional_notes": "",
        }

    def generate(
        self,
        result: OptimizationResult,
        days: int = 1,
        profile_name: Optional[str] = None,
    ) -> str:
        """Generate a complete LLM prompt from optimization result.

        Args:
            result: Optimization result with food allocations
            days: Number of days for the meal plan
            profile_name: Optional profile name for context

        Returns:
            Complete prompt ready for LLM input
        """
        # Build food table with cleaned names
        # Include weekly totals when planning for multiple days
        if days > 1:
            food_lines = [
                "| Food | Daily Amount | Weekly Total |",
                "|------|--------------|--------------|",
            ]
            for food in sorted(result.foods, key=lambda f: -f.grams):
                daily = f"{food.grams:.0f}g"
                weekly = f"{food.grams * days:.0f}g"
                name = clean_food_name(food.description)
                food_lines.append(f"| {name} | {daily} | {weekly} |")

            # Add totals
            total_grams = sum(f.grams for f in result.foods)
            food_lines.append(
                f"| **Total** | **{total_grams:.0f}g** | **{total_grams * days:.0f}g** |"
            )
        else:
            food_lines = ["| Food | Daily Amount |", "|------|--------------|"]
            for food in sorted(result.foods, key=lambda f: -f.grams):
                amount_str = f"{food.grams:.0f}g"
                name = clean_food_name(food.description)
                food_lines.append(f"| {name} | {amount_str} |")

            # Add totals
            total_grams = sum(f.grams for f in result.foods)
            food_lines.append(f"| **Total** | **{total_grams:.0f}g** |")

        if result.total_cost:
            food_lines.append(f"\n**Daily Cost:** ${result.total_cost:.2f}")

        food_table = "\n".join(food_lines)

        # Format preferences
        prefs = self.preferences
        cuisines = ", ".join(prefs.get("cuisines", ["any"]))
        equipment = ", ".join(prefs.get("equipment", ["stovetop", "oven"]))
        flavor_notes = prefs.get("flavor_notes", "")
        additional_notes = prefs.get("additional_notes", "")

        # Add profile info to additional notes if provided
        if profile_name and additional_notes:
            additional_notes = f"Profile used: {profile_name}\n\n{additional_notes}"
        elif profile_name:
            additional_notes = f"Profile used: {profile_name}"

        # Substitute template
        template = Template(self.template)
        prompt = template.safe_substitute(
            FOOD_TABLE=food_table,
            CUISINES=cuisines,
            SKILL_LEVEL=prefs.get("skill_level", "intermediate"),
            MAX_PREP_TIME=prefs.get("max_prep_time", "30 minutes"),
            EQUIPMENT=equipment,
            FLAVOR_NOTES=flavor_notes if flavor_notes else "No specific preferences",
            NUM_DAYS=str(days),
            ADDITIONAL_NOTES=additional_notes,
        )

        return prompt

    def save_default_template(self) -> Path:
        """Save the default template to the config directory.

        Returns:
            Path to the saved template
        """
        self.config_dir.mkdir(parents=True, exist_ok=True)
        template_path = self.config_dir / "llm_prompt_template.md"
        template_path.write_text(DEFAULT_TEMPLATE)
        return template_path

    def save_default_preferences(self) -> Path:
        """Save default preferences to the config directory.

        Returns:
            Path to the saved preferences file
        """
        self.config_dir.mkdir(parents=True, exist_ok=True)
        pref_path = self.config_dir / "preferences.yaml"

        default_prefs = {
            "cuisines": ["Mediterranean", "Asian", "Mexican"],
            "skill_level": "intermediate",
            "max_prep_time": "30 minutes",
            "equipment": ["stovetop", "oven", "blender"],
            "flavor_notes": "Prefer savory over sweet\nLike spicy food\nEnjoy garlic and herbs",
            "additional_notes": "I meal prep on Sundays for the week\nPrefer meals that reheat well\nBreakfast should be quick (under 10 min)",
        }

        with open(pref_path, "w") as f:
            yaml.dump(default_prefs, f, default_flow_style=False, sort_keys=False)

        return pref_path


def generate_llm_prompt(
    result: OptimizationResult,
    days: int = 1,
    profile_name: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> str:
    """Convenience function to generate an LLM prompt.

    Args:
        result: Optimization result
        days: Number of days for meal plan
        profile_name: Optional profile name
        config_dir: Optional config directory

    Returns:
        Complete LLM prompt
    """
    generator = LLMPromptGenerator(config_dir)
    return generator.generate(result, days, profile_name)
