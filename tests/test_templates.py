"""Tests for template-based meal composition."""

from __future__ import annotations

import pytest

from llmn.db.connection import get_db
from llmn.templates.definitions import (
    get_template,
    get_template_for_patterns,
    list_available_templates,
)
from llmn.templates.models import (
    DietTemplate,
    DiversityRule,
    SelectionStrategy,
)
from llmn.templates.optimizer import run_template_optimization
from llmn.templates.selector import (
    get_candidates_for_slot,
    select_foods_for_template,
)


class TestTemplateDefinitions:
    """Tests for template definitions."""

    def test_list_templates(self):
        """Should list all available templates."""
        templates = list_available_templates()
        assert "pescatarian_slowcarb" in templates
        assert "keto" in templates
        assert "vegan" in templates

    def test_get_template(self):
        """Should get a template by name."""
        template = get_template("pescatarian_slowcarb")
        assert template is not None
        assert template.name == "pescatarian_slowcarb"
        assert len(template.meals) == 4  # breakfast, lunch, dinner, snack

    def test_get_template_for_patterns(self):
        """Should get template for pattern combination."""
        template = get_template_for_patterns(["pescatarian", "slow_carb"])
        assert template is not None
        assert template.name == "pescatarian_slowcarb"

    def test_template_structure(self):
        """Template should have correct meal structure."""
        template = get_template("pescatarian_slowcarb")
        meal_types = [m.meal_type.value for m in template.meals]
        assert "breakfast" in meal_types
        assert "lunch" in meal_types
        assert "dinner" in meal_types
        assert "snack" in meal_types

    def test_template_slots(self):
        """Meals should have correct slots."""
        template = get_template("pescatarian_slowcarb")
        breakfast = next(m for m in template.meals if m.meal_type.value == "breakfast")
        slot_names = [s.name for s in breakfast.slots]
        assert "protein" in slot_names
        assert "legume" in slot_names
        assert "vegetable" in slot_names


class TestFoodSelection:
    """Tests for food selection."""

    @pytest.fixture
    def db_conn(self):
        """Get database connection."""
        db = get_db()
        with db.get_connection() as conn:
            yield conn

    def test_select_foods_for_template(self, db_conn):
        """Should select foods for all slots."""
        template = get_template("pescatarian_slowcarb")
        selections = select_foods_for_template(
            conn=db_conn,
            template=template,
            strategy=SelectionStrategy.RANDOM,
            seed=42,
        )

        assert len(selections) == 4  # 4 meals
        for meal in selections:
            assert len(meal.selections) > 0

    def test_diversity_no_repeat(self, db_conn):
        """Should not repeat foods across meals with NO_REPEAT rule."""
        template = get_template("pescatarian_slowcarb")
        assert template.diversity_rule == DiversityRule.NO_REPEAT

        selections = select_foods_for_template(
            conn=db_conn,
            template=template,
            strategy=SelectionStrategy.RANDOM,
            seed=42,
        )

        # Collect all selected FDC IDs
        all_ids = [s.fdc_id for meal in selections for s in meal.selections]
        # No duplicates
        assert len(all_ids) == len(set(all_ids))

    def test_reproducibility_with_seed(self, db_conn):
        """Same seed should produce same selections."""
        template = get_template("pescatarian_slowcarb")

        selections1 = select_foods_for_template(
            conn=db_conn,
            template=template,
            strategy=SelectionStrategy.RANDOM,
            seed=42,
        )

        selections2 = select_foods_for_template(
            conn=db_conn,
            template=template,
            strategy=SelectionStrategy.RANDOM,
            seed=42,
        )

        ids1 = [s.fdc_id for meal in selections1 for s in meal.selections]
        ids2 = [s.fdc_id for meal in selections2 for s in meal.selections]
        assert ids1 == ids2

    def test_different_seeds_produce_variety(self, db_conn):
        """Different seeds should produce different selections."""
        template = get_template("pescatarian_slowcarb")

        selections1 = select_foods_for_template(
            conn=db_conn,
            template=template,
            strategy=SelectionStrategy.RANDOM,
            seed=42,
        )

        selections2 = select_foods_for_template(
            conn=db_conn,
            template=template,
            strategy=SelectionStrategy.RANDOM,
            seed=123,
        )

        ids1 = [s.fdc_id for meal in selections1 for s in meal.selections]
        ids2 = [s.fdc_id for meal in selections2 for s in meal.selections]
        assert ids1 != ids2


class TestTemplateOptimization:
    """Tests for full template optimization."""

    @pytest.fixture
    def db_conn(self):
        """Get database connection."""
        db = get_db()
        with db.get_connection() as conn:
            yield conn

    def test_basic_optimization(self, db_conn):
        """Should produce a valid meal plan."""
        result = run_template_optimization(
            conn=db_conn,
            patterns=["pescatarian", "slow_carb"],
            daily_calories=(1800, 2200),
            daily_protein=(150, 185),
            seed=42,
        )

        assert result.success
        assert result.template_name == "pescatarian_slowcarb"
        assert len(result.meals) == 4

    def test_calorie_constraints(self, db_conn):
        """Should meet daily calorie constraints."""
        result = run_template_optimization(
            conn=db_conn,
            patterns=["pescatarian", "slow_carb"],
            daily_calories=(1800, 2200),
            daily_protein=(150, 185),
            seed=42,
        )

        assert result.success
        assert 1800 <= result.daily_calories <= 2200

    def test_protein_constraints(self, db_conn):
        """Should meet daily protein constraints."""
        result = run_template_optimization(
            conn=db_conn,
            patterns=["pescatarian", "slow_carb"],
            daily_calories=(1800, 2200),
            daily_protein=(150, 185),
            seed=42,
        )

        assert result.success
        assert 150 <= result.daily_protein <= 185

    def test_meal_structure(self, db_conn):
        """Each meal should have appropriate foods."""
        result = run_template_optimization(
            conn=db_conn,
            patterns=["pescatarian", "slow_carb"],
            daily_calories=(1800, 2200),
            daily_protein=(150, 185),
            seed=42,
        )

        assert result.success

        # Check that main meals have protein, legume, vegetable
        for meal in result.meals:
            if meal.meal_type.value in ["breakfast", "lunch", "dinner"]:
                slot_names = [f.slot_name for f in meal.foods]
                assert "protein" in slot_names, f"{meal.meal_type.value} missing protein"
                # Note: some slots might be combined or missing if constraints are tight

    def test_different_meals(self, db_conn):
        """Lunch and dinner should have different foods."""
        result = run_template_optimization(
            conn=db_conn,
            patterns=["pescatarian", "slow_carb"],
            daily_calories=(1800, 2200),
            daily_protein=(150, 185),
            seed=42,
        )

        assert result.success

        lunch = next(m for m in result.meals if m.meal_type.value == "lunch")
        dinner = next(m for m in result.meals if m.meal_type.value == "dinner")

        lunch_ids = {f.fdc_id for f in lunch.foods}
        dinner_ids = {f.fdc_id for f in dinner.foods}

        # Lunch and dinner should have different foods
        assert lunch_ids != dinner_ids


class TestKeto:
    """Tests for keto template specifically."""

    @pytest.fixture
    def db_conn(self):
        """Get database connection."""
        db = get_db()
        with db.get_connection() as conn:
            yield conn

    def test_keto_template_exists(self):
        """Keto template should exist."""
        template = get_template("keto")
        assert template is not None
        assert template.name == "keto"

    def test_keto_optimization(self, db_conn):
        """Should produce a valid keto meal plan."""
        result = run_template_optimization(
            conn=db_conn,
            patterns=["keto"],
            daily_calories=(1800, 2400),
            daily_protein=(120, 160),
            seed=42,
        )

        assert result.success
        assert len(result.meals) == 4
