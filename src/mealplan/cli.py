"""CLI interface using Typer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from mealplan.config import get_settings
from mealplan.db import get_db
from mealplan.db.queries import (
    FoodQueries,
    OptimizationRunQueries,
    PriceQueries,
    ProfileQueries,
    TagQueries,
)

app = typer.Typer(
    help="Personal meal planning optimization with LP/QP",
    no_args_is_help=True,
)
console = Console()

# Subcommand groups
prices_app = typer.Typer(help="Manage food prices")
tags_app = typer.Typer(help="Manage food tags")
profile_app = typer.Typer(help="Manage constraint profiles")
schema_app = typer.Typer(help="Export schemas for LLM agents")
explore_app = typer.Typer(help="Explore foods and run what-if analysis")

app.add_typer(prices_app, name="prices")
app.add_typer(tags_app, name="tags")
app.add_typer(profile_app, name="profile")
app.add_typer(schema_app, name="schema")
app.add_typer(explore_app, name="explore")

# Patterns subcommand
patterns_app = typer.Typer(help="Manage dietary patterns")
app.add_typer(patterns_app, name="patterns")

# Weight tracking subcommands
user_app = typer.Typer(help="Manage user profile for weight tracking")
weight_app = typer.Typer(help="Log and track weight with EMA trend")
calories_app = typer.Typer(help="Log planned calorie intake")
tdee_app = typer.Typer(help="TDEE estimation and adaptive targets")

app.add_typer(user_app, name="user")
app.add_typer(weight_app, name="weight")
app.add_typer(calories_app, name="calories")
app.add_typer(tdee_app, name="tdee")


# ============================================================================
# Helper for JSON output
# ============================================================================


def output_json(response: dict, file=None) -> None:
    """Output JSON response to stdout or file."""
    json_str = json.dumps(response, indent=2)
    if file:
        file.write(json_str)
    else:
        print(json_str)


def ensure_tracking_tables() -> None:
    """Ensure tracking tables exist (idempotent)."""
    db = get_db()
    db.initialize_schema()


def require_food_data() -> None:
    """Check that database has been initialized with food data.

    Raises typer.Exit(1) with a friendly message if no food data exists.
    """
    db = get_db()
    try:
        has_foods = db.table_exists("foods") and db.get_table_count("foods") > 0
    except Exception:
        has_foods = False

    if not has_foods:
        console.print("[red]Database not initialized with food data.[/red]")
        console.print("Run: [cyan]mealplan init <path-to-usda-csv>[/cyan]")
        raise typer.Exit(1)


# Callbacks for tracking sub-apps to auto-create tables on first use
@user_app.callback()
def user_callback() -> None:
    """Ensure tracking tables exist before any user command."""
    ensure_tracking_tables()


@weight_app.callback()
def weight_callback() -> None:
    """Ensure tracking tables exist before any weight command."""
    ensure_tracking_tables()


@calories_app.callback()
def calories_callback() -> None:
    """Ensure tracking tables exist before any calories command."""
    ensure_tracking_tables()


@tdee_app.callback()
def tdee_callback() -> None:
    """Ensure tracking tables exist before any tdee command."""
    ensure_tracking_tables()


# Callbacks for sub-apps that require food data
@prices_app.callback()
def prices_callback() -> None:
    """Ensure food data exists before any prices command."""
    require_food_data()


@tags_app.callback()
def tags_callback() -> None:
    """Ensure food data exists before any tags command."""
    require_food_data()


@explore_app.callback()
def explore_callback() -> None:
    """Ensure food data exists before any explore command."""
    require_food_data()


@patterns_app.callback()
def patterns_callback() -> None:
    """Ensure food data exists before any patterns command."""
    require_food_data()


# ============================================================================
# Main Commands
# ============================================================================


@app.command()
def init(
    usda_path: Path = typer.Argument(
        ..., help="Path to USDA FoodData Central CSV directory"
    ),
    db_path: Optional[Path] = typer.Option(
        None, "--db", help="Custom database path"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON"
    ),
) -> None:
    """Initialize database from USDA data."""
    from mealplan.data.usda_loader import USDALoader
    from mealplan.db.connection import DatabaseConnection

    # Use custom path or default
    if db_path:
        db = DatabaseConnection(db_path)
    else:
        db = get_db()

    if not json_output:
        console.print(f"Initializing database at: {db.db_path}")

    # Create schema
    db.initialize_schema()
    if not json_output:
        console.print("[green]Schema created[/green]")

    # Load USDA data
    settings = get_settings()
    data_types = settings.usda.data_types

    if not json_output:
        console.print(f"Loading USDA data from: {usda_path}")
        console.print(f"Data types: {', '.join(data_types)}")

    if json_output:
        with db.get_connection() as conn:
            loader = USDALoader(usda_path, conn)
            counts = loader.load_all(data_types)
        output_json({
            "success": True,
            "command": "init",
            "data": {
                "db_path": str(db.db_path),
                "foods_loaded": counts["foods"],
                "nutrients_loaded": counts["nutrients"],
                "food_nutrients_loaded": counts["food_nutrients"],
                "servings_loaded": counts.get("servings", 0),
            },
            "human_summary": f"Loaded {counts['foods']} foods with {counts['food_nutrients']} nutrient records",
        })
    else:
        with console.status("[bold green]Loading data...") as status:
            with db.get_connection() as conn:
                loader = USDALoader(usda_path, conn)

                def progress(msg: str) -> None:
                    status.update(f"[bold green]{msg}")
                    console.print(f"  {msg}")

                counts = loader.load_all(data_types, progress)

        console.print()
        console.print("[bold green]Database initialized successfully![/bold green]")
        console.print(f"  Foods loaded: {counts['foods']}")
        console.print(f"  Nutrients loaded: {counts['nutrients']}")
        console.print(f"  Food-nutrient records: {counts['food_nutrients']}")
        if counts.get("servings"):
            console.print(f"  Serving sizes: {counts['servings']}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search term for food descriptions"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search for foods by description."""
    require_food_data()
    db = get_db()
    with db.get_connection() as conn:
        results = FoodQueries.search_foods(conn, query, limit)

    if json_output:
        output_json({
            "success": True,
            "command": "search",
            "data": {
                "query": query,
                "results": [
                    {
                        "fdc_id": row["fdc_id"],
                        "description": row["description"],
                        "data_type": row["data_type"],
                        "price_per_100g": row["price_per_100g"],
                        "tags": row["tags"].split(", ") if row["tags"] else [],
                    }
                    for row in results
                ],
                "total_matches": len(results),
            },
            "human_summary": f"Found {len(results)} foods matching '{query}'",
        })
        return

    if not results:
        console.print(f"[yellow]No foods found matching '{query}'[/yellow]")
        return

    table = Table(title=f"Foods matching '{query}'")
    table.add_column("FDC ID", style="cyan")
    table.add_column("Description")
    table.add_column("Type", style="dim")
    table.add_column("Price/100g", style="green")
    table.add_column("Tags", style="blue")

    for row in results:
        price_str = f"${row['price_per_100g']:.2f}" if row['price_per_100g'] is not None else ""
        tags_str = row['tags'] if row['tags'] else ""
        table.add_row(
            str(row["fdc_id"]),
            row["description"],
            row["data_type"] or "",
            price_str,
            tags_str,
        )

    console.print(table)
    console.print(f"[dim]Showing {len(results)} results[/dim]")


@app.command()
def info(
    fdc_id: int = typer.Argument(..., help="FDC ID of food to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show detailed nutrient information for a food."""
    require_food_data()
    db = get_db()
    with db.get_connection() as conn:
        food = FoodQueries.get_food_by_id(conn, fdc_id)
        if not food:
            if json_output:
                output_json({
                    "success": False,
                    "command": "info",
                    "errors": [f"Food with FDC ID {fdc_id} not found"],
                    "suggestions": ["Use 'mealplan search <query>' to find valid FDC IDs"],
                })
            else:
                console.print(f"[red]Food with FDC ID {fdc_id} not found[/red]")
            raise typer.Exit(1)

        nutrients = FoodQueries.get_food_nutrients(conn, fdc_id)
        price = PriceQueries.get_price(conn, fdc_id)
        tags = TagQueries.get_tags_for_food(conn, fdc_id)

    if json_output:
        output_json({
            "success": True,
            "command": "info",
            "data": {
                "fdc_id": fdc_id,
                "description": food["description"],
                "data_type": food["data_type"],
                "is_active": bool(food["is_active"]),
                "price_per_100g": price["price_per_100g"] if price else None,
                "price_source": price["price_source"] if price else None,
                "tags": tags,
                "nutrients": [
                    {
                        "nutrient_id": row["nutrient_id"],
                        "name": row["display_name"] or row["name"],
                        "amount": row["amount"],
                        "unit": row["unit"],
                    }
                    for row in nutrients
                ],
            },
            "human_summary": f"{food['description']}: {len(nutrients)} nutrients",
        })
        return

    console.print(f"\n[bold]{food['description']}[/bold]")
    console.print(f"FDC ID: {fdc_id}")
    console.print(f"Data Type: {food['data_type'] or 'N/A'}")
    console.print(f"Active: {'Yes' if food['is_active'] else 'No'}")

    if price:
        console.print(
            f"Price: ${price['price_per_100g']:.2f}/100g "
            f"({price['price_source'] or 'unknown source'})"
        )
    else:
        console.print("[yellow]No price set[/yellow]")

    if tags:
        console.print(f"Tags: {', '.join(tags)}")

    if nutrients:
        table = Table(title="\nNutrients (per 100g)")
        table.add_column("Nutrient")
        table.add_column("Amount", justify="right")
        table.add_column("Unit")

        for row in nutrients:
            table.add_row(
                row["display_name"] or row["name"],
                f"{row['amount']:.2f}",
                row["unit"],
            )

        console.print(table)


@app.command()
def optimize(
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Constraint profile name"
    ),
    profile_file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="YAML constraint file"
    ),
    foods: Optional[str] = typer.Option(
        None, "--foods", help="Comma-separated FDC IDs to include (bypasses tag filtering)"
    ),
    pattern: Optional[list[str]] = typer.Option(
        None, "--pattern", help="Dietary pattern (pescatarian, keto, slow_carb, etc). Repeatable."
    ),
    min_protein: Optional[float] = typer.Option(
        None, "--min-protein", help="Minimum protein per 100g for pattern foods (e.g., 10 for high-protein pool)"
    ),
    goal: Optional[str] = typer.Option(
        None, "--goal", help="Body comp goal, e.g. 'fat_loss:185lbs:165lbs' or 'maintenance'"
    ),
    days: int = typer.Option(1, "--days", "-d", help="Number of days to plan"),
    output: str = typer.Option(
        "table", "--output", "-o", help="Output format: table, json, markdown"
    ),
    practical: bool = typer.Option(
        False, "--practical", help="Output consolidated meals (4-6 foods/meal) with shopping list"
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Save run to history"),
    minimize_cost: bool = typer.Option(
        False, "--minimize-cost", help="Minimize cost instead of finding diverse feasible solution"
    ),
    max_foods: int = typer.Option(
        300, "--max-foods", help="Maximum foods to consider (randomly sampled if exceeded)"
    ),
    max_foods_in_solution: Optional[int] = typer.Option(
        None, "--max-foods-in-solution", help="Limit distinct foods in solution (uses two-phase solver)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Display KKT optimality conditions"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON with agent-friendly envelope"
    ),
    allocate_meals: bool = typer.Option(
        False, "--allocate-meals", help="Distribute foods into meal slots (breakfast/lunch/dinner/snack)"
    ),
    multiperiod: bool = typer.Option(
        False, "--multiperiod", "-m", help="Use multi-period solver (auto-detected if profile has 'meals' section)"
    ),
    include_all_foods: bool = typer.Option(
        False, "--include-all-foods", help="Use full USDA database instead of curated staples (may include exotic foods)"
    ),
    template: bool = typer.Option(
        False, "--template", "-t", help="Use template-based meal composition (recommended for realistic meals)"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for template-based selection (for reproducibility)"
    ),
) -> None:
    """Run meal plan optimization."""
    require_food_data()

    from mealplan.agent.response import create_response
    from mealplan.export.formatters import format_result
    from mealplan.explore.diagnosis import diagnose_infeasibility
    from mealplan.optimizer.constraints import (
        deserialize_profile_json,
        has_meals_section,
        load_multiperiod_profile_from_yaml,
        load_profile_from_yaml,
    )
    from mealplan.optimizer.models import OptimizationRequest, NutrientConstraint
    from mealplan.optimizer.multiperiod_solver import solve_multiperiod_diet
    from mealplan.optimizer.multiperiod_models import (
        MultiPeriodRequest,
        derive_default_meal_configs,
    )
    from mealplan.optimizer.solver import solve_diet_problem

    db = get_db()
    profile_id: Optional[int] = None
    profile_name: Optional[str] = None

    # Determine if we should use multi-period solver
    use_multiperiod = multiperiod
    mp_request = None  # Will be set if using multi-period

    # Load constraints
    if profile_file:
        # Load from YAML file
        if not profile_file.exists():
            if json_output:
                output_json({
                    "success": False,
                    "command": "optimize",
                    "errors": [f"Profile file not found: {profile_file}"],
                    "suggestions": [
                        "Check the file path",
                        "Use 'mealplan profile list' to see saved profiles",
                    ],
                })
            else:
                console.print(f"[red]Profile file not found: {profile_file}[/red]")
            raise typer.Exit(1)

        # Auto-detect multi-period from profile
        if has_meals_section(profile_file):
            use_multiperiod = True
            if not json_output:
                console.print("[dim]Auto-detected multi-period profile (has 'meals' section)[/dim]")

        if use_multiperiod:
            mp_request = load_multiperiod_profile_from_yaml(profile_file)
            # Also load single-period for fallback/info
            request = load_profile_from_yaml(profile_file)
        else:
            request = load_profile_from_yaml(profile_file)
        profile_name = profile_file.stem
    elif profile:
        # Load from database
        with db.get_connection() as conn:
            profile_row = ProfileQueries.get_profile_by_name(conn, profile)
            if not profile_row:
                if json_output:
                    output_json({
                        "success": False,
                        "command": "optimize",
                        "errors": [f"Profile not found: {profile}"],
                        "suggestions": [
                            "Use 'mealplan profile list' to see available profiles",
                            "Use 'mealplan profile create' to create a new profile",
                        ],
                    })
                else:
                    console.print(f"[red]Profile not found: {profile}[/red]")
                raise typer.Exit(1)
            profile_id = profile_row["profile_id"]
            profile_name = profile_row["name"]
            # Parse stored JSON into request
            constraints_data = json.loads(profile_row["constraints_json"])
            request = deserialize_profile_json(constraints_data)
            if not json_output:
                console.print(f"[yellow]Loading profile '{profile}' from database[/yellow]")
    else:
        # Use default request
        request = OptimizationRequest()
        if not json_output:
            console.print("[yellow]Using default constraints[/yellow]")

    request.planning_days = days
    request.max_foods = max_foods

    # Create default multi-period request if --multiperiod flag but no YAML with meals
    if use_multiperiod and mp_request is None:
        # Derive meal configs from the request's daily constraints
        daily_cal_range = request.calorie_range
        meal_configs = derive_default_meal_configs(daily_cal_range, request.nutrient_constraints)

        mp_request = MultiPeriodRequest(
            daily_calorie_range=daily_cal_range,
            daily_nutrient_constraints=list(request.nutrient_constraints),
            meals=meal_configs,
            mode=request.mode,
            exclude_tags=list(request.exclude_tags),
            include_tags=list(request.include_tags),
            max_foods=request.max_foods,
            explicit_food_ids=request.explicit_food_ids,
        )

        if not json_output:
            console.print("[dim]Created default multi-period config (25/35/35/5% meal splits)[/dim]")

    # Override mode if --minimize-cost flag is set
    if minimize_cost:
        request.mode = "minimize_cost"
        if mp_request is not None:
            mp_request.mode = "minimize_cost"

    # Parse explicit food IDs if provided
    if foods:
        try:
            food_ids = [int(fid.strip()) for fid in foods.split(",")]
            request.explicit_food_ids = food_ids
            if not json_output:
                console.print(f"[dim]Using explicit food pool: {len(food_ids)} foods[/dim]")
        except ValueError:
            if json_output:
                output_json({
                    "success": False,
                    "command": "optimize",
                    "errors": [f"Invalid --foods format: {foods}"],
                    "suggestions": ["Use comma-separated numeric FDC IDs, e.g., --foods 175167,171287,172421"],
                })
            else:
                console.print(f"[red]Invalid --foods format: {foods}[/red]")
                console.print("[red]Use comma-separated numeric FDC IDs, e.g., --foods 175167,171287,172421[/red]")
            raise typer.Exit(1)

    # Handle --goal flag for body composition targets
    if goal:
        from mealplan.profiles.body_calc import calculate_targets, parse_goal_string
        from mealplan.optimizer.models import NutrientConstraint
        from mealplan.data.nutrient_ids import NUTRIENT_IDS

        try:
            parsed = parse_goal_string(goal)
            # If weight not in goal string, use default
            weight = parsed.get("weight_lbs") or 170.0
            target_weight = parsed.get("target_weight_lbs")

            targets = calculate_targets(
                age=35,  # Default age if not specified
                sex="male",  # Default sex if not specified
                height_inches=70,  # Default height if not specified
                weight_lbs=weight,
                goal=parsed["goal"],
                target_weight_lbs=target_weight,
                activity_level="moderate",
            )

            # Update request with calculated targets
            request.calorie_range = (targets.calories_min, targets.calories_max)

            # Add protein constraint
            protein_constraint = NutrientConstraint(
                nutrient_id=NUTRIENT_IDS["protein"],
                min_value=float(targets.protein_min),
                max_value=float(targets.protein_max),
            )
            request.nutrient_constraints = list(request.nutrient_constraints) + [protein_constraint]

            # Update mp_request if it was already created
            if mp_request is not None:
                mp_request.daily_calorie_range = (targets.calories_min, targets.calories_max)
                mp_request.daily_nutrient_constraints = list(request.nutrient_constraints)
                # Regenerate meal configs with new calorie range
                meal_configs = derive_default_meal_configs(
                    (targets.calories_min, targets.calories_max),
                    request.nutrient_constraints
                )
                mp_request.meals = meal_configs

            if not json_output:
                console.print(f"[dim]Goal '{parsed['goal']}': {targets.calories_min}-{targets.calories_max} kcal, "
                            f"{targets.protein_min}-{targets.protein_max}g protein[/dim]")
                if targets.projected_weeks:
                    console.print(f"[dim]Projected timeline: {targets.projected_weeks} weeks[/dim]")

        except (ValueError, KeyError) as e:
            if json_output:
                output_json({
                    "success": False,
                    "command": "optimize",
                    "errors": [f"Invalid --goal format: {goal}. Error: {e}"],
                    "suggestions": [
                        "Use format: goal:weight:target_weight",
                        "Example: --goal fat_loss:185lbs:165lbs",
                        "Available goals: fat_loss, aggressive_fat_loss, maintenance, lean_gain, muscle_gain",
                    ],
                })
            else:
                console.print(f"[red]Invalid --goal format: {goal}[/red]")
                console.print("[red]Use format: goal:weight:target_weight (e.g., fat_loss:185lbs:165lbs)[/red]")
            raise typer.Exit(1)

    # Run optimization with verbose output
    with db.get_connection() as conn:
        # Handle --pattern flag for dietary patterns
        if pattern and not foods:
            from mealplan.data.pattern_pools import build_food_pool

            try:
                pattern_food_ids = build_food_pool(
                    conn,
                    patterns=pattern,
                    max_foods=max_foods,
                    min_protein_per_100g=min_protein,
                    require_calorie_data=True,
                    balance_categories=True,
                    use_staples=not include_all_foods,
                )

                if not pattern_food_ids:
                    if json_output:
                        output_json({
                            "success": False,
                            "command": "optimize",
                            "errors": [f"No foods found matching patterns: {pattern}"],
                            "suggestions": ["Check pattern names with 'mealplan patterns list'"],
                        })
                    else:
                        console.print(f"[red]No foods found matching patterns: {pattern}[/red]")
                    raise typer.Exit(1)

                request.explicit_food_ids = pattern_food_ids
                if mp_request is not None:
                    mp_request.explicit_food_ids = pattern_food_ids

                if not json_output:
                    source = "staples" if not include_all_foods else "full USDA"
                    console.print(f"[dim]Pattern '{'+'.join(pattern)}' ({source}): {len(pattern_food_ids)} foods[/dim]")

            except ValueError as e:
                if json_output:
                    output_json({
                        "success": False,
                        "command": "optimize",
                        "errors": [str(e)],
                        "suggestions": ["Available patterns: pescatarian, vegetarian, vegan, keto, slow_carb, mediterranean, paleo, whole30"],
                    })
                else:
                    console.print(f"[red]{e}[/red]")
                raise typer.Exit(1)

        # Template-based optimization path (recommended for realistic meals)
        if template:
            from mealplan.templates.optimizer import run_template_optimization
            from mealplan.templates.models import SelectionStrategy

            if not pattern:
                if json_output:
                    output_json({
                        "success": False,
                        "command": "optimize",
                        "errors": ["--template requires --pattern to be specified"],
                        "suggestions": [
                            "Add --pattern pescatarian --pattern slow_carb",
                            "Available patterns: pescatarian, vegetarian, vegan, keto, slow_carb, mediterranean, paleo, whole30",
                        ],
                    })
                else:
                    console.print("[red]--template requires --pattern to be specified[/red]")
                raise typer.Exit(1)

            # Get calorie/protein targets from request
            cal_min, cal_max = request.calorie_range if request.calorie_range else (1800, 2200)
            prot_min = next((n.min_value for n in request.nutrient_constraints if n.nutrient_id == 1003 and n.min_value), 150)
            prot_max = next((n.max_value for n in request.nutrient_constraints if n.nutrient_id == 1003 and n.max_value), 200)

            if not json_output:
                console.print(f"[dim]Using template-based optimization for {'+'.join(pattern)}[/dim]")

            template_result = run_template_optimization(
                conn=conn,
                patterns=pattern,
                daily_calories=(cal_min, cal_max),
                daily_protein=(prot_min, prot_max),
                strategy=SelectionStrategy.RANDOM,
                seed=seed,
                max_retries=5,
            )

            # Format template output
            if json_output:
                if template_result.success:
                    response_data = {
                        "run_id": None,
                        "status": "optimal",
                        "profile": profile_name,
                        "template_mode": True,
                        "template_name": template_result.template_name,
                        "selection_attempts": template_result.selection_attempts,
                        "meals": {
                            meal.meal_type.value: {
                                "foods": [
                                    {
                                        "fdc_id": f.fdc_id,
                                        "description": f.description,
                                        "grams": float(round(f.grams, 1)),
                                        "calories": float(round(f.calories, 0)),
                                        "protein": float(round(f.protein, 1)),
                                        "slot": f.slot_name,
                                    }
                                    for f in meal.foods
                                ],
                                "totals": {
                                    "calories": float(round(meal.total_calories, 0)),
                                    "protein": float(round(meal.total_protein, 1)),
                                    "carbs": float(round(meal.total_carbs, 1)),
                                    "fat": float(round(meal.total_fat, 1)),
                                },
                            }
                            for meal in template_result.meals
                        },
                        "daily_totals": {
                            "calories": float(round(template_result.daily_calories, 0)),
                            "protein": float(round(template_result.daily_protein, 1)),
                            "carbs": float(round(template_result.daily_carbs, 1)),
                            "fat": float(round(template_result.daily_fat, 1)),
                        },
                        "human_summary": (
                            f"Template '{template_result.template_name}': "
                            f"{template_result.daily_calories:.0f} kcal, "
                            f"{template_result.daily_protein:.0f}g protein"
                        ),
                    }
                    output_json(create_response(
                        success=True,
                        command="optimize",
                        data=response_data,
                        warnings=[],
                        suggestions=[],
                        human_summary=response_data["human_summary"],
                    ).to_dict())
                else:
                    output_json(create_response(
                        success=False,
                        command="optimize",
                        data={},
                        errors=[template_result.message],
                        suggestions=["Try a different --pattern combination or relax constraints"],
                        human_summary=f"Failed: {template_result.message}",
                    ).to_dict())
                raise typer.Exit(0 if template_result.success else 1)
            else:
                # Rich table output for template results
                if template_result.success:
                    console.print(Panel(
                        f"Template: [bold]{template_result.template_name}[/bold]\n"
                        f"Selection attempts: {template_result.selection_attempts}",
                        title="Template-Based Meal Plan",
                    ))

                    for meal in template_result.meals:
                        table = Table(title=f"{meal.meal_type.value.capitalize()} ({meal.total_calories:.0f} kcal)")
                        table.add_column("Food", style="cyan")
                        table.add_column("Amount", justify="right")
                        table.add_column("Slot", style="dim")

                        for food in meal.foods:
                            # Truncate description if needed
                            desc = food.description[:45] + "..." if len(food.description) > 48 else food.description
                            table.add_row(desc, f"{food.grams:.0f}g", food.slot_name)

                        console.print(table)
                        console.print(f"[dim]P: {meal.total_protein:.0f}g | C: {meal.total_carbs:.0f}g | F: {meal.total_fat:.0f}g[/dim]\n")

                    console.print(Panel(
                        f"Calories: {template_result.daily_calories:.0f}\n"
                        f"Protein: {template_result.daily_protein:.0f}g\n"
                        f"Carbs: {template_result.daily_carbs:.0f}g\n"
                        f"Fat: {template_result.daily_fat:.0f}g",
                        title="Daily Totals",
                    ))
                else:
                    console.print(f"[red]Template optimization failed: {template_result.message}[/red]")

                raise typer.Exit(0 if template_result.success else 1)

        # Multi-period optimization path
        if use_multiperiod and mp_request is not None:
            # Apply explicit food IDs if provided
            if foods:
                try:
                    food_ids = [int(fid.strip()) for fid in foods.split(",")]
                    mp_request.explicit_food_ids = food_ids
                except ValueError:
                    pass  # Already handled above

            if not json_output:
                console.print(f"[dim]Running multi-period optimization with {len(mp_request.meals)} meals[/dim]")

            if json_output:
                mp_result = solve_multiperiod_diet(mp_request, conn, verbose=verbose)
            else:
                with console.status("[bold green]Optimizing (multi-period)..."):
                    mp_result = solve_multiperiod_diet(mp_request, conn, verbose=verbose)

            # Apply --practical consolidation if requested
            consolidated = None
            shopping = None
            if practical and mp_result.success:
                from mealplan.export.meal_consolidator import consolidate_result, consolidated_result_to_dict
                from mealplan.export.shopping_list import generate_shopping_list, shopping_list_to_dict

                consolidated = consolidate_result(mp_result, max_foods_per_meal=5)
                shopping = generate_shopping_list(consolidated, days=days if days > 1 else 7)

            # Format multi-period output
            if json_output:
                response_data = {
                    "run_id": None,  # TODO: save multi-period runs
                    "status": mp_result.status,
                    "profile": profile_name,
                    "multiperiod": True,
                    "practical_mode": practical,
                }

                if mp_result.success:
                    if consolidated:
                        # Use consolidated output
                        response_data["consolidated_meals"] = consolidated_result_to_dict(consolidated)["meals"]
                        response_data["shopping_list"] = shopping_list_to_dict(shopping)
                        response_data["daily_totals"] = {
                            k: float(round(v, 1)) for k, v in consolidated.daily_totals.items()
                        }
                        response_data["human_summary"] = (
                            f"{len(consolidated.meals)} meals (consolidated): "
                            f"{consolidated.daily_totals['calories']:.0f} kcal, "
                            f"{consolidated.daily_totals['protein']:.0f}g protein"
                        )
                    else:
                        # Use raw output
                        response_data["meals"] = {
                            meal.meal_type.value: {
                                "foods": [
                                    {
                                        "fdc_id": f.fdc_id,
                                        "description": f.description,
                                        "grams": float(round(f.grams, 1)),
                                        "calories": float(round(f.nutrients.get(1008, 0), 0)),
                                    }
                                    for f in meal.foods
                                ],
                                "totals": {
                                    "calories": float(round(meal.total_calories, 0)),
                                    "protein": float(round(meal.total_protein, 1)),
                                    "carbs": float(round(meal.total_carbs, 1)),
                                    "fat": float(round(meal.total_fat, 1)),
                                },
                            }
                            for meal in mp_result.meals
                        }
                        response_data["daily_totals"] = {
                            k: float(round(v, 1)) for k, v in mp_result.daily_totals.items()
                        }
                        response_data["human_summary"] = (
                            f"{len(mp_result.meals)} meals: "
                            f"{mp_result.daily_totals['calories']:.0f} kcal, "
                            f"{mp_result.daily_totals['protein']:.0f}g protein"
                        )
                else:
                    response_data["errors"] = [mp_result.message]
                    if mp_result.infeasibility_diagnosis:
                        response_data["diagnosis"] = {
                            "conflicting_constraints": mp_result.infeasibility_diagnosis.conflicting_constraints,
                            "suggested_relaxations": mp_result.infeasibility_diagnosis.suggested_relaxations,
                        }
                    response_data["human_summary"] = f"Failed: {mp_result.message}"

                output_json(create_response(
                    success=mp_result.success,
                    command="optimize",
                    data=response_data,
                    errors=[mp_result.message] if not mp_result.success else [],
                    suggestions=mp_result.infeasibility_diagnosis.suggested_relaxations if mp_result.infeasibility_diagnosis else [],
                    human_summary=response_data.get("human_summary", ""),
                ).to_dict())
            else:
                # Console output for multi-period
                # Panel and Table are imported at module level

                if mp_result.success:
                    if consolidated:
                        # Practical mode: show consolidated meals + shopping list
                        from mealplan.export.meal_consolidator import format_consolidated_result
                        from mealplan.export.shopping_list import format_shopping_list

                        console.print(Panel(
                            f"[bold green]Practical Meal Plan (Consolidated)[/bold green]\n"
                            f"Profile: {profile_name or 'default'}\n"
                            f"Meals: {len(consolidated.meals)} (max 5 foods each)",
                            title="Result"
                        ))

                        for meal in consolidated.meals:
                            table = Table(title=f"{meal.meal_name.title()} ({meal.total_calories:.0f} kcal)")
                            table.add_column("Food", style="cyan")
                            table.add_column("Amount", justify="right")

                            for food in meal.foods:
                                table.add_row(food.name, f"{food.grams:.0f}g")

                            console.print(table)
                            console.print(
                                f"[dim]P: {meal.total_protein:.0f}g | "
                                f"C: {meal.total_carbs:.0f}g | "
                                f"F: {meal.total_fat:.0f}g[/dim]\n"
                            )

                        console.print(Panel(
                            f"Calories: {consolidated.daily_totals['calories']:.0f}\n"
                            f"Protein: {consolidated.daily_totals['protein']:.0f}g\n"
                            f"Carbs: {consolidated.daily_totals['carbs']:.0f}g\n"
                            f"Fat: {consolidated.daily_totals['fat']:.0f}g",
                            title="Daily Totals"
                        ))

                        # Shopping list
                        console.print("\n")
                        console.print(format_shopping_list(shopping))
                    else:
                        # Standard output
                        console.print(Panel(
                            f"[bold green]Multi-Period Optimization Successful[/bold green]\n"
                            f"Profile: {profile_name or 'default'}\n"
                            f"Meals: {len(mp_result.meals)}",
                            title="Result"
                        ))

                        for meal in mp_result.meals:
                            table = Table(title=f"{meal.meal_type.value.title()} ({meal.total_calories:.0f} kcal)")
                            table.add_column("Food", style="cyan")
                            table.add_column("Grams", justify="right")

                            for food in meal.foods:
                                from mealplan.export.llm_prompt import clean_food_name
                                table.add_row(clean_food_name(food.description), f"{food.grams:.0f}g")

                            console.print(table)
                            console.print(
                                f"[dim]P: {meal.total_protein:.0f}g | "
                                f"C: {meal.total_carbs:.0f}g | "
                                f"F: {meal.total_fat:.0f}g[/dim]\n"
                            )

                        console.print(Panel(
                            f"Calories: {mp_result.daily_totals['calories']:.0f}\n"
                            f"Protein: {mp_result.daily_totals['protein']:.0f}g\n"
                            f"Carbs: {mp_result.daily_totals['carbs']:.0f}g\n"
                            f"Fat: {mp_result.daily_totals['fat']:.0f}g",
                            title="Daily Totals"
                        ))
                else:
                    console.print(f"[red]Optimization failed: {mp_result.message}[/red]")
                    if mp_result.infeasibility_diagnosis:
                        console.print("\n[yellow]Conflicting constraints:[/yellow]")
                        for c in mp_result.infeasibility_diagnosis.conflicting_constraints[:5]:
                            console.print(f"  - {c}")
                        console.print("\n[yellow]Suggestions:[/yellow]")
                        for s in mp_result.infeasibility_diagnosis.suggested_relaxations[:5]:
                            console.print(f"  - {s}")

            return  # Exit after multi-period handling

        # Standard single-period path
        # Build constraints first to get food count info
        from mealplan.optimizer.constraints import ConstraintBuilder

        builder = ConstraintBuilder(conn, request)
        constraint_data = builder.build()

        total_foods = constraint_data["total_eligible_foods"]
        used_foods = len(constraint_data["food_ids"])

        # Get data quality warnings
        data_quality_warnings = constraint_data.get("data_quality_warnings", [])

        if not json_output:
            if constraint_data["was_sampled"]:
                console.print(
                    f"[dim]Found {total_foods} eligible foods, randomly sampled {used_foods}[/dim]"
                )
            else:
                console.print(f"[dim]Found {used_foods} eligible foods[/dim]")

            # Warn if include_tags filter resulted in no foods
            if request.include_tags and total_foods == 0:
                tags_str = ", ".join(request.include_tags)
                console.print(
                    f"[yellow]Warning: No foods found with tags: {tags_str}[/yellow]"
                )
                console.print(
                    "[yellow]Use 'mealplan tags add <fdc_id> <tag>' to tag foods first.[/yellow]"
                )

            # Display data quality warnings
            if data_quality_warnings:
                console.print(f"[yellow]Data quality warnings ({len(data_quality_warnings)} foods):[/yellow]")
                for warning in data_quality_warnings[:5]:  # Limit to 5 for console
                    console.print(f"[yellow]  - {warning}[/yellow]")
                if len(data_quality_warnings) > 5:
                    console.print(f"[yellow]  ... and {len(data_quality_warnings) - 5} more[/yellow]")

        # Run optimization - use sparse solver if max_foods_in_solution is set
        if max_foods_in_solution is not None:
            from mealplan.optimizer.sparse_solver import solve_sparse_diet

            if json_output:
                result = solve_sparse_diet(request, conn, max_foods_in_solution, verbose=verbose)
            else:
                with console.status(f"[bold green]Optimizing (max {max_foods_in_solution} foods)..."):
                    result = solve_sparse_diet(request, conn, max_foods_in_solution, verbose=verbose)
        else:
            if json_output:
                result = solve_diet_problem(request, conn, verbose=verbose)
            else:
                with console.status("[bold green]Optimizing..."):
                    result = solve_diet_problem(request, conn, verbose=verbose)

        # If failed, run diagnosis
        diagnosis = None
        if not result.success:
            diagnosis = diagnose_infeasibility(conn, request)

    # Save to history if requested
    run_id: Optional[int] = None
    if save:
        from mealplan.export.formatters import JSONFormatter

        formatter = JSONFormatter()
        # Include request for constraints_used to enable what-if round-trip
        result_json = formatter.format(
            result,
            profile_name,
            request=request,
            data_quality_warnings=data_quality_warnings,
        )
        with db.get_connection() as conn:
            run_id = OptimizationRunQueries.save_run(
                conn,
                profile_id,
                result.status,
                result.total_cost,
                result_json,
            )

    # JSON output with agent-friendly envelope
    if json_output:
        response_data = {
            "run_id": run_id,
            "status": result.status,
            "profile": profile_name,
            "food_pool": {
                "total_eligible": total_foods,
                "used": used_foods,
                "was_sampled": constraint_data["was_sampled"],
            },
        }

        if result.success:
            response_data["solution"] = {
                "foods": [
                    {
                        "fdc_id": f.fdc_id,
                        "description": f.description,
                        "grams": round(f.grams, 1),
                        "cost": round(f.cost, 2),
                    }
                    for f in result.foods
                ],
                "nutrients": {
                    str(n.nutrient_id): {
                        "name": n.name,
                        "amount": round(n.amount, 2),
                        "unit": n.unit,
                        "min": n.min_constraint,
                        "max": n.max_constraint,
                        "satisfied": n.satisfied,
                    }
                    for n in result.nutrients
                },
                "total_cost": round(result.total_cost, 2) if result.total_cost else None,
            }

            # Add meal allocation if requested
            if allocate_meals:
                from mealplan.export.meal_allocator import allocate_to_meals, format_meal_allocation

                total_calories = sum(n.amount for n in result.nutrients if n.nutrient_id == 1008)
                if total_calories > 0:
                    meals = allocate_to_meals(result.foods, total_calories)
                    response_data["meal_allocation"] = format_meal_allocation(meals)

            # Add binding constraints info
            if result.kkt_analysis:
                binding = [
                    c for c in result.kkt_analysis.nutrient_constraints if c.is_binding
                ]
                response_data["binding_constraints"] = [
                    {
                        "type": c.constraint_type,
                        "name": c.name,
                        "bound": c.bound,
                        "value": round(c.value, 2),
                    }
                    for c in binding
                ]

                foods_at_limit = [
                    c for c in result.kkt_analysis.binding_food_bounds
                    if c.constraint_type == "food_upper"
                ]
                response_data["foods_at_limit"] = [
                    {"name": c.name, "grams": round(c.value, 1), "limit": round(c.bound, 1)}
                    for c in foods_at_limit[:10]
                ]

        # Add suggestions
        suggestions = []
        warnings = list(data_quality_warnings)  # Include data quality warnings

        if result.success:
            # Check for binding constraints
            if result.kkt_analysis:
                binding = [c for c in result.kkt_analysis.nutrient_constraints if c.is_binding]
                if binding:
                    for c in binding[:3]:
                        if "min" in c.constraint_type.lower():
                            suggestions.append(
                                f"{c.name} is at minimum - consider relaxing to allow more variety"
                            )
                        else:
                            suggestions.append(
                                f"{c.name} is at maximum - consider increasing limit"
                            )

            # Cost suggestion
            if result.total_cost and result.total_cost > 15:
                suggestions.append(
                    "Daily cost is high - try 'mealplan optimize --minimize-cost' or expand food pool"
                )

            human_summary = (
                f"Optimal solution: {len(result.foods)} foods, "
                f"{sum(n.amount for n in result.nutrients if n.nutrient_id == 1008):.0f} kcal"
            )
            if result.total_cost:
                human_summary += f", ${result.total_cost:.2f}/day"

        else:
            human_summary = f"Optimization failed: {result.message}"
            if diagnosis:
                response_data["diagnosis"] = diagnosis
                for conflict in diagnosis.get("likely_conflicts", []):
                    suggestions.extend(
                        [s.get("detail", s.get("action", "")) for s in conflict.get("suggestions", [])]
                    )
                for s in diagnosis.get("suggestions", []):
                    suggestions.append(s.get("detail", s.get("action", "")))

        output_json({
            "success": result.success,
            "command": "optimize",
            "data": response_data,
            "warnings": warnings,
            "suggestions": suggestions,
            "human_summary": human_summary,
        })
        return

    # Format output (non-JSON)
    formatted = format_result(result, output, profile_name, run_id, console)
    if formatted:
        console.print(formatted)

    # Display KKT analysis if verbose mode is enabled
    if verbose and result.success and result.kkt_analysis:
        from mealplan.export.formatters import KKTFormatter

        console.print()  # Blank line before KKT section
        kkt_formatter = KKTFormatter(console)
        kkt_formatter.format(result.kkt_analysis)

    # Display meal allocation if requested
    if allocate_meals and result.success and result.foods:
        from mealplan.export.meal_allocator import allocate_to_meals, format_meal_allocation_text

        total_calories = sum(n.amount for n in result.nutrients if n.nutrient_id == 1008)
        if total_calories > 0:
            console.print()
            meals = allocate_to_meals(result.foods, total_calories)
            console.print(format_meal_allocation_text(meals))

    # Show diagnosis if failed
    if not result.success and diagnosis:
        console.print()
        console.print(Panel("[bold red]Infeasibility Diagnosis[/bold red]"))

        if diagnosis.get("food_pool_issues"):
            fpi = diagnosis["food_pool_issues"]
            console.print(f"[yellow]Food pool: {fpi.get('problem', 'Issue detected')}[/yellow]")
            for cause in fpi.get("causes", []):
                console.print(f"  - {cause}")

        for conflict in diagnosis.get("likely_conflicts", []):
            console.print(f"\n[yellow]Conflict:[/yellow] {' AND '.join(conflict['constraints'])}")
            console.print(f"  {conflict['explanation']}")
            for s in conflict.get("suggestions", []):
                console.print(f"  → Try: {s.get('detail', s.get('command', ''))}")

        for s in diagnosis.get("suggestions", []):
            console.print(f"\n[cyan]Suggestion:[/cyan] {s.get('detail', s.get('action', ''))}")


@app.command("optimize-batch")
def optimize_batch_cmd(
    pools_file: Path = typer.Argument(..., help="JSON file with batch pools and constraints"),
    json_output: bool = typer.Option(True, "--json/--no-json", help="Output as JSON"),
) -> None:
    """Run batch optimization across multiple food pools.

    Input JSON format:
    {
        "base_constraints": {
            "calories": [1700, 1900],
            "nutrients": {"protein": {"min": 150}, "fiber": {"min": 30}}
        },
        "pools": [
            {"name": "pool_a", "foods": [175167, 171287, 172421]},
            {"name": "pool_b", "foods": [171955, 175139, 173757]}
        ]
    }
    """
    require_food_data()

    import json as json_module

    from mealplan.optimizer.batch import (
        compare_batch_results,
        format_batch_response,
        parse_batch_request_json,
        run_batch_optimization,
    )

    if not pools_file.exists():
        if json_output:
            output_json({
                "success": False,
                "command": "optimize-batch",
                "errors": [f"Pools file not found: {pools_file}"],
                "suggestions": ["Check the file path"],
            })
        else:
            console.print(f"[red]Pools file not found: {pools_file}[/red]")
        raise typer.Exit(1)

    try:
        with open(pools_file) as f:
            data = json_module.load(f)
    except json_module.JSONDecodeError as e:
        if json_output:
            output_json({
                "success": False,
                "command": "optimize-batch",
                "errors": [f"Invalid JSON in pools file: {e}"],
                "suggestions": ["Validate the JSON format"],
            })
        else:
            console.print(f"[red]Invalid JSON in pools file: {e}[/red]")
        raise typer.Exit(1)

    try:
        batch_request = parse_batch_request_json(data)
    except Exception as e:
        if json_output:
            output_json({
                "success": False,
                "command": "optimize-batch",
                "errors": [f"Failed to parse batch request: {e}"],
                "suggestions": ["Check the JSON format matches the expected schema"],
            })
        else:
            console.print(f"[red]Failed to parse batch request: {e}[/red]")
        raise typer.Exit(1)

    if not batch_request.pools:
        if json_output:
            output_json({
                "success": False,
                "command": "optimize-batch",
                "errors": ["No pools specified in batch request"],
                "suggestions": ["Add at least one pool with food IDs"],
            })
        else:
            console.print("[red]No pools specified in batch request[/red]")
        raise typer.Exit(1)

    db = get_db()
    with db.get_connection() as conn:
        if not json_output:
            console.print(f"[dim]Running batch optimization across {len(batch_request.pools)} pools...[/dim]")

        results = run_batch_optimization(conn, batch_request, save_runs=False)
        comparison = compare_batch_results(results)

    if json_output:
        response_data = format_batch_response(results, comparison)
        successful = sum(1 for r in results if r.success)

        output_json({
            "success": successful > 0,
            "command": "optimize-batch",
            "data": response_data,
            "warnings": [],
            "suggestions": [],
            "human_summary": f"{successful}/{len(results)} pools feasible. Best: {comparison.get('best', 'none')}",
        })
    else:
        # Table output
        table = Table(title="Batch Optimization Results")
        table.add_column("Pool", style="cyan")
        table.add_column("Status")
        table.add_column("Calories", justify="right")
        table.add_column("Protein", justify="right")
        table.add_column("Cost", justify="right", style="green")
        table.add_column("Foods", justify="right")

        for r in results:
            if r.success and r.summary:
                status = "[green]optimal[/green]"
                calories = f"{r.summary.get('total_calories', 0):.0f}"
                protein = f"{r.summary.get('total_protein', 0):.1f}g"
                cost = f"${r.summary.get('total_cost', 0):.2f}" if r.summary.get('total_cost') else "-"
                foods = str(r.summary.get('food_count', 0))
            else:
                status = f"[red]{r.status}[/red]"
                calories = protein = cost = foods = "-"

            table.add_row(r.pool_name, status, calories, protein, cost, foods)

        console.print(table)

        if comparison.get("best"):
            console.print(f"\n[green]Best pool: {comparison['best']}[/green]")


@app.command("export-for-llm")
def export_for_llm(
    run_id: str = typer.Argument(..., help="Run ID or 'latest'"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    days: int = typer.Option(1, "--days", "-d", help="Number of days for meal plan"),
) -> None:
    """Export optimization result as LLM-ready prompt."""
    require_food_data()

    from mealplan.export.llm_prompt import LLMPromptGenerator
    from mealplan.optimizer.models import FoodResult, NutrientResult, OptimizationResult

    db = get_db()

    # Get the run
    with db.get_connection() as conn:
        if run_id.lower() == "latest":
            run = OptimizationRunQueries.get_latest_run(conn)
        else:
            run = OptimizationRunQueries.get_run_by_id(conn, int(run_id))

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    # Parse the stored result
    result_data = json.loads(run["result_json"])

    # Fetch per-food nutrient data from database
    # Nutrient IDs: 1008=Kcal, 1003=Protein, 1005=Carbs, 1004=Fat
    macro_nutrient_ids = [1008, 1003, 1005, 1004]
    food_nutrients_map: dict[int, dict[int, float]] = {}

    with db.get_connection() as conn:
        fdc_ids = [f["fdc_id"] for f in result_data["solution"]["foods"]]
        if fdc_ids:
            placeholders = ",".join("?" * len(fdc_ids))
            nutrient_placeholders = ",".join("?" * len(macro_nutrient_ids))
            cursor = conn.execute(
                f"""
                SELECT fdc_id, nutrient_id, amount / 100.0 as amount_per_gram
                FROM food_nutrients
                WHERE fdc_id IN ({placeholders})
                  AND nutrient_id IN ({nutrient_placeholders})
                """,
                fdc_ids + macro_nutrient_ids,
            )
            for row in cursor.fetchall():
                fdc_id, nutrient_id, amount_per_gram = row
                if fdc_id not in food_nutrients_map:
                    food_nutrients_map[fdc_id] = {}
                food_nutrients_map[fdc_id][nutrient_id] = amount_per_gram

    # Reconstruct OptimizationResult from JSON
    foods = []
    for f in result_data["solution"]["foods"]:
        fdc_id = f["fdc_id"]
        grams = f["grams"]
        # Calculate nutrient amounts based on grams
        per_gram = food_nutrients_map.get(fdc_id, {})
        nutrients = {nid: per_gram.get(nid, 0.0) * grams for nid in macro_nutrient_ids}
        foods.append(
            FoodResult(
                fdc_id=fdc_id,
                description=f["description"],
                grams=grams,
                cost=f["cost"],
                nutrients=nutrients,
            )
        )

    nutrients = [
        NutrientResult(
            nutrient_id=int(nid),
            name=n["name"],
            unit=n["unit"],
            amount=n["amount"],
            min_constraint=n.get("min"),
            max_constraint=n.get("max"),
            satisfied=n.get("satisfied", True),
        )
        for nid, n in result_data["solution"]["nutrients"].items()
    ]

    result = OptimizationResult(
        success=result_data["success"],
        status=result_data["status"],
        message=result_data["message"],
        foods=foods,
        total_cost=result_data["solution"]["total_cost"],
        nutrients=nutrients,
        solver_info=result_data.get("solver_info", {}),
    )

    # Generate prompt
    generator = LLMPromptGenerator()
    prompt = generator.generate(result, days, result_data.get("profile"))

    if output:
        output.write_text(prompt)
        console.print(f"[green]Prompt saved to: {output}[/green]")
    else:
        console.print(prompt)


@app.command()
def history(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of runs to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show optimization run history."""
    require_food_data()
    db = get_db()
    with db.get_connection() as conn:
        runs = OptimizationRunQueries.list_runs(conn, limit)

    if json_output:
        output_json({
            "success": True,
            "command": "history",
            "data": {
                "runs": [
                    {
                        "run_id": run["run_id"],
                        "date": str(run["run_date"]),
                        "profile": run["profile_name"],
                        "status": run["status"],
                        "total_cost": run["total_cost"],
                    }
                    for run in runs
                ],
            },
            "human_summary": f"{len(runs)} optimization runs",
        })
        return

    if not runs:
        console.print("[yellow]No optimization runs found[/yellow]")
        return

    table = Table(title="Optimization History")
    table.add_column("Run ID", style="cyan")
    table.add_column("Date")
    table.add_column("Profile")
    table.add_column("Status")
    table.add_column("Cost", justify="right")

    for run in runs:
        status_style = "green" if run["status"] == "optimal" else "red"
        cost = f"${run['total_cost']:.2f}" if run["total_cost"] else "-"
        table.add_row(
            str(run["run_id"]),
            str(run["run_date"])[:19],
            run["profile_name"],
            f"[{status_style}]{run['status']}[/{status_style}]",
            cost,
        )

    console.print(table)


# ============================================================================
# Schema Subcommands (for LLM agents)
# ============================================================================


@schema_app.command("constraints")
def schema_constraints() -> None:
    """Show constraint schema for LLM agents."""
    from mealplan.agent.schema import get_constraint_schema
    output_json(get_constraint_schema())


@schema_app.command("nutrients")
def schema_nutrients() -> None:
    """List all available nutrients with metadata."""
    from mealplan.agent.schema import get_nutrient_list
    output_json(get_nutrient_list())


@schema_app.command("tags")
def schema_tags() -> None:
    """List all tags in the database."""
    from mealplan.agent.schema import get_tag_list
    db = get_db()
    with db.get_connection() as conn:
        output_json(get_tag_list(conn))


@schema_app.command("foods")
def schema_foods() -> None:
    """Show food filter schema."""
    from mealplan.agent.schema import get_food_filter_schema
    output_json(get_food_filter_schema())


@schema_app.command("all")
def schema_all() -> None:
    """Export complete schema documentation."""
    from mealplan.agent.schema import get_full_schema
    db = get_db()
    with db.get_connection() as conn:
        output_json(get_full_schema(conn))


# ============================================================================
# Explore Subcommands
# ============================================================================


@explore_app.command("foods")
def explore_foods_cmd(
    query: Optional[str] = typer.Argument(None, help="Text search query"),
    min_protein: Optional[float] = typer.Option(None, "--min-protein", help="Min protein per 100g"),
    max_protein: Optional[float] = typer.Option(None, "--max-protein", help="Max protein per 100g"),
    min_calories: Optional[float] = typer.Option(None, "--min-calories", help="Min calories per 100g"),
    max_calories: Optional[float] = typer.Option(None, "--max-calories", help="Max calories per 100g"),
    min_cost: Optional[float] = typer.Option(None, "--min-cost", help="Min cost per 100g"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Max cost per 100g"),
    has_tag: Optional[str] = typer.Option(None, "--tag", help="Filter to foods with this tag"),
    has_price: Optional[bool] = typer.Option(None, "--has-price/--no-price", help="Filter by price availability"),
    category: Optional[str] = typer.Option(
        None, "--category", help="Filter by food category (protein, fat, carb, legume, vegetable, fruit, mixed)"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search and filter foods with nutrient criteria."""
    from mealplan.explore.foods import FoodFilter, explore_foods

    filters = FoodFilter(
        query=query,
        min_protein=min_protein,
        max_protein=max_protein,
        min_calories=min_calories,
        max_calories=max_calories,
        min_cost=min_cost,
        max_cost=max_cost,
        has_tag=has_tag,
        has_price=has_price,
        category=category,
        limit=limit,
    )

    db = get_db()
    with db.get_connection() as conn:
        results = explore_foods(conn, filters)

    if json_output:
        output_json({
            "success": True,
            "command": "explore foods",
            "data": {
                "filters": {
                    "query": query,
                    "min_protein": min_protein,
                    "max_protein": max_protein,
                    "min_calories": min_calories,
                    "max_calories": max_calories,
                    "min_cost": min_cost,
                    "max_cost": max_cost,
                    "has_tag": has_tag,
                    "has_price": has_price,
                },
                "results": results,
                "total_matches": len(results),
            },
            "human_summary": f"Found {len(results)} foods matching filters",
        })
        return

    if not results:
        console.print("[yellow]No foods match the specified filters[/yellow]")
        return

    table = Table(title="Foods")
    table.add_column("FDC ID", style="cyan")
    table.add_column("Description")
    table.add_column("Protein", justify="right")
    table.add_column("Calories", justify="right")
    table.add_column("Price", justify="right", style="green")
    table.add_column("Tags", style="blue")

    for food in results:
        protein = f"{food['protein_per_100g']:.1f}g" if food["protein_per_100g"] else "-"
        calories = f"{food['calories_per_100g']:.0f}" if food["calories_per_100g"] else "-"
        price = f"${food['price_per_100g']:.2f}" if food["price_per_100g"] else "-"
        tags = ", ".join(food["tags"]) if food["tags"] else ""

        table.add_row(
            str(food["fdc_id"]),
            food["description"][:50],
            protein,
            calories,
            price,
            tags,
        )

    console.print(table)
    console.print(f"[dim]Showing {len(results)} results[/dim]")


@explore_app.command("high-nutrient")
def explore_high_nutrient(
    nutrient: str = typer.Argument(..., help="Nutrient name (e.g., protein, iron, vitamin_d)"),
    min_amount: Optional[float] = typer.Option(None, "--min", help="Minimum amount per 100g"),
    max_amount: Optional[float] = typer.Option(None, "--max", help="Maximum amount per 100g"),
    has_tag: Optional[str] = typer.Option(None, "--tag", help="Filter to foods with this tag"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Find foods high in a specific nutrient."""
    from mealplan.explore.foods import find_high_nutrient_foods
    from mealplan.data.nutrient_ids import NUTRIENT_IDS, NUTRIENT_UNITS

    # Validate nutrient name
    nutrient_lower = nutrient.lower().replace(" ", "_").replace("-", "_")
    if nutrient_lower not in NUTRIENT_IDS:
        if json_output:
            output_json({
                "success": False,
                "command": "explore high-nutrient",
                "errors": [f"Unknown nutrient: {nutrient}"],
                "suggestions": [f"Available nutrients: {', '.join(sorted(NUTRIENT_IDS.keys())[:10])}..."],
            })
        else:
            console.print(f"[red]Unknown nutrient: {nutrient}[/red]")
            console.print(f"[dim]Available: {', '.join(sorted(NUTRIENT_IDS.keys())[:10])}...[/dim]")
        raise typer.Exit(1)

    db = get_db()
    with db.get_connection() as conn:
        results = find_high_nutrient_foods(
            conn, nutrient, min_amount, max_amount, has_tag, limit
        )

    unit = NUTRIENT_UNITS.get(NUTRIENT_IDS[nutrient_lower], "")

    if json_output:
        output_json({
            "success": True,
            "command": "explore high-nutrient",
            "data": {
                "nutrient": nutrient,
                "unit": unit,
                "filters": {"min": min_amount, "max": max_amount, "tag": has_tag},
                "results": results,
            },
            "human_summary": f"Found {len(results)} foods ranked by {nutrient} content",
        })
        return

    if not results:
        console.print(f"[yellow]No foods found with {nutrient} content[/yellow]")
        return

    table = Table(title=f"Foods High in {nutrient.replace('_', ' ').title()}")
    table.add_column("FDC ID", style="cyan")
    table.add_column("Description")
    table.add_column(f"{nutrient.title()}/{unit}", justify="right", style="green")
    table.add_column("Calories", justify="right")
    table.add_column("Price", justify="right")

    for food in results:
        calories = f"{food['calories_per_100g']:.0f}" if food["calories_per_100g"] else "-"
        price = f"${food['price_per_100g']:.2f}" if food["price_per_100g"] else "-"

        table.add_row(
            str(food["fdc_id"]),
            food["description"][:45],
            f"{food['amount_per_100g']:.1f}",
            calories,
            price,
        )

    console.print(table)


@explore_app.command("compare")
def explore_compare(
    fdc_ids: list[int] = typer.Argument(..., help="FDC IDs to compare"),
    nutrients: Optional[str] = typer.Option(
        None, "--nutrients", help="Comma-separated nutrient names to compare"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Compare nutrient content of multiple foods side-by-side."""
    from mealplan.explore.foods import compare_foods

    nutrient_list = None
    if nutrients:
        nutrient_list = [n.strip() for n in nutrients.split(",")]

    db = get_db()
    with db.get_connection() as conn:
        result = compare_foods(conn, fdc_ids, nutrient_list)

    if json_output:
        output_json({
            "success": True,
            "command": "explore compare",
            "data": result,
            "human_summary": f"Comparing {len(result['foods'])} foods across {len(result['comparison'])} nutrients",
        })
        return

    if not result["foods"]:
        console.print("[yellow]No foods found[/yellow]")
        return

    # Build comparison table
    table = Table(title="Food Comparison (per 100g)")
    table.add_column("Nutrient")
    for food in result["foods"]:
        table.add_column(food["description"][:25], justify="right")

    for row in result["comparison"]:
        values = [
            f"{row['values'].get(f['fdc_id'], '-'):.1f}" if row['values'].get(f['fdc_id']) else "-"
            for f in result["foods"]
        ]
        table.add_row(f"{row['name']} ({row.get('unit', '')})", *values)

    console.print(table)


@explore_app.command("whatif")
def explore_whatif(
    base_run: str = typer.Option("latest", "--base", help="Base run ID or 'latest'"),
    add: Optional[list[str]] = typer.Option(None, "--add", help="Add constraint (e.g., 'protein:min:180')"),
    remove: Optional[list[str]] = typer.Option(None, "--remove", help="Remove nutrient constraint"),
    calories_min: Optional[float] = typer.Option(None, "--cal-min", help="New calorie minimum"),
    calories_max: Optional[float] = typer.Option(None, "--cal-max", help="New calorie maximum"),
    exclude_food: Optional[list[int]] = typer.Option(None, "--exclude-food", help="Exclude food by FDC ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include KKT analysis"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run what-if analysis by modifying constraints from a baseline."""
    from mealplan.explore.whatif import run_whatif_analysis

    base_id = -1 if base_run.lower() == "latest" else int(base_run)

    new_calorie_range = None
    if calories_min is not None or calories_max is not None:
        # Need to get current range and modify
        new_calorie_range = (
            calories_min if calories_min is not None else 1800,
            calories_max if calories_max is not None else 2200,
        )

    db = get_db()
    with db.get_connection() as conn:
        result = run_whatif_analysis(
            conn,
            base_id,
            add_constraints=add,
            remove_constraints=remove,
            new_calorie_range=new_calorie_range,
            remove_foods=exclude_food,
            verbose=verbose,
        )

    if json_output:
        output_json({
            "success": result["success"],
            "command": "explore whatif",
            "data": result,
            "human_summary": (
                f"What-if analysis: {result['result']['status']}"
                if result["success"] else result.get("error", "Analysis failed")
            ),
        })
        return

    if not result["success"]:
        console.print(f"[red]{result.get('error', 'What-if analysis failed')}[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]What-If Analysis[/bold] (base run: {result['base_run_id']})"))

    # Show modifications
    mods = result["modifications"]
    if mods.get("add_constraints"):
        console.print(f"[cyan]Added:[/cyan] {', '.join(mods['add_constraints'])}")
    if mods.get("remove_constraints"):
        console.print(f"[cyan]Removed:[/cyan] {', '.join(mods['remove_constraints'])}")
    if mods.get("new_calorie_range"):
        console.print(f"[cyan]Calories:[/cyan] {mods['new_calorie_range']}")
    if mods.get("remove_foods"):
        console.print(f"[cyan]Excluded foods:[/cyan] {mods['remove_foods']}")

    console.print()

    res = result["result"]
    if res["success"]:
        console.print(f"[green]Status: {res['status']}[/green]")

        # Show foods
        table = Table(title="Modified Solution")
        table.add_column("Food")
        table.add_column("Grams", justify="right")
        table.add_column("Cost", justify="right")

        for f in res["foods"]:
            table.add_row(
                f["description"][:40],
                f"{f['grams']:.1f}",
                f"${f['cost']:.2f}",
            )

        console.print(table)

        # Show comparison
        comp = result.get("comparison", {})
        if comp.get("foods_added") or comp.get("foods_removed") or comp.get("cost_change"):
            console.print("\n[bold]Changes from baseline:[/bold]")
            if comp.get("foods_added"):
                console.print(f"  [green]+ Added {len(comp['foods_added'])} foods[/green]")
            if comp.get("foods_removed"):
                console.print(f"  [red]- Removed {len(comp['foods_removed'])} foods[/red]")
            if comp.get("cost_change") is not None:
                sign = "+" if comp["cost_change"] > 0 else ""
                color = "red" if comp["cost_change"] > 0 else "green"
                console.print(f"  [{color}]Cost: {sign}${comp['cost_change']:.2f}[/{color}]")
    else:
        console.print(f"[red]Status: {res['status']}[/red]")
        console.print(f"[red]{res['message']}[/red]")


@explore_app.command("suggest-pools")
def explore_suggest_pools(
    strategies: Optional[str] = typer.Option(
        None, "--strategies", "-s", help="Comma-separated strategies: balanced,high_protein,budget"
    ),
    target_size: int = typer.Option(30, "--size", help="Target number of foods per pool"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter to foods with this tag"),
    require_price: bool = typer.Option(False, "--require-price", help="Only include foods with prices"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Suggest food pools for optimization using different strategies.

    Strategies:
    - balanced: Proportional selection across macro categories
    - high_protein: Foods with >= 20g protein per 100g
    - budget: Foods with price <= $1/100g

    Examples:
        mealplan explore suggest-pools --json
        mealplan explore suggest-pools --strategies balanced,high_protein --tag staple
    """
    from mealplan.explore.suggest import format_pool_suggestions, suggest_pools

    db = get_db()

    # Parse strategies
    strategy_list = None
    if strategies:
        strategy_list = [s.strip() for s in strategies.split(",")]

    with db.get_connection() as conn:
        suggestions = suggest_pools(
            conn,
            strategies=strategy_list,
            target_size=target_size,
            tag=tag,
            require_price=require_price,
        )

    if json_output:
        response = format_pool_suggestions(suggestions)
        output_json({
            "success": True,
            "command": "explore suggest-pools",
            "data": response,
            "human_summary": f"Generated {len(suggestions)} pool suggestions",
        })
        return

    if not suggestions:
        console.print("[yellow]No pool suggestions generated. Check your filters.[/yellow]")
        return

    for suggestion in suggestions:
        console.print(Panel(
            f"[bold]{suggestion.name}[/bold]\n\n"
            f"Strategy: {suggestion.strategy}\n"
            f"Foods: {len(suggestion.food_ids)}\n"
            f"{suggestion.description}",
            title=f"Pool: {suggestion.name}",
        ))

        # Show food IDs in a format easy to copy for --foods flag
        if suggestion.food_ids:
            ids_str = ",".join(str(fid) for fid in suggestion.food_ids[:10])
            if len(suggestion.food_ids) > 10:
                ids_str += f"... (+{len(suggestion.food_ids) - 10} more)"
            console.print(f"[dim]IDs: {ids_str}[/dim]")
        console.print()


@explore_app.command("runs")
def explore_runs(
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum runs to show"),
    profile_filter: Optional[str] = typer.Option(None, "--profile", "-p", help="Filter by profile name"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List recent optimization runs.

    Examples:
        mealplan explore runs
        mealplan explore runs --limit 5 --profile cutting
    """
    from mealplan.explore.compare import format_run_list, list_runs

    db = get_db()

    with db.get_connection() as conn:
        runs = list_runs(conn, limit=limit, profile=profile_filter)

    if json_output:
        response = format_run_list(runs)
        output_json({
            "success": True,
            "command": "explore runs",
            "data": response,
            "human_summary": f"Listed {len(runs)} optimization runs",
        })
        return

    if not runs:
        console.print("[yellow]No optimization runs found.[/yellow]")
        return

    table = Table(title="Recent Optimization Runs")
    table.add_column("ID", style="cyan")
    table.add_column("Timestamp")
    table.add_column("Status")
    table.add_column("Profile")
    table.add_column("Foods", justify="right")
    table.add_column("Calories", justify="right")
    table.add_column("Protein", justify="right")
    table.add_column("Cost", justify="right")

    for run in runs:
        status_color = "green" if run.success else "red"
        cost_str = f"${run.total_cost:.2f}" if run.total_cost else "-"
        table.add_row(
            str(run.run_id),
            run.timestamp[:16],  # Truncate for display
            f"[{status_color}]{run.status}[/{status_color}]",
            run.profile or "-",
            str(run.food_count),
            f"{run.calories:.0f}",
            f"{run.protein:.1f}g",
            cost_str,
        )

    console.print(table)


@explore_app.command("compare-runs")
def explore_compare_runs(
    run_a: int = typer.Argument(..., help="First run ID"),
    run_b: int = typer.Argument(..., help="Second run ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Compare two optimization runs.

    Shows differences in cost, nutrients, and food selections between runs.

    Examples:
        mealplan explore compare-runs 5 8
        mealplan explore compare-runs 5 8 --json
    """
    from mealplan.explore.compare import compare_runs, format_run_comparison

    db = get_db()

    with db.get_connection() as conn:
        comparison = compare_runs(conn, run_a, run_b)

    if comparison is None:
        if json_output:
            output_json({
                "success": False,
                "command": "explore compare-runs",
                "errors": [f"Could not find run {run_a} or {run_b}"],
            })
        else:
            console.print(f"[red]Could not find run {run_a} or {run_b}[/red]")
        raise typer.Exit(1)

    if json_output:
        response = format_run_comparison(comparison)
        output_json({
            "success": True,
            "command": "explore compare-runs",
            "data": response,
            "human_summary": f"Compared run {run_a} vs {run_b}",
        })
        return

    # Display comparison
    console.print(Panel(f"[bold]Run Comparison: #{run_a} vs #{run_b}[/bold]"))

    # Summary table
    table = Table(title="Summary")
    table.add_column("Metric")
    table.add_column(f"Run {run_a}", justify="right")
    table.add_column(f"Run {run_b}", justify="right")
    table.add_column("Difference", justify="right")

    # Calories
    cal_diff = comparison.calorie_difference
    cal_color = "green" if abs(cal_diff) < 100 else ("yellow" if abs(cal_diff) < 200 else "red")
    cal_sign = "+" if cal_diff > 0 else ""
    table.add_row(
        "Calories",
        f"{comparison.run_a.calories:.0f}",
        f"{comparison.run_b.calories:.0f}",
        f"[{cal_color}]{cal_sign}{cal_diff:.0f}[/{cal_color}]",
    )

    # Protein
    prot_diff = comparison.protein_difference
    prot_sign = "+" if prot_diff > 0 else ""
    table.add_row(
        "Protein",
        f"{comparison.run_a.protein:.1f}g",
        f"{comparison.run_b.protein:.1f}g",
        f"{prot_sign}{prot_diff:.1f}g",
    )

    # Cost
    if comparison.cost_difference is not None:
        cost_diff = comparison.cost_difference
        cost_sign = "+" if cost_diff > 0 else ""
        cost_color = "red" if cost_diff > 0 else "green"
        table.add_row(
            "Cost",
            f"${comparison.run_a.total_cost:.2f}",
            f"${comparison.run_b.total_cost:.2f}",
            f"[{cost_color}]{cost_sign}${cost_diff:.2f}[/{cost_color}]",
        )
    else:
        table.add_row("Cost", "-", "-", "-")

    # Food count
    food_diff = comparison.run_b.food_count - comparison.run_a.food_count
    food_sign = "+" if food_diff > 0 else ""
    table.add_row(
        "Foods",
        str(comparison.run_a.food_count),
        str(comparison.run_b.food_count),
        f"{food_sign}{food_diff}",
    )

    console.print(table)

    # Food differences
    if comparison.foods_only_in_a:
        console.print(f"\n[red]Foods only in run {run_a}:[/red]")
        for f in comparison.foods_only_in_a[:5]:
            console.print(f"  - {f[:50]}")
        if len(comparison.foods_only_in_a) > 5:
            console.print(f"  ... and {len(comparison.foods_only_in_a) - 5} more")

    if comparison.foods_only_in_b:
        console.print(f"\n[green]Foods only in run {run_b}:[/green]")
        for f in comparison.foods_only_in_b[:5]:
            console.print(f"  + {f[:50]}")
        if len(comparison.foods_only_in_b) > 5:
            console.print(f"  ... and {len(comparison.foods_only_in_b) - 5} more")

    console.print(f"\n[dim]Foods in common: {len(comparison.foods_in_both)}[/dim]")


# ============================================================================
# Prices Subcommands
# ============================================================================


@prices_app.command("add")
def prices_add(
    fdc_id: int = typer.Argument(..., help="FDC ID of food"),
    price_per_100g: float = typer.Argument(..., help="Price per 100 grams"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Price source"),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Notes"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Add or update price for a food."""
    db = get_db()
    with db.get_connection() as conn:
        # Verify food exists
        food = FoodQueries.get_food_by_id(conn, fdc_id)
        if not food:
            if json_output:
                output_json({
                    "success": False,
                    "command": "prices add",
                    "errors": [f"Food with FDC ID {fdc_id} not found"],
                })
            else:
                console.print(f"[red]Food with FDC ID {fdc_id} not found[/red]")
            raise typer.Exit(1)

        PriceQueries.upsert_price(conn, fdc_id, price_per_100g, source, notes)

    if json_output:
        output_json({
            "success": True,
            "command": "prices add",
            "data": {
                "fdc_id": fdc_id,
                "description": food["description"],
                "price_per_100g": price_per_100g,
                "source": source,
            },
            "human_summary": f"Price set: ${price_per_100g:.2f}/100g for {food['description'][:40]}",
        })
    else:
        console.print(
            f"[green]Price set for '{food['description'][:50]}': "
            f"${price_per_100g:.2f}/100g[/green]"
        )


@prices_app.command("import")
def prices_import(
    csv_path: Path = typer.Argument(..., help="Path to prices CSV file"),
) -> None:
    """Import prices from CSV file."""
    from mealplan.data.price_loader import PriceLoader

    if not csv_path.exists():
        console.print(f"[red]File not found: {csv_path}[/red]")
        raise typer.Exit(1)

    db = get_db()
    with db.get_connection() as conn:
        loader = PriceLoader(conn)
        counts = loader.load_from_csv(csv_path)

    console.print(f"[green]Imported {counts['loaded']} prices[/green]")
    if counts["skipped_invalid_id"]:
        console.print(
            f"[yellow]Skipped {counts['skipped_invalid_id']} "
            f"rows with invalid FDC IDs[/yellow]"
        )
    if counts["skipped_missing_price"]:
        console.print(
            f"[yellow]Skipped {counts['skipped_missing_price']} "
            f"rows with missing prices[/yellow]"
        )


@prices_app.command("list")
def prices_list(
    missing: bool = typer.Option(
        False, "--missing", "-m", help="Show foods without prices"
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List foods with prices or missing prices."""
    db = get_db()
    with db.get_connection() as conn:
        if missing:
            results = PriceQueries.get_foods_without_prices(conn)[:limit]
            title = "Foods Without Prices"
        else:
            results = PriceQueries.get_foods_with_prices(conn)[:limit]
            title = "Foods With Prices"

    if json_output:
        data = []
        for row in results:
            item = {"fdc_id": row["fdc_id"], "description": row["description"]}
            if not missing:
                item["price_per_100g"] = row["price_per_100g"]
                item["source"] = row["price_source"]
            data.append(item)

        output_json({
            "success": True,
            "command": "prices list",
            "data": {"foods": data, "showing_missing": missing},
            "human_summary": f"{len(results)} foods {'without' if missing else 'with'} prices",
        })
        return

    if not results:
        if missing:
            console.print("[green]All foods have prices![/green]")
        else:
            console.print("[yellow]No prices set yet[/yellow]")
        return

    table = Table(title=title)
    table.add_column("FDC ID", style="cyan")
    table.add_column("Description")
    if not missing:
        table.add_column("Price/100g", justify="right", style="green")
        table.add_column("Source", style="dim")

    for row in results:
        if missing:
            table.add_row(str(row["fdc_id"]), row["description"])
        else:
            table.add_row(
                str(row["fdc_id"]),
                row["description"][:50],
                f"${row['price_per_100g']:.2f}",
                row["price_source"] or "",
            )

    console.print(table)
    console.print(f"[dim]Showing {len(results)} results[/dim]")


# ============================================================================
# Tags Subcommands
# ============================================================================


@tags_app.command("add")
def tags_add(
    fdc_id: int = typer.Argument(..., help="FDC ID of food"),
    tag: str = typer.Argument(..., help="Tag to add"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Add a tag to a food."""
    db = get_db()
    with db.get_connection() as conn:
        food = FoodQueries.get_food_by_id(conn, fdc_id)
        if not food:
            if json_output:
                output_json({
                    "success": False,
                    "command": "tags add",
                    "errors": [f"Food with FDC ID {fdc_id} not found"],
                })
            else:
                console.print(f"[red]Food with FDC ID {fdc_id} not found[/red]")
            raise typer.Exit(1)

        TagQueries.add_tag(conn, fdc_id, tag)

    if json_output:
        output_json({
            "success": True,
            "command": "tags add",
            "data": {"fdc_id": fdc_id, "tag": tag, "description": food["description"]},
            "human_summary": f"Added tag '{tag}' to {food['description'][:40]}",
        })
    else:
        console.print(f"[green]Added tag '{tag}' to '{food['description'][:50]}'[/green]")


@tags_app.command("remove")
def tags_remove(
    fdc_id: int = typer.Argument(..., help="FDC ID of food"),
    tag: str = typer.Argument(..., help="Tag to remove"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Remove a tag from a food."""
    db = get_db()
    with db.get_connection() as conn:
        removed = TagQueries.remove_tag(conn, fdc_id, tag)

    if json_output:
        output_json({
            "success": removed,
            "command": "tags remove",
            "data": {"fdc_id": fdc_id, "tag": tag, "removed": removed},
            "human_summary": f"Removed tag '{tag}'" if removed else f"Tag '{tag}' not found",
        })
    elif removed:
        console.print(f"[green]Removed tag '{tag}' from food {fdc_id}[/green]")
    else:
        console.print(f"[yellow]Tag '{tag}' not found on food {fdc_id}[/yellow]")


@tags_app.command("list")
def tags_list(
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    fdc_id: Optional[int] = typer.Option(
        None, "--food", "-f", help="Show tags for food"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List tags or foods with a specific tag."""
    db = get_db()
    with db.get_connection() as conn:
        if fdc_id:
            # Show tags for a specific food
            food = FoodQueries.get_food_by_id(conn, fdc_id)
            if not food:
                if json_output:
                    output_json({
                        "success": False,
                        "command": "tags list",
                        "errors": [f"Food with FDC ID {fdc_id} not found"],
                    })
                else:
                    console.print(f"[red]Food with FDC ID {fdc_id} not found[/red]")
                raise typer.Exit(1)

            tags = TagQueries.get_tags_for_food(conn, fdc_id)

            if json_output:
                output_json({
                    "success": True,
                    "command": "tags list",
                    "data": {"fdc_id": fdc_id, "description": food["description"], "tags": tags},
                    "human_summary": f"{len(tags)} tags for {food['description'][:40]}",
                })
            elif tags:
                console.print(f"Tags for '{food['description'][:50]}':")
                for t in tags:
                    console.print(f"  - {t}")
            else:
                console.print(f"[yellow]No tags for food {fdc_id}[/yellow]")

        elif tag:
            # Show foods with a specific tag
            foods = TagQueries.get_foods_with_tag(conn, tag)

            if json_output:
                output_json({
                    "success": True,
                    "command": "tags list",
                    "data": {
                        "tag": tag,
                        "foods": [{"fdc_id": f["fdc_id"], "description": f["description"]} for f in foods],
                    },
                    "human_summary": f"{len(foods)} foods with tag '{tag}'",
                })
                return

            if not foods:
                console.print(f"[yellow]No foods with tag '{tag}'[/yellow]")
                return

            table = Table(title=f"Foods with tag '{tag}'")
            table.add_column("FDC ID", style="cyan")
            table.add_column("Description")

            for row in foods:
                table.add_row(str(row["fdc_id"]), row["description"])

            console.print(table)

        else:
            # Show all tags with counts
            all_tags = TagQueries.get_all_tags(conn)

            if json_output:
                output_json({
                    "success": True,
                    "command": "tags list",
                    "data": {"tags": [{"tag": t, "count": c} for t, c in all_tags]},
                    "human_summary": f"{len(all_tags)} unique tags",
                })
                return

            if not all_tags:
                console.print("[yellow]No tags defined[/yellow]")
                return

            table = Table(title="All Tags")
            table.add_column("Tag")
            table.add_column("Count", justify="right")

            for tag_name, count in all_tags:
                table.add_row(tag_name, str(count))

            console.print(table)


@tags_app.command("interactive")
def tags_interactive(
    tag: str = typer.Option(
        "staple", "--tag", "-t", help="Tag to apply (default: staple)"
    ),
) -> None:
    """Interactive mode to search and tag foods."""
    tag = tag.lower()
    console.print(f"[bold]Interactive Tagging Mode[/bold] (Tag: [cyan]{tag}[/cyan])")
    console.print("Type a search term. Then select food numbers to toggle the tag.")
    console.print("Type 'q' to quit.\n")

    db = get_db()

    while True:
        query = Prompt.ask("\n[bold green]Search[/bold green]")
        if query.lower() in ("q", "exit", "quit"):
            break

        if not query.strip():
            continue

        with db.get_connection() as conn:
            results = FoodQueries.search_foods(conn, query, limit=20)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            continue

        # Display results with index
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("FDC ID", style="cyan")
        table.add_column("Description")
        table.add_column("Current Tags")

        selection_map = {}

        for idx, row in enumerate(results, 1):
            selection_map[str(idx)] = row["fdc_id"]

            current_tags = row["tags"] if row["tags"] else ""

            # Highlight if it already has the target tag
            has_tag = tag in (current_tags or "").split(", ")

            if has_tag:
                style = "dim"
                tag_display = f"[green]{current_tags}[/green]"
            else:
                style = None
                tag_display = current_tags

            table.add_row(
                str(idx),
                str(row["fdc_id"]),
                row["description"],
                tag_display,
                style=style,
            )

        console.print(table)

        # Selection loop
        choice = Prompt.ask(
            "Select # to toggle (e.g. '1 3') or Enter to search again", default=""
        )
        if not choice.strip():
            continue

        selections = choice.split()
        with db.get_connection() as conn:
            for sel in selections:
                if sel in selection_map:
                    fdc_id = selection_map[sel]
                    # Check current state from DB for toggle accuracy
                    current_tags = TagQueries.get_tags_for_food(conn, fdc_id)

                    if tag in current_tags:
                        TagQueries.remove_tag(conn, fdc_id, tag)
                        console.print(f"  [yellow]- Removed '{tag}' from {fdc_id}[/yellow]")
                    else:
                        TagQueries.add_tag(conn, fdc_id, tag)
                        console.print(f"  [green]+ Added '{tag}' to {fdc_id}[/green]")
                else:
                    if sel.lower() not in ("s", "q"):
                        console.print(f"  [red]Invalid selection: {sel}[/red]")


# ============================================================================
# Profile Subcommands
# ============================================================================


@profile_app.command("wizard")
def profile_wizard() -> None:
    """Interactive wizard to create a constraint profile."""
    console.print("[bold]Profile Creation Wizard[/bold]")
    console.print(
        "Answer a few questions to generate a personalized constraint profile.\n"
    )

    name = Prompt.ask("Profile Name (e.g., 'summer_cut')")
    description = Prompt.ask("Description", default="Custom profile")

    console.print("\n[bold]Calorie Targets[/bold]")
    cal_min = int(Prompt.ask("Minimum Calories", default="2000"))
    cal_max = int(Prompt.ask("Maximum Calories", default=str(cal_min + 200)))

    console.print("\n[bold]Nutrient Targets[/bold]")
    prot_min = int(Prompt.ask("Minimum Protein (g)", default="150"))
    fiber_min = int(Prompt.ask("Minimum Fiber (g)", default="30"))
    sodium_max = int(Prompt.ask("Maximum Sodium (mg)", default="2300"))

    use_staples = typer.confirm("Restrict to 'staple' tagged foods only?", default=True)

    # Build constraint dictionary
    constraints = {
        "calories": {"min": cal_min, "max": cal_max},
        "nutrients": {
            "protein": {"min": prot_min},
            "fiber": {"min": fiber_min},
            "sodium": {"max": sodium_max},
        },
        "exclude_tags": ["exclude", "junk_food"],
        "options": {
            "max_grams_per_food": 500,
            "lambda_deviation": 0.001,
        },
    }

    if use_staples:
        constraints["include_tags"] = ["staple"]

    # Serialize
    constraints_json = json.dumps(constraints)

    db = get_db()
    with db.get_connection() as conn:
        existing = ProfileQueries.get_profile_by_name(conn, name)
        if existing:
            if not typer.confirm(f"Profile '{name}' exists. Overwrite?"):
                raise typer.Abort()
            ProfileQueries.update_profile(conn, name, constraints_json, description)
            console.print(f"[green]Updated profile '{name}'[/green]")
        else:
            ProfileQueries.create_profile(conn, name, constraints_json, description)
            console.print(f"[green]Created profile '{name}'[/green]")

    # Show result
    console.print("\n[dim]Profile Preview:[/dim]")
    import yaml

    console.print(yaml.dump(constraints, default_flow_style=False))


@profile_app.command("create")
def profile_create(
    name: str = typer.Argument(..., help="Profile name"),
    from_file: Path = typer.Option(
        ..., "--from-file", "-f", help="YAML constraint file"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Profile description"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Create a constraint profile from YAML file."""
    if not from_file.exists():
        if json_output:
            output_json({
                "success": False,
                "command": "profile create",
                "errors": [f"File not found: {from_file}"],
            })
        else:
            console.print(f"[red]File not found: {from_file}[/red]")
        raise typer.Exit(1)

    # Read and validate YAML
    constraints_yaml = from_file.read_text()

    # Try to parse to validate
    from mealplan.optimizer.constraints import load_profile_from_yaml

    try:
        load_profile_from_yaml(from_file)
    except Exception as e:
        if json_output:
            output_json({
                "success": False,
                "command": "profile create",
                "errors": [f"Invalid profile YAML: {e}"],
            })
        else:
            console.print(f"[red]Invalid profile YAML: {e}[/red]")
        raise typer.Exit(1)

    # Store as JSON for flexibility
    import yaml

    constraints_data = yaml.safe_load(constraints_yaml)
    constraints_json = json.dumps(constraints_data)

    db = get_db()
    with db.get_connection() as conn:
        # Check if profile already exists
        existing = ProfileQueries.get_profile_by_name(conn, name)
        if existing:
            if json_output:
                output_json({
                    "success": False,
                    "command": "profile create",
                    "errors": [f"Profile '{name}' already exists"],
                    "suggestions": ["Use a different name or delete the existing profile"],
                })
            else:
                console.print(f"[red]Profile '{name}' already exists[/red]")
            raise typer.Exit(1)

        profile_id = ProfileQueries.create_profile(
            conn, name, constraints_json, description
        )

    if json_output:
        output_json({
            "success": True,
            "command": "profile create",
            "data": {"profile_id": profile_id, "name": name},
            "human_summary": f"Created profile '{name}'",
        })
    else:
        console.print(f"[green]Created profile '{name}' (ID: {profile_id})[/green]")


@profile_app.command("list")
def profile_list(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List all saved constraint profiles."""
    db = get_db()
    with db.get_connection() as conn:
        profiles = ProfileQueries.list_profiles(conn)

    if json_output:
        output_json({
            "success": True,
            "command": "profile list",
            "data": {
                "profiles": [
                    {
                        "profile_id": p["profile_id"],
                        "name": p["name"],
                        "description": p["description"],
                        "created_at": str(p["created_at"]),
                    }
                    for p in profiles
                ],
            },
            "human_summary": f"{len(profiles)} profiles",
        })
        return

    if not profiles:
        console.print("[yellow]No profiles defined[/yellow]")
        return

    table = Table(title="Constraint Profiles")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Created")

    for row in profiles:
        table.add_row(
            row["name"],
            row["description"] or "",
            str(row["created_at"])[:19],
        )

    console.print(table)


@profile_app.command("show")
def profile_show(
    name: str = typer.Argument(..., help="Profile name"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show details of a constraint profile."""
    db = get_db()
    with db.get_connection() as conn:
        profile = ProfileQueries.get_profile_by_name(conn, name)

    if not profile:
        if json_output:
            output_json({
                "success": False,
                "command": "profile show",
                "errors": [f"Profile '{name}' not found"],
            })
        else:
            console.print(f"[red]Profile '{name}' not found[/red]")
        raise typer.Exit(1)

    constraints = json.loads(profile["constraints_json"])

    if json_output:
        output_json({
            "success": True,
            "command": "profile show",
            "data": {
                "profile_id": profile["profile_id"],
                "name": profile["name"],
                "description": profile["description"],
                "created_at": str(profile["created_at"]),
                "constraints": constraints,
            },
            "human_summary": f"Profile '{name}'",
        })
        return

    console.print(f"\n[bold]{profile['name']}[/bold]")
    if profile["description"]:
        console.print(f"Description: {profile['description']}")
    console.print(f"Created: {profile['created_at']}")
    console.print("\nConstraints:")

    import yaml
    console.print(yaml.dump(constraints, default_flow_style=False))


@profile_app.command("delete")
def profile_delete(
    name: str = typer.Argument(..., help="Profile name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Delete a constraint profile."""
    db = get_db()
    with db.get_connection() as conn:
        profile = ProfileQueries.get_profile_by_name(conn, name)
        if not profile:
            if json_output:
                output_json({
                    "success": False,
                    "command": "profile delete",
                    "errors": [f"Profile '{name}' not found"],
                })
            else:
                console.print(f"[red]Profile '{name}' not found[/red]")
            raise typer.Exit(1)

        if not force and not json_output:
            confirm = typer.confirm(f"Delete profile '{name}'?")
            if not confirm:
                raise typer.Abort()

        ProfileQueries.delete_profile(conn, name)

    if json_output:
        output_json({
            "success": True,
            "command": "profile delete",
            "data": {"name": name},
            "human_summary": f"Deleted profile '{name}'",
        })
    else:
        console.print(f"[green]Deleted profile '{name}'[/green]")


# ============================================================================
# Patterns Commands
# ============================================================================


@patterns_app.command("list")
def patterns_list(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List available dietary patterns."""
    from mealplan.data.dietary_patterns import list_patterns, DIETARY_PATTERNS

    patterns = list_patterns()

    if json_output:
        output_json({
            "success": True,
            "command": "patterns list",
            "data": {"patterns": patterns},
            "human_summary": f"{len(patterns)} dietary patterns available",
        })
    else:
        from rich.table import Table

        table = Table(title="Available Dietary Patterns")
        table.add_column("Name", style="cyan")
        table.add_column("Description")

        for p in patterns:
            table.add_row(p["name"], p["description"])

        console.print(table)
        console.print(f"\n[dim]Use with: mealplan optimize --pattern <name>[/dim]")


@patterns_app.command("info")
def patterns_info(
    name: str = typer.Argument(..., help="Pattern name"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show details about a dietary pattern."""
    from mealplan.data.dietary_patterns import get_pattern

    pattern = get_pattern(name)
    if not pattern:
        if json_output:
            output_json({
                "success": False,
                "command": "patterns info",
                "errors": [f"Pattern '{name}' not found"],
                "suggestions": ["Use 'mealplan patterns list' to see available patterns"],
            })
        else:
            console.print(f"[red]Pattern '{name}' not found[/red]")
        raise typer.Exit(1)

    if json_output:
        output_json({
            "success": True,
            "command": "patterns info",
            "data": {
                "name": pattern.name,
                "description": pattern.description,
                "include_keywords_count": len(pattern.include_keywords),
                "exclude_keywords_count": len(pattern.exclude_keywords),
                "sample_exclude": pattern.exclude_keywords[:10],
                "has_macro_targets": pattern.macro_targets is not None,
            },
            "human_summary": f"Pattern '{name}': {pattern.description}",
        })
    else:
        console.print(f"[bold]{pattern.name}[/bold]")
        console.print(f"  {pattern.description}")
        console.print(f"\n  Include keywords: {len(pattern.include_keywords)}")
        console.print(f"  Exclude keywords: {len(pattern.exclude_keywords)}")
        if pattern.exclude_keywords:
            console.print(f"  Sample excludes: {', '.join(pattern.exclude_keywords[:10])}")
        if pattern.macro_targets:
            console.print(f"  Has macro targets: Yes")


# ============================================================================
# User Profile Commands
# ============================================================================


@user_app.command("create")
def user_create(
    age: int = typer.Option(..., "--age", help="Age in years"),
    sex: str = typer.Option(..., "--sex", help="Sex (male/female)"),
    height: float = typer.Option(..., "--height", help="Height in inches"),
    weight: float = typer.Option(..., "--weight", help="Current weight in lbs"),
    activity: str = typer.Option(
        "moderate",
        "--activity",
        help="Activity level (sedentary/lightly_active/moderate/active/very_active)",
    ),
    goal: Optional[str] = typer.Option(
        None, "--goal", help="Goal (e.g., 'fat_loss:185:165')"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Create a user profile for weight tracking."""
    from mealplan.tracking.models import UserProfile
    from mealplan.tracking.queries import UserQueries

    db = get_db()

    # Parse goal if provided
    target_weight = None
    if goal and ":" in goal:
        parts = goal.split(":")
        if len(parts) >= 3:
            try:
                target_weight = float(parts[2])
            except ValueError:
                pass

    profile = UserProfile(
        user_id=None,
        age=age,
        sex=sex,
        height_inches=height,
        weight_lbs=weight,
        activity_level=activity,
        goal=goal,
        target_weight_lbs=target_weight,
    )

    with db.get_connection() as conn:
        user_id = UserQueries.create_user(conn, profile)

    if json_output:
        output_json({
            "success": True,
            "command": "user create",
            "data": {"user_id": user_id, "profile": {
                "age": age, "sex": sex, "height_inches": height,
                "weight_lbs": weight, "activity_level": activity,
                "goal": goal, "target_weight_lbs": target_weight,
            }},
            "human_summary": f"Created user profile (ID: {user_id})",
        })
    else:
        console.print(f"[green]Created user profile (ID: {user_id})[/green]")


@user_app.command("show")
def user_show(
    user_id: Optional[int] = typer.Option(None, "--id", help="User ID (default: first user)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show user profile."""
    from mealplan.tracking.queries import UserQueries

    db = get_db()
    with db.get_connection() as conn:
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

    if profile is None:
        if json_output:
            output_json({
                "success": False,
                "command": "user show",
                "errors": ["No user profile found"],
                "suggestions": ["Create a profile with: mealplan user create --age 35 --sex male --height 70 --weight 180"],
            })
        else:
            console.print("[red]No user profile found[/red]")
            console.print("Create one with: mealplan user create --age 35 --sex male --height 70 --weight 180")
        raise typer.Exit(1)

    if json_output:
        output_json({
            "success": True,
            "command": "user show",
            "data": {
                "user_id": profile.user_id,
                "age": profile.age,
                "sex": profile.sex,
                "height_inches": profile.height_inches,
                "weight_lbs": profile.weight_lbs,
                "activity_level": profile.activity_level,
                "goal": profile.goal,
                "target_weight_lbs": profile.target_weight_lbs,
            },
            "human_summary": f"User {profile.user_id}: {profile.sex}, {profile.age}y, {profile.height_inches}in, {profile.weight_lbs}lbs",
        })
    else:
        console.print(f"[bold]User Profile (ID: {profile.user_id})[/bold]")
        console.print(f"  Age: {profile.age}")
        console.print(f"  Sex: {profile.sex}")
        console.print(f"  Height: {profile.height_inches} inches")
        console.print(f"  Weight: {profile.weight_lbs} lbs")
        console.print(f"  Activity: {profile.activity_level}")
        if profile.goal:
            console.print(f"  Goal: {profile.goal}")
        if profile.target_weight_lbs:
            console.print(f"  Target weight: {profile.target_weight_lbs} lbs")


@user_app.command("update")
def user_update(
    user_id: Optional[int] = typer.Option(None, "--id", help="User ID (default: first user)"),
    weight: Optional[float] = typer.Option(None, "--weight", help="Update current weight"),
    goal: Optional[str] = typer.Option(None, "--goal", help="Update goal"),
    activity: Optional[str] = typer.Option(None, "--activity", help="Update activity level"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Update user profile."""
    from mealplan.tracking.queries import UserQueries

    db = get_db()
    with db.get_connection() as conn:
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

        if profile is None:
            if json_output:
                output_json({
                    "success": False,
                    "command": "user update",
                    "errors": ["No user profile found"],
                })
            else:
                console.print("[red]No user profile found[/red]")
            raise typer.Exit(1)

        # Update fields
        if weight is not None:
            profile.weight_lbs = weight
        if activity is not None:
            profile.activity_level = activity
        if goal is not None:
            profile.goal = goal
            # Parse target weight from goal
            if ":" in goal:
                parts = goal.split(":")
                if len(parts) >= 3:
                    try:
                        profile.target_weight_lbs = float(parts[2])
                    except ValueError:
                        pass

        UserQueries.update_user(conn, profile)

    if json_output:
        output_json({
            "success": True,
            "command": "user update",
            "data": {"user_id": profile.user_id},
            "human_summary": "Profile updated",
        })
    else:
        console.print("[green]Profile updated[/green]")


# ============================================================================
# Weight Tracking Commands
# ============================================================================


@weight_app.command("add")
def weight_add(
    weight: float = typer.Argument(..., help="Weight in lbs"),
    date_str: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date (YYYY-MM-DD, default: today)"
    ),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Optional notes"),
    user_id: Optional[int] = typer.Option(None, "--user", help="User ID (default: first user)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Add a weight entry (EMA trend computed automatically)."""
    from datetime import date, datetime

    from mealplan.tracking.queries import UserQueries, WeightQueries

    db = get_db()

    # Parse date
    if date_str:
        measured_at = date.fromisoformat(date_str)
    else:
        measured_at = date.today()

    with db.get_connection() as conn:
        # Get user
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

        if profile is None:
            if json_output:
                output_json({
                    "success": False,
                    "command": "weight add",
                    "errors": ["No user profile found"],
                    "suggestions": ["Create a profile first: mealplan user create ..."],
                })
            else:
                console.print("[red]No user profile found. Create one first.[/red]")
            raise typer.Exit(1)

        entry = WeightQueries.add_weight(
            conn, profile.user_id, weight, measured_at, notes  # type: ignore
        )

    if json_output:
        output_json({
            "success": True,
            "command": "weight add",
            "data": {
                "weight_lbs": entry.weight_lbs,
                "trend_lbs": round(entry.trend_lbs, 1),
                "measured_at": entry.measured_at.isoformat(),
            },
            "human_summary": f"Logged {weight:.1f} lbs, trend: {entry.trend_lbs:.1f} lbs",
        })
    else:
        console.print(f"[green]Logged:[/green] {weight:.1f} lbs on {measured_at}")
        console.print(f"[blue]Trend:[/blue] {entry.trend_lbs:.1f} lbs (EMA)")


@weight_app.command("list")
def weight_list(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to show"),
    user_id: Optional[int] = typer.Option(None, "--user", help="User ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List weight history with trends."""
    from mealplan.tracking.queries import UserQueries, WeightQueries

    db = get_db()
    with db.get_connection() as conn:
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

        if profile is None:
            if json_output:
                output_json({"success": False, "errors": ["No user profile found"]})
            else:
                console.print("[red]No user profile found[/red]")
            raise typer.Exit(1)

        history = WeightQueries.get_weight_history(conn, profile.user_id, days=days)  # type: ignore

    if not history:
        if json_output:
            output_json({"success": True, "data": {"entries": []}, "human_summary": "No weight entries found"})
        else:
            console.print("No weight entries found")
        return

    if json_output:
        output_json({
            "success": True,
            "command": "weight list",
            "data": {
                "entries": [
                    {
                        "date": e.measured_at.isoformat(),
                        "weight_lbs": e.weight_lbs,
                        "trend_lbs": round(e.trend_lbs, 1),
                    }
                    for e in history
                ]
            },
            "human_summary": f"{len(history)} entries over {days} days",
        })
    else:
        table = Table(title=f"Weight History (last {days} days)")
        table.add_column("Date", style="cyan")
        table.add_column("Weight", justify="right")
        table.add_column("Trend", justify="right", style="blue")
        table.add_column("Δ", justify="right")

        prev_trend = None
        for entry in history:
            delta = ""
            if prev_trend is not None:
                d = entry.trend_lbs - prev_trend
                delta = f"{d:+.1f}"
            prev_trend = entry.trend_lbs

            table.add_row(
                entry.measured_at.isoformat(),
                f"{entry.weight_lbs:.1f}",
                f"{entry.trend_lbs:.1f}",
                delta,
            )

        console.print(table)


@weight_app.command("trend")
def weight_trend(
    days: int = typer.Option(30, "--days", "-d", help="Days to analyze"),
    user_id: Optional[int] = typer.Option(None, "--user", help="User ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show weight trend analysis."""
    from mealplan.tracking.diagnostics import format_weight_report, generate_weight_report
    from mealplan.tracking.queries import UserQueries

    db = get_db()
    with db.get_connection() as conn:
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

        if profile is None:
            if json_output:
                output_json({"success": False, "errors": ["No user profile found"]})
            else:
                console.print("[red]No user profile found[/red]")
            raise typer.Exit(1)

        report = generate_weight_report(conn, profile.user_id, days=days)  # type: ignore

    if report is None:
        if json_output:
            output_json({"success": False, "errors": ["Not enough data for trend analysis"]})
        else:
            console.print("[yellow]Not enough data for trend analysis[/yellow]")
        raise typer.Exit(1)

    if json_output:
        output_json({
            "success": True,
            "command": "weight trend",
            "data": {
                "current_weight_lbs": report.current_weight,
                "current_trend_lbs": round(report.current_trend, 1),
                "trend_change_lbs": round(report.trend_change, 1),
                "weekly_rate_lbs": round(report.weekly_rate, 2),
                "implied_daily_deficit_kcal": round(report.implied_daily_deficit, 0),
                "period_days": report.period_days,
            },
            "human_summary": f"Trend: {report.current_trend:.1f} lbs, {report.weekly_rate:+.1f} lbs/week",
        })
    else:
        console.print(format_weight_report(report))


# ============================================================================
# Calorie Logging Commands
# ============================================================================


@calories_app.command("log")
def calories_log(
    calories: float = typer.Argument(..., help="Planned calories"),
    date_str: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date (YYYY-MM-DD, default: today)"
    ),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Optional notes"),
    user_id: Optional[int] = typer.Option(None, "--user", help="User ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Log planned calorie intake for a day."""
    from datetime import date

    from mealplan.tracking.queries import CalorieQueries, UserQueries

    db = get_db()

    if date_str:
        log_date = date.fromisoformat(date_str)
    else:
        log_date = date.today()

    with db.get_connection() as conn:
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

        if profile is None:
            if json_output:
                output_json({"success": False, "errors": ["No user profile found"]})
            else:
                console.print("[red]No user profile found[/red]")
            raise typer.Exit(1)

        entry = CalorieQueries.log_calories(
            conn, profile.user_id, log_date, calories, notes  # type: ignore
        )

    if json_output:
        output_json({
            "success": True,
            "command": "calories log",
            "data": {
                "date": entry.date.isoformat(),
                "planned_calories": entry.planned_calories,
            },
            "human_summary": f"Logged {calories:.0f} kcal for {log_date}",
        })
    else:
        console.print(f"[green]Logged:[/green] {calories:.0f} kcal for {log_date}")


@calories_app.command("list")
def calories_list(
    days: int = typer.Option(30, "--days", "-d", help="Days to show"),
    user_id: Optional[int] = typer.Option(None, "--user", help="User ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List calorie log history."""
    from datetime import date, timedelta

    from mealplan.tracking.queries import CalorieQueries, UserQueries

    db = get_db()
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    with db.get_connection() as conn:
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

        if profile is None:
            if json_output:
                output_json({"success": False, "errors": ["No user profile found"]})
            else:
                console.print("[red]No user profile found[/red]")
            raise typer.Exit(1)

        history = CalorieQueries.get_calorie_history(
            conn, profile.user_id, start_date, end_date  # type: ignore
        )

    if not history:
        if json_output:
            output_json({"success": True, "data": {"entries": []}, "human_summary": "No calorie entries"})
        else:
            console.print("No calorie entries found")
        return

    if json_output:
        avg = sum(e.planned_calories for e in history) / len(history)
        output_json({
            "success": True,
            "command": "calories list",
            "data": {
                "entries": [
                    {"date": e.date.isoformat(), "planned_calories": e.planned_calories}
                    for e in history
                ],
                "average_calories": round(avg, 0),
            },
            "human_summary": f"{len(history)} entries, avg {avg:.0f} kcal/day",
        })
    else:
        table = Table(title=f"Calorie Log (last {days} days)")
        table.add_column("Date", style="cyan")
        table.add_column("Planned Calories", justify="right")

        for entry in history:
            table.add_row(entry.date.isoformat(), f"{entry.planned_calories:.0f}")

        console.print(table)

        avg = sum(e.planned_calories for e in history) / len(history)
        console.print(f"\nAverage: {avg:.0f} kcal/day")


# ============================================================================
# TDEE Estimation Commands
# ============================================================================


@tdee_app.command("estimate")
def tdee_estimate(
    user_id: Optional[int] = typer.Option(None, "--user", help="User ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run TDEE estimation using Kalman filter on weight/calorie data."""
    from datetime import date, timedelta

    from mealplan.tracking.diagnostics import (
        calculate_mifflin_tdee,
        calculate_mifflin_tdee_at_weight,
    )
    from mealplan.tracking.models import TDEEEstimate
    from mealplan.tracking.queries import (
        CalorieQueries,
        TDEEQueries,
        UserQueries,
        WeightQueries,
    )
    from mealplan.tracking.tdee_filter import TDEEFilter

    db = get_db()

    with db.get_connection() as conn:
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

        if profile is None:
            if json_output:
                output_json({"success": False, "errors": ["No user profile found"]})
            else:
                console.print("[red]No user profile found[/red]")
            raise typer.Exit(1)

        uid = profile.user_id  # type: ignore

        # Get weight history (last 8 weeks for weekly updates)
        end_date = date.today()
        start_date = end_date - timedelta(days=56)
        weight_history = WeightQueries.get_weight_history(
            conn, uid, start_date=start_date, end_date=end_date
        )

        if len(weight_history) < 7:
            if json_output:
                output_json({
                    "success": False,
                    "errors": ["Not enough weight data (need at least 1 week)"],
                    "suggestions": ["Log your weight daily with: mealplan weight add <weight>"],
                })
            else:
                console.print("[yellow]Not enough weight data. Need at least 1 week.[/yellow]")
            raise typer.Exit(1)

        # Calculate baseline Mifflin-St Jeor TDEE (for final display)
        mifflin_tdee = calculate_mifflin_tdee(profile)

        # Always start fresh - recalculating from the 8-week history is cheap
        # and avoids compounding bias if tdee estimate is run multiple times
        tdee_filter = TDEEFilter()

        # Run filter on weekly data
        # Group weight data by week and compute trend changes
        # Use time-varying baseline: recalculate Mifflin each week using that
        # week's starting trend weight. This keeps the learned bias interpretable
        # as "personal deviation from formula" rather than absorbing weight-drift.
        weeks_processed = 0
        for week_start in range(0, len(weight_history) - 7, 7):
            week_entries = weight_history[week_start : week_start + 7]
            if len(week_entries) < 2:
                continue

            trend_start = week_entries[0].trend_lbs
            trend_end = week_entries[-1].trend_lbs

            # Get average calories for this week
            week_start_date = week_entries[0].measured_at
            week_end_date = week_entries[-1].measured_at
            avg_calories = CalorieQueries.get_average_calories(
                conn, uid, week_start_date, week_end_date
            )

            if avg_calories is None:
                continue  # Skip weeks without calorie data

            # Calculate Mifflin TDEE using this week's starting weight
            # (time-varying baseline for accurate bias learning)
            week_mifflin_tdee = calculate_mifflin_tdee_at_weight(profile, trend_start)

            # Implied deficit from trend change
            implied_deficit = (trend_start - trend_end) * 3500 / 7

            # Expected deficit from plan (using this week's weight-adjusted TDEE)
            expected_deficit = week_mifflin_tdee - avg_calories

            # Update filter
            tdee_filter.predict_and_update(implied_deficit, expected_deficit, days=7)
            weeks_processed += 1

        # Save estimate
        adjusted_tdee, uncertainty = tdee_filter.get_adjusted_tdee(mifflin_tdee)
        estimate = TDEEEstimate(
            estimate_id=None,
            user_id=uid,
            estimated_at=end_date,
            mifflin_tdee=mifflin_tdee,
            tdee_bias=tdee_filter.bias,
            variance=tdee_filter.variance,
            adjusted_tdee=adjusted_tdee,
        )
        TDEEQueries.save_estimate(conn, estimate)

    if json_output:
        output_json({
            "success": True,
            "command": "tdee estimate",
            "data": {
                "mifflin_tdee": round(mifflin_tdee, 0),
                "tdee_bias": round(tdee_filter.bias, 0),
                "adjusted_tdee": round(adjusted_tdee, 0),
                "uncertainty_95ci": round(uncertainty, 0),
                "weeks_processed": weeks_processed,
            },
            "human_summary": f"TDEE: {adjusted_tdee:.0f} ± {uncertainty:.0f} kcal/day (bias: {tdee_filter.bias:+.0f})",
        })
    else:
        console.print(f"[green]Updated Total Daily Energy Expenditure (TDEE) estimate[/green] ({weeks_processed} weeks processed)")
        console.print(f"  Mifflin-St Jeor baseline: {mifflin_tdee:.0f} kcal/day")
        console.print(f"  Learned bias: {tdee_filter.bias:+.0f} kcal/day")
        console.print(f"  [bold]Your TDEE: {adjusted_tdee:.0f} ± {uncertainty:.0f} kcal/day[/bold]")
        console.print()
        console.print("[dim]Run 'mealplan tdee progress' for full report[/dim]")


@tdee_app.command("targets")
def tdee_targets(
    user_id: Optional[int] = typer.Option(None, "--user", help="User ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Get adaptive calorie/macro targets based on current progress."""
    from datetime import date, timedelta

    from mealplan.tracking.diagnostics import calculate_mifflin_tdee, generate_weight_report
    from mealplan.tracking.queries import CalorieQueries, TDEEQueries, UserQueries

    db = get_db()

    with db.get_connection() as conn:
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

        if profile is None:
            if json_output:
                output_json({"success": False, "errors": ["No user profile found"]})
            else:
                console.print("[red]No user profile found[/red]")
            raise typer.Exit(1)

        uid = profile.user_id  # type: ignore
        latest_estimate = TDEEQueries.get_latest_estimate(conn, uid)

        # Get current intake and weight loss rate
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        avg_intake = CalorieQueries.get_average_calories(conn, uid, start_date, end_date)
        weight_report = generate_weight_report(conn, uid, days=30)

    # Use estimate if available, otherwise Mifflin-St Jeor
    if latest_estimate:
        tdee = latest_estimate.adjusted_tdee
        uncertainty = latest_estimate.uncertainty_95ci
    else:
        tdee = calculate_mifflin_tdee(profile)
        uncertainty = 100  # Default uncertainty for no data

    # Determine target rate based on goal
    goal = profile.goal
    if goal and goal.startswith("fat_loss"):
        target_rate = -1.0  # 1 lb/week loss
        goal_desc = "fat loss (1 lb/week)"
    elif goal and goal.startswith("aggressive"):
        target_rate = -1.5  # 1.5 lb/week loss
        goal_desc = "aggressive fat loss (1.5 lb/week)"
    elif goal and goal.startswith("lean_gain"):
        target_rate = 0.5  # 0.5 lb/week gain
        goal_desc = "lean gain (0.5 lb/week)"
    elif goal and goal.startswith("muscle_gain"):
        target_rate = 1.0  # 1 lb/week gain
        goal_desc = "muscle gain (1 lb/week)"
    else:
        target_rate = 0.0
        goal_desc = "maintenance"

    # Current situation
    current_rate = weight_report.weekly_rate if weight_report else 0.0
    current_intake = avg_intake if avg_intake else tdee  # Assume maintenance if no data

    # Calculate what intake would achieve target rate
    # target_rate (lbs/week) * 500 = daily calorie change needed
    target_deficit = -target_rate * 500  # positive = deficit, negative = surplus
    target_calories = tdee - target_deficit

    # Protein: 0.8-1.0 g/lb for fat loss, 0.7-0.8 for maintenance
    weight = profile.weight_lbs
    if target_rate < 0:  # Cutting
        protein_min = int(weight * 0.8)
        protein_max = int(weight * 1.0)
    else:
        protein_min = int(weight * 0.7)
        protein_max = int(weight * 0.9)

    # Generate recommendation based on current progress
    recommendation = None
    recommended_intake = target_calories  # Default to theoretical target

    if weight_report and avg_intake:
        rate_diff = current_rate - target_rate
        if abs(rate_diff) < 0.2:
            # On track - keep current intake
            recommendation = "You're on track! Keep doing what you're doing."
            recommended_intake = current_intake
        elif target_rate < 0:  # Trying to lose
            if current_rate > target_rate + 0.2:  # Not losing fast enough (rate is less negative)
                cal_adjust = int((current_rate - target_rate) * 500)
                recommended_intake = current_intake - cal_adjust
                recommendation = f"Losing slower than target. Consider reducing intake by ~{cal_adjust} kcal/day."
            elif current_rate < target_rate - 0.2:  # Losing too fast (rate is more negative)
                cal_adjust = int((target_rate - current_rate) * 500)
                recommended_intake = current_intake + abs(cal_adjust)
                recommendation = f"Losing faster than target ({abs(current_rate):.1f} lb/week). Consider increasing intake by ~{abs(cal_adjust)} kcal/day to preserve muscle."
        elif target_rate > 0:  # Trying to gain
            if current_rate < target_rate - 0.2:  # Not gaining fast enough
                cal_adjust = int((target_rate - current_rate) * 500)
                recommended_intake = current_intake + cal_adjust
                recommendation = f"Gaining slower than target. Consider increasing intake by ~{cal_adjust} kcal/day."

    if json_output:
        output_json({
            "success": True,
            "command": "tdee targets",
            "data": {
                "tdee": round(tdee, 0),
                "tdee_uncertainty": round(uncertainty, 0),
                "goal": goal_desc,
                "target_rate_lbs_week": target_rate,
                "current_rate_lbs_week": round(current_rate, 2) if weight_report else None,
                "current_intake": round(current_intake, 0) if avg_intake else None,
                "recommended_intake": round(recommended_intake, 0),
                "protein_min": protein_min,
                "protein_max": protein_max,
                "recommendation": recommendation,
            },
            "human_summary": recommendation or f"Target: {recommended_intake:.0f} kcal/day for {goal_desc}",
        })
    else:
        console.print("[bold]Adaptive Calorie Targets[/bold]")
        console.print(f"  Estimated TDEE: {tdee:.0f} ± {uncertainty:.0f} kcal/day")
        console.print(f"  Goal: {goal_desc}")
        console.print()

        if weight_report and avg_intake:
            console.print("[bold]Current status:[/bold]")
            console.print(f"  Average intake: {current_intake:.0f} kcal/day")
            rate_dir = "losing" if current_rate < 0 else "gaining" if current_rate > 0 else "maintaining"
            console.print(f"  Current rate: {abs(current_rate):.1f} lbs/week ({rate_dir})")
            console.print()

        console.print("[bold]Recommendation:[/bold]")
        console.print(f"  [green]Calorie target:[/green] {recommended_intake:.0f} kcal/day")
        console.print(f"  [green]Protein range:[/green] {protein_min} - {protein_max} g/day")

        if recommendation:
            console.print()
            console.print(f"  [cyan]{recommendation}[/cyan]")


@tdee_app.command("progress")
def tdee_progress(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to analyze"),
    user_id: Optional[int] = typer.Option(None, "--user", help="User ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show comprehensive weight loss progress report."""
    from mealplan.tracking.diagnostics import (
        format_progress_report,
        generate_progress_report,
    )
    from mealplan.tracking.queries import UserQueries

    db = get_db()

    with db.get_connection() as conn:
        if user_id:
            profile = UserQueries.get_user(conn, user_id)
        else:
            profile = UserQueries.get_default_user(conn)

        if profile is None:
            if json_output:
                output_json({"success": False, "errors": ["No user profile found"]})
            else:
                console.print("[red]No user profile found[/red]")
            raise typer.Exit(1)

        uid = profile.user_id  # type: ignore
        report = generate_progress_report(conn, uid, days)

    if report is None:
        if json_output:
            output_json({
                "success": False,
                "errors": ["Not enough weight data for progress report"],
                "suggestions": ["Log at least 2 weight entries with: mealplan weight add <weight>"],
            })
        else:
            console.print("[yellow]Not enough weight data for progress report[/yellow]")
            console.print("Log weight with: mealplan weight add <weight>")
        raise typer.Exit(1)

    if json_output:
        output_json({
            "success": True,
            "command": "tdee progress",
            "data": {
                "weight": {
                    "current_weight": report.weight.current_weight,
                    "current_trend": report.weight.current_trend,
                    "trend_change": report.weight.trend_change,
                    "weekly_rate": report.weight.weekly_rate,
                    "implied_daily_deficit": report.weight.implied_daily_deficit,
                    "period_days": report.weight.period_days,
                },
                "tdee": {
                    "mifflin_tdee": report.tdee.mifflin_tdee,
                    "tdee_bias": report.tdee.tdee_bias,
                    "adjusted_tdee": report.tdee.adjusted_tdee,
                    "uncertainty_95ci": report.tdee.uncertainty_95ci,
                    "avg_planned_calories": report.tdee.avg_planned_calories,
                    "expected_deficit": report.tdee.expected_deficit,
                    "implied_deficit": report.tdee.implied_deficit,
                } if report.tdee else None,
                "goal_weight": report.goal_weight,
                "remaining_lbs": report.remaining_lbs,
                "weeks_to_goal": report.weeks_to_goal,
                "notes": report.notes,
            },
            "human_summary": f"Lost {abs(report.weight.trend_change):.1f} lbs over {report.weight.period_days} days ({abs(report.weight.weekly_rate):.1f} lbs/week)",
        })
    else:
        console.print(format_progress_report(report))


if __name__ == "__main__":
    app()
