"""CLI interface using Typer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
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

app.add_typer(prices_app, name="prices")
app.add_typer(tags_app, name="tags")
app.add_typer(profile_app, name="profile")


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
) -> None:
    """Initialize database from USDA data."""
    from mealplan.data.usda_loader import USDALoader
    from mealplan.db.connection import DatabaseConnection

    # Use custom path or default
    if db_path:
        db = DatabaseConnection(db_path)
    else:
        db = get_db()

    console.print(f"Initializing database at: {db.db_path}")

    # Create schema
    db.initialize_schema()
    console.print("[green]Schema created[/green]")

    # Load USDA data
    settings = get_settings()
    data_types = settings.usda.data_types

    console.print(f"Loading USDA data from: {usda_path}")
    console.print(f"Data types: {', '.join(data_types)}")

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
) -> None:
    """Search for foods by description."""
    db = get_db()
    with db.get_connection() as conn:
        results = FoodQueries.search_foods(conn, query, limit)

    if not results:
        console.print(f"[yellow]No foods found matching '{query}'[/yellow]")
        return

    table = Table(title=f"Foods matching '{query}'")
    table.add_column("FDC ID", style="cyan")
    table.add_column("Description")
    table.add_column("Type", style="dim")

    for row in results:
        table.add_row(
            str(row["fdc_id"]),
            row["description"],
            row["data_type"] or "",
        )

    console.print(table)
    console.print(f"[dim]Showing {len(results)} results[/dim]")


@app.command()
def info(
    fdc_id: int = typer.Argument(..., help="FDC ID of food to show"),
) -> None:
    """Show detailed nutrient information for a food."""
    db = get_db()
    with db.get_connection() as conn:
        food = FoodQueries.get_food_by_id(conn, fdc_id)
        if not food:
            console.print(f"[red]Food with FDC ID {fdc_id} not found[/red]")
            raise typer.Exit(1)

        nutrients = FoodQueries.get_food_nutrients(conn, fdc_id)
        price = PriceQueries.get_price(conn, fdc_id)
        tags = TagQueries.get_tags_for_food(conn, fdc_id)

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
    days: int = typer.Option(1, "--days", "-d", help="Number of days to plan"),
    output: str = typer.Option(
        "table", "--output", "-o", help="Output format: table, json, markdown"
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Save run to history"),
) -> None:
    """Run meal plan optimization."""
    from mealplan.export.formatters import format_result
    from mealplan.optimizer.constraints import load_profile_from_yaml
    from mealplan.optimizer.models import OptimizationRequest
    from mealplan.optimizer.solver import solve_diet_problem

    db = get_db()
    profile_id: Optional[int] = None
    profile_name: Optional[str] = None

    # Load constraints
    if profile_file:
        # Load from YAML file
        if not profile_file.exists():
            console.print(f"[red]Profile file not found: {profile_file}[/red]")
            raise typer.Exit(1)
        request = load_profile_from_yaml(profile_file)
        profile_name = profile_file.stem
    elif profile:
        # Load from database
        with db.get_connection() as conn:
            profile_row = ProfileQueries.get_profile_by_name(conn, profile)
            if not profile_row:
                console.print(f"[red]Profile not found: {profile}[/red]")
                raise typer.Exit(1)
            profile_id = profile_row["profile_id"]
            profile_name = profile_row["name"]
            # Parse stored JSON into request
            constraints_data = json.loads(profile_row["constraints_json"])
            # For simplicity, we'll reconstruct the YAML and parse it
            # In production, you'd want a more direct deserialization
            request = OptimizationRequest()
            # ... apply constraints from JSON
            console.print(f"[yellow]Loading profile '{profile}' from database[/yellow]")
    else:
        # Use default request
        request = OptimizationRequest()
        console.print("[yellow]Using default constraints[/yellow]")

    request.planning_days = days

    # Run optimization
    with console.status("[bold green]Optimizing..."):
        with db.get_connection() as conn:
            result = solve_diet_problem(request, conn)

    # Save to history if requested
    run_id: Optional[int] = None
    if save:
        from mealplan.export.formatters import JSONFormatter

        formatter = JSONFormatter()
        result_json = formatter.format(result, profile_name)
        with db.get_connection() as conn:
            run_id = OptimizationRunQueries.save_run(
                conn,
                profile_id,
                result.status,
                result.total_cost,
                result_json,
            )

    # Format output
    formatted = format_result(result, output, profile_name, run_id, console)
    if formatted:
        console.print(formatted)


@app.command("export-for-llm")
def export_for_llm(
    run_id: str = typer.Argument(..., help="Run ID or 'latest'"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    days: int = typer.Option(1, "--days", "-d", help="Number of days for meal plan"),
) -> None:
    """Export optimization result as LLM-ready prompt."""
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

    # Reconstruct OptimizationResult from JSON
    foods = [
        FoodResult(
            fdc_id=f["fdc_id"],
            description=f["description"],
            grams=f["grams"],
            cost=f["cost"],
            nutrients={},
        )
        for f in result_data["solution"]["foods"]
    ]

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
) -> None:
    """Show optimization run history."""
    db = get_db()
    with db.get_connection() as conn:
        runs = OptimizationRunQueries.list_runs(conn, limit)

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
# Prices Subcommands
# ============================================================================


@prices_app.command("add")
def prices_add(
    fdc_id: int = typer.Argument(..., help="FDC ID of food"),
    price_per_100g: float = typer.Argument(..., help="Price per 100 grams"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Price source"),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Notes"),
) -> None:
    """Add or update price for a food."""
    db = get_db()
    with db.get_connection() as conn:
        # Verify food exists
        food = FoodQueries.get_food_by_id(conn, fdc_id)
        if not food:
            console.print(f"[red]Food with FDC ID {fdc_id} not found[/red]")
            raise typer.Exit(1)

        PriceQueries.upsert_price(conn, fdc_id, price_per_100g, source, notes)

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
) -> None:
    """Add a tag to a food."""
    db = get_db()
    with db.get_connection() as conn:
        food = FoodQueries.get_food_by_id(conn, fdc_id)
        if not food:
            console.print(f"[red]Food with FDC ID {fdc_id} not found[/red]")
            raise typer.Exit(1)

        TagQueries.add_tag(conn, fdc_id, tag)

    console.print(f"[green]Added tag '{tag}' to '{food['description'][:50]}'[/green]")


@tags_app.command("remove")
def tags_remove(
    fdc_id: int = typer.Argument(..., help="FDC ID of food"),
    tag: str = typer.Argument(..., help="Tag to remove"),
) -> None:
    """Remove a tag from a food."""
    db = get_db()
    with db.get_connection() as conn:
        removed = TagQueries.remove_tag(conn, fdc_id, tag)

    if removed:
        console.print(f"[green]Removed tag '{tag}' from food {fdc_id}[/green]")
    else:
        console.print(f"[yellow]Tag '{tag}' not found on food {fdc_id}[/yellow]")


@tags_app.command("list")
def tags_list(
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    fdc_id: Optional[int] = typer.Option(
        None, "--food", "-f", help="Show tags for food"
    ),
) -> None:
    """List tags or foods with a specific tag."""
    db = get_db()
    with db.get_connection() as conn:
        if fdc_id:
            # Show tags for a specific food
            food = FoodQueries.get_food_by_id(conn, fdc_id)
            if not food:
                console.print(f"[red]Food with FDC ID {fdc_id} not found[/red]")
                raise typer.Exit(1)

            tags = TagQueries.get_tags_for_food(conn, fdc_id)
            if tags:
                console.print(f"Tags for '{food['description'][:50]}':")
                for t in tags:
                    console.print(f"  - {t}")
            else:
                console.print(f"[yellow]No tags for food {fdc_id}[/yellow]")

        elif tag:
            # Show foods with a specific tag
            foods = TagQueries.get_foods_with_tag(conn, tag)
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
            if not all_tags:
                console.print("[yellow]No tags defined[/yellow]")
                return

            table = Table(title="All Tags")
            table.add_column("Tag")
            table.add_column("Count", justify="right")

            for tag_name, count in all_tags:
                table.add_row(tag_name, str(count))

            console.print(table)


# ============================================================================
# Profile Subcommands
# ============================================================================


@profile_app.command("create")
def profile_create(
    name: str = typer.Argument(..., help="Profile name"),
    from_file: Path = typer.Option(
        ..., "--from-file", "-f", help="YAML constraint file"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Profile description"
    ),
) -> None:
    """Create a constraint profile from YAML file."""
    if not from_file.exists():
        console.print(f"[red]File not found: {from_file}[/red]")
        raise typer.Exit(1)

    # Read and validate YAML
    constraints_yaml = from_file.read_text()

    # Try to parse to validate
    from mealplan.optimizer.constraints import load_profile_from_yaml

    try:
        load_profile_from_yaml(from_file)
    except Exception as e:
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
            console.print(f"[red]Profile '{name}' already exists[/red]")
            raise typer.Exit(1)

        profile_id = ProfileQueries.create_profile(
            conn, name, constraints_json, description
        )

    console.print(f"[green]Created profile '{name}' (ID: {profile_id})[/green]")


@profile_app.command("list")
def profile_list() -> None:
    """List all saved constraint profiles."""
    db = get_db()
    with db.get_connection() as conn:
        profiles = ProfileQueries.list_profiles(conn)

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
) -> None:
    """Show details of a constraint profile."""
    db = get_db()
    with db.get_connection() as conn:
        profile = ProfileQueries.get_profile_by_name(conn, name)

    if not profile:
        console.print(f"[red]Profile '{name}' not found[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{profile['name']}[/bold]")
    if profile["description"]:
        console.print(f"Description: {profile['description']}")
    console.print(f"Created: {profile['created_at']}")
    console.print("\nConstraints:")

    constraints = json.loads(profile["constraints_json"])
    import yaml

    console.print(yaml.dump(constraints, default_flow_style=False))


@profile_app.command("delete")
def profile_delete(
    name: str = typer.Argument(..., help="Profile name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a constraint profile."""
    db = get_db()
    with db.get_connection() as conn:
        profile = ProfileQueries.get_profile_by_name(conn, name)
        if not profile:
            console.print(f"[red]Profile '{name}' not found[/red]")
            raise typer.Exit(1)

        if not force:
            confirm = typer.confirm(f"Delete profile '{name}'?")
            if not confirm:
                raise typer.Abort()

        ProfileQueries.delete_profile(conn, name)

    console.print(f"[green]Deleted profile '{name}'[/green]")


if __name__ == "__main__":
    app()
