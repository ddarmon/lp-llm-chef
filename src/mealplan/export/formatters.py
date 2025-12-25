"""Output formatters for optimization results."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mealplan.optimizer.models import OptimizationResult


class TableFormatter:
    """Format results as Rich tables for terminal display."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the formatter.

        Args:
            console: Rich console for output. If None, creates a new one.
        """
        self.console = console or Console()

    def format(
        self,
        result: OptimizationResult,
        profile_name: Optional[str] = None,
    ) -> None:
        """Print formatted tables to console.

        Args:
            result: Optimization result to format
            profile_name: Optional profile name to display
        """
        # Header panel
        status_color = "green" if result.success else "red"
        header_lines = [
            f"[bold]OPTIMIZATION RESULT[/bold] - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        if profile_name:
            header_lines.append(f"Profile: {profile_name}")
        header_lines.append(f"Status: [{status_color}]{result.status.upper()}[/{status_color}]")

        self.console.print(Panel("\n".join(header_lines), title="Meal Plan"))

        if not result.success:
            self.console.print(f"[red]Error: {result.message}[/red]")
            return

        # Food allocation table
        food_table = Table(title="Daily Food Allocation")
        food_table.add_column("Food", style="cyan", max_width=50)
        food_table.add_column("Grams", justify="right")
        food_table.add_column("Cost", justify="right", style="green")

        total_grams = 0.0
        for food in result.foods:
            food_table.add_row(
                food.description[:50],
                f"{food.grams:.1f}",
                f"${food.cost:.2f}",
            )
            total_grams += food.grams

        food_table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_grams:.1f}[/bold]",
            f"[bold]${result.total_cost:.2f}[/bold]",
            style="bold",
        )

        self.console.print(food_table)

        # Nutrient summary table
        nutrient_table = Table(title="Nutrient Summary")
        nutrient_table.add_column("Nutrient")
        nutrient_table.add_column("Amount", justify="right")
        nutrient_table.add_column("Min", justify="right")
        nutrient_table.add_column("Max", justify="right")
        nutrient_table.add_column("Status", justify="center")

        for nutrient in result.nutrients:
            status = "[green]OK[/green]" if nutrient.satisfied else "[red]![/red]"
            min_str = f"{nutrient.min_constraint:.1f}" if nutrient.min_constraint else "-"
            max_str = f"{nutrient.max_constraint:.1f}" if nutrient.max_constraint else "-"

            nutrient_table.add_row(
                nutrient.name,
                f"{nutrient.amount:.1f} {nutrient.unit}",
                min_str,
                max_str,
                status,
            )

        self.console.print(nutrient_table)

        # Solver info
        if result.solver_info:
            info_parts = []
            if "elapsed_seconds" in result.solver_info:
                info_parts.append(f"Time: {result.solver_info['elapsed_seconds']:.3f}s")
            if "iterations" in result.solver_info and result.solver_info["iterations"]:
                info_parts.append(f"Iterations: {result.solver_info['iterations']}")
            if "solver" in result.solver_info:
                info_parts.append(f"Solver: {result.solver_info['solver']}")
            if info_parts:
                self.console.print(f"[dim]{' | '.join(info_parts)}[/dim]")


class JSONFormatter:
    """Format results as JSON for programmatic use."""

    def format(
        self,
        result: OptimizationResult,
        profile_name: Optional[str] = None,
        run_id: Optional[int] = None,
    ) -> str:
        """Return JSON string.

        Args:
            result: Optimization result to format
            profile_name: Optional profile name
            run_id: Optional run ID

        Returns:
            JSON string
        """
        data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "profile": profile_name,
            "status": result.status,
            "success": result.success,
            "message": result.message,
            "solution": {
                "foods": [
                    {
                        "fdc_id": f.fdc_id,
                        "description": f.description,
                        "grams": round(f.grams, 1),
                        "cost": round(f.cost, 2),
                    }
                    for f in result.foods
                ],
                "total_cost": round(result.total_cost, 2) if result.total_cost else None,
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
            },
            "solver_info": result.solver_info,
        }
        return json.dumps(data, indent=2)


class MarkdownFormatter:
    """Format results as Markdown for LLM handoff or documentation."""

    def format(
        self,
        result: OptimizationResult,
        profile_name: Optional[str] = None,
        days: int = 1,
    ) -> str:
        """Return Markdown string.

        Args:
            result: Optimization result to format
            profile_name: Optional profile name
            days: Number of days for meal plan (for scaling display)

        Returns:
            Markdown string
        """
        lines = [
            "# Optimized Daily Food Allocation",
            "",
        ]

        if profile_name:
            lines.append(f"**Profile:** {profile_name}")
        if result.total_cost:
            lines.append(f"**Daily Cost:** ${result.total_cost:.2f}")

        # Find calories
        for n in result.nutrients:
            if n.name.lower() == "energy" or n.nutrient_id == 1008:
                lines.append(f"**Calories:** {n.amount:.0f} kcal")
                break

        lines.extend(["", "## Foods", "", "| Food | Amount | Cost |", "|------|--------|------|"])

        for food in result.foods:
            lines.append(f"| {food.description} | {food.grams:.0f}g | ${food.cost:.2f} |")

        lines.extend(
            [
                "",
                "## Nutrient Summary",
                "",
                "| Nutrient | Amount | Target |",
                "|----------|--------|--------|",
            ]
        )

        for n in result.nutrients:
            target = ""
            if n.min_constraint and n.max_constraint:
                target = f"{n.min_constraint:.0f}-{n.max_constraint:.0f}"
            elif n.min_constraint:
                target = f">= {n.min_constraint:.0f}"
            elif n.max_constraint:
                target = f"<= {n.max_constraint:.0f}"

            status = "" if n.satisfied else " (!)"
            lines.append(f"| {n.name} | {n.amount:.1f} {n.unit}{status} | {target} |")

        return "\n".join(lines)


def format_result(
    result: OptimizationResult,
    output_format: str = "table",
    profile_name: Optional[str] = None,
    run_id: Optional[int] = None,
    console: Optional[Console] = None,
) -> Optional[str]:
    """Format optimization result in the specified format.

    Args:
        result: Optimization result to format
        output_format: One of 'table', 'json', 'markdown'
        profile_name: Optional profile name
        run_id: Optional run ID
        console: Rich console (for table format)

    Returns:
        Formatted string for json/markdown, None for table (prints directly)
    """
    if output_format == "table":
        formatter = TableFormatter(console)
        formatter.format(result, profile_name)
        return None
    elif output_format == "json":
        formatter = JSONFormatter()
        return formatter.format(result, profile_name, run_id)
    elif output_format == "markdown":
        formatter = MarkdownFormatter()
        return formatter.format(result, profile_name)
    else:
        raise ValueError(f"Unknown output format: {output_format}")
