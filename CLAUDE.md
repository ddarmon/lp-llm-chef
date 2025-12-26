# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project Overview

`lp-llm-chef` is a meal planning optimization system designed to be used
both by humans and as a **tool for LLM agents**. It uses quadratic
programming (QP) to find diverse food combinations satisfying nutritional
constraints, with USDA FoodData Central as its nutrition database.

**Key capability**: All CLI commands support `--json` output with a
structured response envelope, making this tool usable by Claude Code,
Codex CLI, and other LLM agents for iterative diet planning.

### Optimization Modes

-   **Feasibility mode** (default): Minimizes `||x - x̄||²` (deviation
    from typical 100g portions). Produces diverse, recipe-friendly food
    lists. No prices required.
-   **Cost minimization mode** (`--minimize-cost`): Minimizes
    `λ₁·cost + λ₂·||x - x̄||²`. Requires foods to have prices set.

## Development Commands

``` bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_solver.py -v

# Run a specific test
pytest tests/test_solver.py::TestSolveLP::test_simple_optimization -v

# Lint with ruff
ruff check src/

# Type check with mypy
mypy src/mealplan/
```

## Quick Start Scripts

Interactive scripts for new users:

``` bash
# Full interactive setup (diet type, style, goals, auto-tag staples)
./scripts/setup-meal-plan.sh

# Tag staple foods interactively with bulk category support
./scripts/tag-staple-foods.sh
```

## CLI Usage

### Core Commands
``` bash
uv run mealplan init <usda-csv-path>      # Initialize database from USDA data
uv run mealplan search <query>             # Search foods (shows price/tags)
uv run mealplan info <fdc_id>              # Show food nutrients
uv run mealplan optimize                   # Run optimization (feasibility mode)
uv run mealplan optimize --minimize-cost   # Run with cost minimization
uv run mealplan optimize --max-foods 500   # Increase food pool (default 300)
uv run mealplan optimize --profile <name>  # Use a constraint profile
uv run mealplan optimize --file <yaml>     # Use YAML file directly
uv run mealplan optimize --verbose         # Show KKT optimality conditions
uv run mealplan export-for-llm latest      # Generate Claude prompt from last run
uv run mealplan prices add <fdc_id> <price>
uv run mealplan tags add <fdc_id> <tag>
uv run mealplan tags list --tag <tag>      # List foods with a tag
uv run mealplan tags interactive           # Interactive mode to search and tag
uv run mealplan profile create <name> --from-file <yaml>
uv run mealplan profile wizard             # Interactive profile creation wizard
```

### Agent/LLM Commands

Schema export (for LLMs to understand constraint vocabulary):
``` bash
uv run mealplan schema constraints         # JSON schema for constraint format
uv run mealplan schema nutrients           # All nutrients with IDs, units, ranges
uv run mealplan schema tags                # All tags in database
uv run mealplan schema all                 # Complete schema documentation
```

Food exploration (for iterative diet design):
``` bash
uv run mealplan explore foods "salmon" --min-protein 25 --json
uv run mealplan explore high-nutrient protein --min 30 --tag staple
uv run mealplan explore compare 170567 170568 --nutrients protein,fat
uv run mealplan explore whatif --base latest --add "fiber:min:40" --json
```

JSON output (add `--json` to any command):
``` bash
uv run mealplan search chicken --json      # Structured search results
uv run mealplan optimize --json            # Solution + suggestions + diagnosis
uv run mealplan info 170567 --json         # Nutrient data as JSON
```

## Architecture

### Data Flow

1.  **USDA CSV → SQLite**: `USDALoader` parses USDA CSV files into
    SQLite tables
2.  **Request → Constraints**: `ConstraintBuilder` transforms an
    `OptimizationRequest` + database into numpy arrays (costs, nutrient
    matrix, bounds)
3.  **Constraints → Solution**: `solve_lp()` or `solve_qp()` finds
    optimal food quantities using scipy.optimize
4.  **Solution → Output**: Formatters produce table/JSON/markdown;
    `LLMPromptGenerator` creates Claude-ready prompts

### Key Modules

-   **`optimizer/solver.py`**: Core LP (`scipy.linprog` with HiGHS) and QP
    (`qpsolvers` with Clarabel interior-point) solvers. Dispatches based
    on `request.mode`:
    -   `feasibility`: QP with `λ_cost=0`, only minimizes deviation
    -   `minimize_cost`: QP with `λ_cost>0` or pure LP

-   **`optimizer/constraints.py`**: `ConstraintBuilder` queries the
    database for eligible foods, builds the nutrient matrix (nutrients
    per gram), and converts nutrient constraints to numpy arrays.
    In feasibility mode, all active foods are eligible; in cost mode,
    only foods with prices. Randomly samples to `max_foods` if exceeded.

-   **`db/queries.py`**: Query classes (`FoodQueries`, `PriceQueries`,
    `TagQueries`, `ProfileQueries`, `OptimizationRunQueries`)
    encapsulate all SQL operations.

-   **`data/nutrient_ids.py`**: Maps friendly names ("protein",
    "vitamin_d") to USDA nutrient IDs (1003, 1114).

-   **`export/llm_prompt.py`**: `LLMPromptGenerator` creates Claude-ready
    prompts with:
    -   Enhanced food names with prep state: `Salmon (raw)`, `Lentils (cooked)`
    -   Full macro columns: Amount, Kcal, Protein, Carbs, Fat
    -   Auto-generated preparation notes based on detected states
    -   `clean_food_name()` simplifies USDA names for CLI display
    -   `clean_food_name_with_context()` preserves prep state for LLM prompts

-   **`agent/`**: Agentic interface layer for LLM tool usage:
    -   `response.py`: `AgentResponse` envelope with success, data, errors,
        warnings, suggestions, and human_summary fields
    -   `schema.py`: Schema export functions for constraint format, nutrient
        vocabulary, tag lists, and food filter options

-   **`explore/`**: Food discovery and iterative analysis:
    -   `foods.py`: `explore_foods()` with filters, `compare_foods()`,
        `find_high_nutrient_foods()`
    -   `whatif.py`: `run_whatif_analysis()` for modifying constraints from
        a baseline optimization run
    -   `diagnosis.py`: `diagnose_infeasibility()` for explaining why
        optimization fails and suggesting fixes

### Database Schema

8 tables in SQLite: `foods`, `nutrients`, `food_nutrients`, `prices`,
`servings`, `food_tags`, `constraint_profiles`, `optimization_runs`. All
nutrient values stored per 100g (USDA standard).

### Constraint Profiles (YAML)

Profiles define calorie ranges, nutrient min/max constraints, tag
filters, and optimizer settings:

``` yaml
calories:
  min: 1800
  max: 2200
nutrients:
  protein:
    min: 150
  sodium:
    max: 2300
# Use include_tags to limit to foods you've tagged as "staple"
# This prevents baby food, exotic meats, etc. from appearing
include_tags:
  - staple
exclude_tags:
  - exclude
  - junk_food
options:
  mode: feasibility       # or "minimize_cost"
  max_foods: 300          # randomly sample if more eligible
  max_grams_per_food: 500
  lambda_deviation: 0.001
```

Example profiles in `examples/constraints/`:
- `cutting.yaml` - Weight loss (1600-1800 cal, 150g protein)
- `bulking.yaml` - Muscle gain (2800-3200 cal, 180g protein)
- `maintenance.yaml` - Balanced (2200-2400 cal, micronutrient focus)
- `slowcarb_pescatarian.yaml` - Tim Ferriss style + fish only

## Key Design Decisions

-   **Raw sqlite3 over SQLAlchemy**: Simpler, fewer dependencies, direct
    SQL control
-   **qpsolvers with Clarabel for QP**: True interior-point solver provides
    high-precision solutions (~1e-11 stationarity) and full dual multipliers
    including bound constraints. `scipy.linprog` with HiGHS for LP.
-   **All values per 100g**: Matches USDA data format; divide by 100
    when building per-gram matrices
-   **`from __future__ import annotations`**: Required in all files for
    forward reference compatibility
-   **KKT analysis**: The `--verbose` flag displays Karush-Kuhn-Tucker
    optimality conditions (primal/dual feasibility, complementary
    slackness, stationarity) with full Lagrange multipliers for all
    binding constraints including variable bounds

## Using as an LLM Tool

This tool is designed to be invoked by LLM agents (Claude Code, Codex CLI,
etc.) for iterative diet planning. The workflow:

1.  **Get schema**: `uv run mealplan schema all` returns constraint vocabulary
2.  **LLM translates goals**: User says "I want to lose weight, vegetarian"
    → LLM outputs structured constraints JSON
3.  **Optimize**: `uv run mealplan optimize --json` returns solution + suggestions
4.  **Iterate**: `uv run mealplan explore whatif --add "protein:min:180" --json`
5.  **Diagnose failures**: If infeasible, response includes diagnosis with
    suggested constraint relaxations

### JSON Response Envelope

All `--json` output uses this structure:
```json
{
  "success": true,
  "command": "optimize",
  "data": { ... },
  "errors": [],
  "warnings": [],
  "suggestions": ["Protein is at minimum - consider relaxing"],
  "human_summary": "Optimal: 12 foods, 1847 kcal, $8.50/day"
}
```

The `human_summary` field ensures transparency for the person watching.
