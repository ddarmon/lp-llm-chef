# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project Overview

`lp-llm-chef` is a personal meal planning optimization system that uses
quadratic programming (QP) to find diverse food combinations satisfying
nutritional constraints. It uses USDA FoodData Central as its nutrition
database and can generate prompts for Claude to create recipes.

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

``` bash
mealplan init <usda-csv-path>      # Initialize database from USDA data
mealplan search <query>             # Search foods by description
mealplan info <fdc_id>              # Show food nutrients
mealplan optimize                   # Run optimization (feasibility mode)
mealplan optimize --minimize-cost   # Run with cost minimization
mealplan optimize --max-foods 500   # Increase food pool (default 300)
mealplan optimize --profile <name>  # Use a constraint profile
mealplan optimize --file <yaml>     # Use YAML file directly
mealplan export-for-llm latest      # Generate Claude prompt from last run
mealplan prices add <fdc_id> <price>
mealplan tags add <fdc_id> <tag>
mealplan tags list --tag <tag>      # List foods with a tag
mealplan profile create <name> --from-file <yaml>
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

-   **`optimizer/solver.py`**: Core LP (`linprog` with HiGHS) and QP
    (`minimize` with SLSQP) solvers. Dispatches based on `request.mode`:
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
-   **scipy.optimize over cvxpy**: Fewer dependencies; `linprog` with
    HiGHS for LP, `minimize` with SLSQP for QP
-   **All values per 100g**: Matches USDA data format; divide by 100
    when building per-gram matrices
-   **`from __future__ import annotations`**: Required in all files for
    Python 3.9 compatibility with generic type hints
