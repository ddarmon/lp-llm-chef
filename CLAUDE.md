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
-   **Multi-period mode** (`--multiperiod`): Optimizes per-meal allocations
    with variables `x_{i,m}` (grams of food i in meal m). Enforces per-meal
    calorie/nutrient constraints at optimization time, fixing issues like
    995-calorie snacks from post-hoc allocation.
-   **Template-based mode** (`--template`): Recommended for realistic meals.
    Uses human-like meal planning: select one protein + one legume + vegetables
    per meal, then optimize quantities. Produces different foods at each meal
    instead of spreading across many foods like Stigler-style optimization.

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
mypy src/llmn/
```

## Git Workflow

Always start new work from a clean branch off main:

``` bash
# For new features
git checkout main && git pull
git checkout -b feature/your-feature-name

# For bug fixes
git checkout main && git pull
git checkout -b bugfix/your-bugfix-name
```

This prevents mixing unrelated changes and simplifies PRs.

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
uv run llmn init <usda-csv-path>      # Initialize database from USDA data
uv run llmn search <query>             # Search foods (shows price/tags)
uv run llmn info <fdc_id>              # Show food nutrients
uv run llmn optimize                   # Run optimization (feasibility mode)
uv run llmn optimize --minimize-cost   # Run with cost minimization
uv run llmn optimize --max-foods 500   # Increase food pool (default 300)
uv run llmn optimize --profile <name>  # Use a constraint profile
uv run llmn optimize --file <yaml>     # Use YAML file directly
uv run llmn optimize --verbose         # Show KKT optimality conditions
uv run llmn export-for-llm latest      # Generate Claude prompt from last run
uv run llmn prices add <fdc_id> <price>
uv run llmn tags add <fdc_id> <tag>
uv run llmn tags list --tag <tag>      # List foods with a tag
uv run llmn tags interactive           # Interactive mode to search and tag
uv run llmn profile create <name> --from-file <yaml>
uv run llmn profile wizard             # Interactive profile creation wizard
```

### Weight Tracking Commands
``` bash
uv run llmn user create --age 38 --sex male --height 72 --activity moderate
uv run llmn user update --weight 185 --goal "fat_loss:165"
uv run llmn user show                  # Show current user profile

uv run llmn weight add 184.2           # Log today's weight (EMA computed)
uv run llmn weight add 183.8 --date 2025-12-27
uv run llmn weight list --days 30      # Show weight history with trends

uv run llmn calories log 1900          # Log planned intake for today
uv run llmn calories log 1900 --date 2025-12-27

uv run llmn tdee estimate              # Run Kalman filter on accumulated data
uv run llmn tdee progress              # Show comprehensive progress report
```

### Agent/LLM Commands

Schema export (for LLMs to understand constraint vocabulary):
``` bash
uv run llmn schema constraints         # JSON schema for constraint format
uv run llmn schema nutrients           # All nutrients with IDs, units, ranges
uv run llmn schema tags                # All tags in database
uv run llmn schema all                 # Complete schema documentation
```

Food exploration (for iterative diet design):
``` bash
uv run llmn explore foods "salmon" --min-protein 25 --json
uv run llmn explore foods "chicken" --category protein --json  # Filter by macro category
uv run llmn explore high-nutrient protein --min 30 --tag staple
uv run llmn explore compare 170567 170568 --nutrients protein,fat
uv run llmn explore whatif --base latest --add "fiber:min:40" --json
uv run llmn explore suggest-pools --json           # Generate food pool suggestions
uv run llmn explore runs --limit 10                # List recent optimization runs
uv run llmn explore compare-runs 5 8 --json        # Compare two runs
```

Advanced optimization:
``` bash
uv run llmn optimize --foods 175167,171287,172421  # Use explicit food IDs
uv run llmn optimize --max-foods-in-solution 10    # Limit to N foods (sparse)
uv run llmn optimize --allocate-meals --json       # Distribute into meals (post-hoc)
uv run llmn optimize-batch pools.json --json       # Run multiple pools at once
```

Multi-period optimization (per-meal constraints):
``` bash
uv run llmn optimize --multiperiod --json          # Auto-derive meal targets (25/35/35/5%)
uv run llmn optimize --file meals.yaml --json      # Use profile with meals: section
```

Template-based optimization (recommended for realistic meals):
``` bash
uv run llmn optimize --pattern pescatarian --template --json       # Template-based
uv run llmn optimize --pattern pescatarian --template --seed 42    # Reproducible
# Patterns: pescatarian, vegetarian, vegan, keto, mediterranean, paleo, slow_carb
```

JSON output (add `--json` to any command):
``` bash
uv run llmn search chicken --json      # Structured search results
uv run llmn optimize --json            # Solution + suggestions + diagnosis
uv run llmn info 170567 --json         # Nutrient data as JSON
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
    -   `suggest.py`: `suggest_pools()` generates balanced/high-protein/budget
        food pool suggestions for LLM meta-optimization
    -   `compare.py`: `compare_runs()` for comparing optimization runs

-   **`optimizer/`**: Core optimization:
    -   `sparse_solver.py`: Two-phase heuristic for limiting food count
        (`--max-foods-in-solution`). Phase 1 runs full QP, Phase 2 re-optimizes
        with only top N foods.
    -   `batch.py`: `run_batch_optimization()` runs multiple food pools
        in parallel for LLM comparison
    -   `serialization.py`: Request round-trip serialization for whatif analysis
    -   `multiperiod_models.py`: Data models for multi-period optimization
        (MealType, MealConfig, MultiPeriodRequest, MealResult)
    -   `multiperiod_constraints.py`: Builds expanded constraint matrices with
        variables `x_{i,m}` for per-meal optimization. Handles per-meal calorie/
        nutrient constraints, daily linking, equi-calorie, food-meal affinity.
    -   `multiperiod_solver.py`: `solve_multiperiod_diet()` main entry point
        for multi-period QP optimization
    -   `multiperiod_diagnosis.py`: IIS-like (Irreducible Infeasible Subset)
        analysis for identifying conflicting constraints

-   **`data/`**: Data utilities:
    -   `food_categories.py`: Macro-based food classification (protein/fat/carb/
        vegetable/legume/fruit) using calorie percentages
    -   `quality.py`: Detects missing/inconsistent nutrient data, calculates
        fallback energy using Atwater factors (4p + 4c + 9f)
    -   `pattern_staples.py`: Curated food ID lists per dietary pattern.
        `SOURCE_MAPPING` maps slot names (eggs, fish, leafy_greens, cruciferous)
        to staple lists. Vegetable sub-categories: `LEAFY_GREENS`, `CRUCIFEROUS`,
        `OTHER_VEGETABLES`, `MUSHROOMS`.

-   **`export/`**: Output formatters:
    -   `meal_allocator.py`: Post-hoc meal distribution using keyword heuristics
        (eggs→breakfast, fish→lunch/dinner, nuts→snack)

-   **`templates/`**: Template-based meal composition (human-like meal planning):
    -   `models.py`: Data models (SlotDefinition, MealTemplate, DietTemplate,
        SelectedMeal, TemplateOptimizationResult, SelectionStrategy)
    -   `definitions.py`: Built-in templates per dietary pattern (pescatarian,
        keto, vegan, vegetarian, mediterranean, paleo, pescatarian_slowcarb)
    -   `selector.py`: Food selection with diversity tracking. `select_foods_for_template()`
        picks one food per slot per meal, excluding already-used foods.
    -   `optimizer.py`: Small QP (~16-20 variables) with slot-aware target portions.
        `run_template_optimization()` is the main entry point.

-   **`tracking/`**: Weight tracking and adaptive TDEE learning:
    -   `models.py`: Data models (UserProfile, WeightEntry, CalorieEntry, TDEEEstimate)
    -   `queries.py`: Database queries (UserQueries, WeightQueries, CalorieQueries,
        TDEEQueries) for CRUD operations on tracking data
    -   `ema.py`: Hacker's Diet exponentially smoothed moving average (EMA) for
        trend calculation. Formula: `T_n = T_{n-1} + 0.1 × (W_n - T_{n-1})`
    -   `tdee_filter.py`: Scalar Kalman filter for learning TDEE bias. Updates
        weekly based on implied deficit (from weight trend) vs expected deficit
        (from planned calories). Converges after 2-3 weeks of data.
    -   `diagnostics.py`: Progress reports comparing Mifflin-St Jeor baseline
        with learned TDEE, trend analysis, goal progress estimation

### Database Schema

12 tables in SQLite:
- Core: `foods`, `nutrients`, `food_nutrients`, `prices`, `servings`, `food_tags`
- Optimization: `constraint_profiles`, `optimization_runs`
- Tracking: `user_profiles`, `weight_log`, `calorie_log`, `tdee_estimates`

All nutrient values stored per 100g (USDA standard).

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

### Multi-Period Profiles (per-meal constraints)

Add a `meals:` section to enable multi-period optimization:

``` yaml
calories:
  min: 1800
  max: 2000

nutrients:
  protein:
    min: 150

meals:
  breakfast:
    calories:
      min: 400
      max: 550
    nutrients:
      protein:
        min: 30
  lunch:
    calories:
      min: 550
      max: 700
  dinner:
    calories:
      min: 550
      max: 700
  snack:
    calories:
      min: 50
      max: 150

# Optional: require lunch/dinner to be within 100 kcal
equicalorie:
  - meals: [lunch, dinner]
    tolerance: 100

# Optional: restrict foods to specific meals (FDC ID -> allowed meals)
food_meal_rules:
  170567: [snack]     # Almonds only in snacks
```

Profiles with `meals:` are auto-detected and routed to the multi-period solver.

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

## Claude Code Skills

This project includes Claude Code skills (slash commands) for interactive meal planning:

| Skill | Trigger | Description |
|-------|---------|-------------|
| `/llmn` | "create a meal plan", "optimize my diet" | Interactive wizard for diet planning |
| `/template` | "realistic meals", "different foods each meal" | Template-based meal composition |
| `/multiperiod` | "per meal calories", "balanced meals" | Per-meal constraint optimization |
| `/recipes` | "create recipes", "weekly menu" | Generate recipes from optimization |
| `/tracking` | "log my weight", "how am I doing", "TDEE" | Weight tracking and adaptive TDEE learning |

Skills are defined in `.claude/skills/*/SKILL.md`.

### Recommended Workflow

1. **Start with `/llmn`** - Gathers user goals and runs optimization
2. **Or use `/template` directly** - For users who want realistic meals immediately
3. **Follow with `/recipes`** - Generate practical meal plans and recipes
4. **Use `/tracking` daily** - Log weight and calories to learn personalized TDEE

## Using as an LLM Tool

This tool is designed to be invoked by LLM agents (Claude Code, Codex CLI,
etc.) for iterative diet planning.

### Basic Workflow

1.  **Get schema**: `uv run llmn schema all` returns constraint vocabulary
2.  **LLM translates goals**: User says "I want to lose weight, vegetarian"
    → LLM outputs structured constraints JSON
3.  **Optimize**: `uv run llmn optimize --json` returns solution + suggestions
4.  **Iterate**: `uv run llmn explore whatif --add "protein:min:180" --json`
5.  **Diagnose failures**: If infeasible, response includes diagnosis with
    suggested constraint relaxations

### Advanced Meta-Optimization Workflow

For LLMs that want to explore multiple diet configurations:

1.  **Template-based (recommended)**: `uv run llmn optimize --pattern pescatarian --template --json`
    Produces realistic meals with proper structure (1 protein + 1 legume + vegetables per meal)
2.  **Generate pool suggestions**: `uv run llmn explore suggest-pools --json`
    Returns balanced, high-protein, and budget food pools
3.  **Run batch optimization**: Create a pools.json and run
    `uv run llmn optimize-batch pools.json --json` to compare pools
4.  **Or use explicit foods**: `uv run llmn optimize --foods 175167,171287 --json`
    Bypass tag filtering with specific food IDs
5.  **Limit food count**: `uv run llmn optimize --max-foods-in-solution 10 --json`
    Get meal-prep friendly solutions with fewer distinct foods
6.  **Allocate to meals**: `uv run llmn optimize --allocate-meals --json`
    Distributes foods into breakfast/lunch/dinner/snack slots (post-hoc heuristic)
7.  **Multi-period optimization**: `uv run llmn optimize --multiperiod --json`
    Enforce per-meal constraints at optimization time (Stigler-style QP)
8.  **Compare runs**: `uv run llmn explore compare-runs 5 8 --json`
    See differences between optimization runs

### When to Use Each Mode

| Mode | Use When | Output Quality |
|------|----------|----------------|
| `--template` | Want realistic, cookable meals | High - proper meal structure |
| `--multiperiod` | Need exact per-meal nutrient control | Medium - may spread foods |
| `--allocate-meals` | Quick post-hoc distribution | Low - heuristic-based |
| (default) | Daily totals only | N/A - no meal structure |

### JSON Response Envelope

All `--json` output uses this structure:
```json
{
  "success": true,
  "command": "optimize",
  "data": { ... },
  "errors": [],
  "warnings": ["Broccoli: Stored energy differs from calculated"],
  "suggestions": ["Protein is at minimum - consider relaxing"],
  "human_summary": "Optimal: 12 foods, 1847 kcal, $8.50/day"
}
```

The `human_summary` field ensures transparency for the person watching.

### Data Quality Warnings

The optimizer now detects foods with missing or inconsistent nutrient data:
- Foods with zero energy but non-zero macros
- Energy values that don't match Atwater factor calculations (4p + 4c + 9f)

These appear in `warnings` to help LLMs understand data limitations.
