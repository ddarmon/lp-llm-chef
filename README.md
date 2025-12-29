# llmn 🍋

**llmn** (pronounced "lemon") = **LLM** + **N**utrition

Diet optimization and weight tracking for humans and AI agents.

- **Meal planning**: Quadratic programming finds diverse food combinations
  meeting calorie/macro/micro targets
- **Weight tracking**: Hacker's Diet-style EMA smoothing filters daily noise
- **Adaptive TDEE**: Kalman filter learns your true metabolism from weight
  observations, improving on generic formulas
- **LLM-first**: All commands support `--json` for tool use with Claude Code,
  Codex CLI, etc.

## Features

-   **Nutritional optimization**: Specify calorie targets, macro/micro
    nutrient minimums and maximums
-   **Two optimization modes**:
    -   **Feasibility mode** (default): Finds diverse, nutritionally
        complete food combinations without considering cost
    -   **Cost minimization mode**: Finds the cheapest diet that meets
        all constraints
-   **USDA nutrition data**: Uses the comprehensive FoodData Central
    database (\~10,000 foods)
-   **Constraint profiles**: Save and reuse different dietary goals
    (cutting, bulking, maintenance)
-   **LLM integration**: Export results as prompts for Claude to
    generate meal plans and recipes
-   **Food tagging**: Tag foods for filtering (breakfast, protein,
    exclude, etc.)
-   **Agentic interface**: All commands support `--json` output with
    structured responses for LLM tool use
-   **Schema export**: LLMs can query available nutrients, tags, and
    constraint format
-   **What-if analysis**: Iterate on solutions by modifying constraints
-   **Infeasibility diagnosis**: When optimization fails, explains why
    and suggests fixes
-   **Meta-optimization**: Batch optimize multiple food pools, generate
    pool suggestions, limit food count for meal-prep
-   **Meal allocation**: Distribute foods into breakfast/lunch/dinner/snack
    slots using heuristics
-   **Multi-period optimization**: Per-meal calorie/nutrient constraints
    enforced at optimization time (not post-hoc). Includes equi-calorie
    constraints and food-meal affinity rules.
-   **Template-based meal composition** (`--template`): Human-like meal planning
    that selects one protein + one legume + vegetables per meal, then optimizes
    quantities. Produces realistic meals with different foods at each meal.
-   **Food categories**: Classify foods by macro dominance (protein, carb,
    fat, vegetable, legume)
-   **Weight tracking**: Hacker's Diet-style exponentially smoothed moving
    average (EMA) for trend tracking that filters out daily noise
-   **Adaptive TDEE learning**: Kalman filter learns your personalized TDEE
    from weight observations over time, improving on generic Mifflin-St Jeor
    estimates

## Installation

Requires Python 3.11+ and scipy 1.16+.

``` bash
git clone https://github.com/ddarmon/llmn.git
cd llmn

# Using uv (recommended)
uv sync
uv run llmn --help

# Or with pip
pip install -e .
```

For development:

``` bash
uv sync
uv pip install pytest ruff mypy

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

### Option A: Interactive Setup (Recommended)

Run the interactive setup script that guides you through everything:

``` bash
./scripts/setup-meal-plan.sh
```

This will:
1. Install the CLI if needed
2. Help you download and load USDA data
3. Ask about your diet type (omnivore, pescatarian, vegetarian, vegan)
4. Ask about your diet style (standard, slow-carb, low-carb, high-protein, Mediterranean)
5. Set your calorie and protein targets
6. Auto-tag recommended staple foods for your diet
7. Run your first optimization
8. Export a prompt for Claude to generate recipes

### Option B: Manual Setup

#### 1. Download USDA Data

Download the FoodData Central CSV files from
https://fdc.nal.usda.gov/download-datasets.html

Get the "Full Download of All Data Types" in CSV format and extract it.

#### 2. Initialize the Database

``` bash
uv run llmn init ~/Downloads/FoodData_Central_csv_2024-04-18/
```

This loads \~10,000 foods from Foundation Foods and SR Legacy datasets
into a local SQLite database at `~/.llmn/llmn.db`.

#### 3. Run Optimization

``` bash
# Default: feasibility mode (finds diverse, nutritionally complete foods)
uv run llmn optimize

# With a constraint profile
uv run llmn optimize --profile cutting

# Limit number of foods considered (default 300, for speed)
uv run llmn optimize --max-foods 500

# Output as JSON or Markdown
uv run llmn optimize --output json
uv run llmn optimize --output markdown

# Show KKT optimality conditions (verify solution is optimal)
uv run llmn optimize --verbose
```

### 4. Add Prices (Optional, for Cost Minimization)

Prices are only needed if you want to minimize cost. In the default
feasibility mode, all foods in the database are eligible.

``` bash
# Search for a food
uv run llmn search "chicken breast"
# Found: 171077 - Chicken, broilers or fryers, breast, skinless, boneless...

# Add price (per 100g)
uv run llmn prices add 171077 0.80 --source costco

# Or import from CSV
uv run llmn prices import my_prices.csv

# Run with cost minimization
uv run llmn optimize --minimize-cost
```

Price CSV format:

``` csv
fdc_id,price_per_100g,price_source,notes
171077,0.80,costco,chicken breast bulk
170148,0.15,costco,brown rice
```

### 5. Generate Recipes with Claude

``` bash
# Export the last optimization as an LLM prompt
uv run llmn export-for-llm latest --days 7 --output meal_request.md

# Then paste the contents into Claude to get recipes
```

The exported prompt includes:
- **Food names with preparation context**: `Salmon (raw)`, `Black beans (cooked)`, `Sardines (canned)`
- **Full macros per food**: Amount, Kcal, Protein, Carbs, Fat
- **Preparation notes**: Guidance on cooking raw items, using canned/cooked items

Example output:
```
| Food | Amount (g) | Kcal | Protein (g) | Carbs (g) | Fat (g) |
|------|------------|------|-------------|-----------|---------|
| Salmon (raw) | 75 | 156 | 15 | 0 | 10 |
| Black beans (cooked) | 85 | 111 | 7 | 21 | 0 |
| Eggs (raw) | 80 | 114 | 10 | 1 | 8 |
```

This gives Claude complete nutritional context to generate accurate recipes.

## Constraint Profiles

Create YAML files to define your dietary constraints:

``` yaml
# ~/.llmn/constraints/cutting.yaml
name: cutting
description: "Weight loss: 1600-1800 cal, high protein"

calories:
  min: 1600
  max: 1800

nutrients:
  protein:
    min: 150      # grams
  fiber:
    min: 30
  sodium:
    max: 2300     # mg
  saturated_fat:
    max: 15

exclude_tags:
  - junk_food
  - high_sugar

options:
  max_grams_per_food: 400
  max_foods: 300            # limit foods for performance
  mode: feasibility         # or "minimize_cost"
```

Save the profile:

``` bash
uv run llmn profile create cutting --from-file cutting.yaml
```

**Interactive Wizard**: You can also create profiles interactively:

``` bash
uv run llmn profile wizard
```

Available nutrient names: `protein`, `carbohydrate`, `total_fat`,
`fiber`, `sugar`, `saturated_fat`, `sodium`, `calcium`, `iron`,
`magnesium`, `potassium`, `zinc`, `vitamin_a`, `vitamin_c`, `vitamin_d`,
`vitamin_e`, `vitamin_k`, `vitamin_b6`, `vitamin_b12`, `folate`,
`thiamin`, `riboflavin`, `niacin`

## Food Tags

Tag foods for filtering in optimization:

``` bash
# Add tags
uv run llmn tags add 171077 protein
uv run llmn tags add 171077 meal-prep

# Exclude foods
uv run llmn tags add 12345 exclude

# List foods with a tag
uv run llmn tags list --tag protein
```

**Interactive Tagging**: Quickly search and tag foods by number:

``` bash
uv run llmn tags interactive
```

Use `exclude_tags` and `include_tags` in profiles to filter foods.

## Staple Foods Workflow (Recommended)

By default, the optimizer considers all ~10,000 foods in the USDA database,
which can produce impractical suggestions (baby food, exotic meats, etc.).

**Solution**: Tag foods you actually buy as "staple", then use `include_tags`
in your profile to only optimize across those foods.

### Interactive Tagging

``` bash
./scripts/tag-staple-foods.sh
```

This provides an interactive shell with bulk category commands:

```
> c fish           # Show common fish options
> c fish tag       # Tag ALL fish as staples
> c legumes tag    # Tag all beans, lentils, chickpeas
> c veggies tag    # Tag common vegetables
> s salmon         # Search for a specific food
> t 175167         # Tag a specific FDC ID
> l                # List current staples
```

Available categories: `fish`, `seafood`, `meat`, `legumes`, `veggies`,
`greens`, `carbs`, `fruits`, `fats`, `dairy`, `eggs`

### Using Staples in Profiles

``` yaml
# Only use foods tagged as "staple"
include_tags:
  - staple

exclude_tags:
  - exclude
```

This ensures optimization only suggests foods you actually buy and cook with.

## Commands Reference

All commands should be prefixed with `uv run` when using `uv` as the project manager.

| Command | Description |
|--------|-------------|
| `llmn init <path>` | Initialize database from USDA CSV directory |
| `llmn search <query>` | Search foods (shows price/tags) |
| `llmn info <fdc_id>` | Show nutrients for a food |
| `llmn optimize` | Run optimization |
| `llmn optimize --template` | Template-based (realistic meals) |
| `llmn optimize --multiperiod` | Multi-period mode (per-meal constraints) |
| `llmn optimize --verbose` | Show KKT optimality conditions |
| `llmn export-for-llm <id>` | Generate LLM prompt (use `latest` for last run) |
| `llmn history` | Show past optimization runs |
| `llmn prices add` | Add/update a food price |
| `llmn prices import` | Import prices from CSV |
| `llmn prices list` | List priced foods |
| `llmn prices list --missing` | List foods without prices |
| `llmn tags add` | Add tag to food |
| `llmn tags remove` | Remove tag from food |
| `llmn tags list` | List all tags or foods with tag |
| `llmn tags interactive` | Interactive mode to search/tag |
| `llmn profile create` | Create profile from YAML |
| `llmn profile wizard` | Interactive profile creator |
| `llmn profile list` | List saved profiles |
| `llmn profile show` | Show profile details |
| `llmn user create` | Create user profile (age, sex, height, activity level) |
| `llmn user show` | Show current user profile |
| `llmn weight add <lbs>` | Log today's weight (computes EMA trend automatically) |
| `llmn weight list` | Show weight history with trends |
| `llmn calories log <kcal>` | Log planned calorie intake |
| `llmn tdee estimate` | Run Kalman filter to estimate personalized TDEE |
| `llmn tdee progress` | Show weight/TDEE progress report |

## Advanced LLM Features

These features are designed for LLM agents doing iterative diet optimization.

**Claude Code users**: This project includes slash commands (`/llmn`, `/template`,
`/multiperiod`, `/recipes`, `/tracking`) for interactive meal planning and weight tracking.
See `.claude/skills/` for details.

### Meta-Optimization

``` bash
# Generate food pool suggestions (balanced, high-protein, budget)
uv run llmn explore suggest-pools --json

# Optimize with explicit food IDs (bypass tag filtering)
uv run llmn optimize --foods 175167,171287,172421 --json

# Batch optimize multiple food pools at once
uv run llmn optimize-batch pools.json --json

# Limit solution to N foods (meal-prep friendly)
uv run llmn optimize --max-foods-in-solution 10 --json
```

### Meal Allocation (Post-hoc)

``` bash
# Distribute foods into breakfast/lunch/dinner/snack
uv run llmn optimize --allocate-meals --json
```

This uses keyword heuristics (eggs→breakfast, fish→lunch/dinner, nuts→snack)
to distribute the optimization result into meal slots.

### Template-Based Meal Composition (Recommended)

For realistic meals that look like what humans actually eat:

``` bash
# Template-based optimization (recommended)
uv run llmn optimize --pattern pescatarian --pattern slow_carb --template

# With reproducible random seed
uv run llmn optimize --pattern pescatarian --template --seed 42

# Available patterns: pescatarian, vegetarian, vegan, keto, mediterranean, paleo, slow_carb
```

This produces meals with proper structure:
- **Breakfast**: Eggs 150g, Edamame 100g, Kale 100g
- **Lunch**: Cod 235g, Kidney beans 216g, Zucchini 161g
- **Dinner**: Salmon 170g, Lentils 113g, Carrots 149g
- **Snack**: Peanuts 30g

Instead of the Stigler-style output (all vegetables, identical meals).

### Multi-Period Optimization (Per-Meal Constraints)

For per-meal constraints enforced at optimization time (Stigler-style QP):

``` bash
# Auto-derive meal targets (25% breakfast, 35% lunch, 35% dinner, 5% snack)
uv run llmn optimize --multiperiod --json

# Use a profile with meals: section
uv run llmn optimize --file meals_profile.yaml --json
```

Multi-period profiles define per-meal constraints:

``` yaml
calories:
  min: 1800
  max: 2000

meals:
  breakfast:
    calories: {min: 400, max: 550}
    nutrients:
      protein: {min: 30}
  lunch:
    calories: {min: 550, max: 700}
  dinner:
    calories: {min: 550, max: 700}
  snack:
    calories: {min: 50, max: 150}  # No more 995-calorie snacks!

# Optional: balance lunch and dinner
equicalorie:
  - meals: [lunch, dinner]
    tolerance: 100
```

This prevents issues like the optimizer putting all nuts into a single
995-calorie "snack" by enforcing constraints at optimization time.

### Run Comparison

``` bash
# List recent optimization runs
uv run llmn explore runs

# Compare two runs side-by-side
uv run llmn explore compare-runs 5 8 --json
```

### Food Categories

Filter foods by macro-based category:

``` bash
# Find high-protein foods
uv run llmn explore foods "chicken" --category protein --json

# Categories: protein, fat, carb, vegetable, legume, fruit, mixed
```

## Example Profiles

Example constraint profiles are provided in `examples/constraints/`:

| Profile | Description |
|---------|-------------|
| `cutting.yaml` | Weight loss: 1600-1800 cal, 150g protein |
| `bulking.yaml` | Muscle gain: 2800-3200 cal, 180g protein |
| `maintenance.yaml` | Balanced: 2200-2400 cal, micronutrient focus |
| `slowcarb_pescatarian.yaml` | Tim Ferriss style slow-carb + fish only |
| `staples_only.yaml` | Template using only tagged staple foods |

Use directly with `--file`:

``` bash
uv run llmn optimize --file examples/constraints/cutting.yaml
```

## Example Workflow

``` bash
# 1. Set up with interactive script (recommended)
./scripts/setup-meal-plan.sh

# OR manual setup:

# 1. Initialize database (one time)
uv run llmn init ~/Downloads/FoodData_Central_csv/

# 2. Tag your staple foods
./scripts/tag-staple-foods.sh
# Use: c veggies tag, c fish tag, c legumes tag, etc.

# 3. Create a constraint profile with include_tags
cat > my_diet.yaml << 'EOF'
calories:
  min: 1700
  max: 1900
nutrients:
  protein:
    min: 140
  fiber:
    min: 25
include_tags:
  - staple
exclude_tags:
  - exclude
EOF

# 4. Run optimization
uv run llmn optimize --file my_diet.yaml

# 5. Get recipes from Claude
uv run llmn export-for-llm latest --days 7 --output ~/Desktop/meal_plan.md
# Open meal_plan.md and paste into Claude
```

## How It Works

The optimizer uses quadratic programming with constraints:

**Subject to**:

-   Calorie target: min ≤ total calories ≤ max
-   Nutrient minimums: protein ≥ 150g, fiber ≥ 30g, etc.
-   Nutrient maximums: sodium ≤ 2300mg, etc.
-   Per-food limits: 0 ≤ each food ≤ 500g (configurable)

### Feasibility Mode (default)

**Minimize**: Deviation from typical portions = Σ (grams - 100)²

This finds diverse food combinations by penalizing extreme quantities.
Foods naturally cluster around 100g portions, producing recipe-friendly
results. No prices required.

### Cost Minimization Mode (`--minimize-cost`)

**Minimize**: λ₁·cost + λ₂·deviation

Finds the cheapest diet meeting constraints, with a small diversity
penalty to avoid extreme solutions (e.g., eating 2kg of one cheap food).
Requires prices to be set for foods.

### Multi-Period Mode (`--multiperiod`)

**Variables**: x_{i,m} = grams of food i in meal m

**Additional constraints**:
- Per-meal calorie bounds (e.g., snack ≤ 150 kcal)
- Per-meal nutrient bounds (e.g., breakfast protein ≥ 30g)
- Daily linking (sum of meal allocations meets daily targets)
- Equi-calorie constraints (|lunch - dinner| ≤ 100 kcal)
- Food-meal affinity (almonds only in snacks)

This enforces meal structure at optimization time, avoiding post-hoc
allocation issues.

### Template-Based Mode (`--template`)

A fundamentally different approach that mirrors human meal planning:

**Phase 1 - Selection (Discrete)**:
- For each meal, select one food per slot (protein, legume, vegetables)
- Enforce diversity: once a food is selected, exclude it from future meals
- Selection strategy: random with optional seed for reproducibility

**Phase 2 - Optimization (Continuous)**:
- Run small QP (~16-20 variables) with only selected foods
- Use slot-specific target portions (protein=175g, legume=125g, vegetables=100g)
- Much simpler optimization that naturally produces reasonable portions

This avoids the Stigler-style problem where the optimizer spreads across
many foods because that minimizes deviation from the uniform 100g target.

## Configuration

Settings are stored in `~/.llmn/`:

    ~/.llmn/
    ├── config.yaml           # App settings
    ├── llmn.db          # SQLite database
    ├── preferences.yaml     # LLM prompt preferences
    ├── llm_prompt_template.md  # Custom prompt template
    └── profiles/            # Your constraint profiles

Helper scripts in `scripts/`:

    scripts/
    ├── setup-meal-plan.sh    # Interactive first-time setup
    └── tag-staple-foods.sh   # Bulk tag staple foods by category

Example profiles in `examples/constraints/`:

    examples/constraints/
    ├── cutting.yaml              # Weight loss
    ├── bulking.yaml              # Muscle gain
    ├── maintenance.yaml          # Balanced
    ├── slowcarb_pescatarian.yaml # Slow-carb + fish
    └── staples_only.yaml         # Template with include_tags

## Development

``` bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/

# Type check
mypy src/llmn/
```

## License

MIT
