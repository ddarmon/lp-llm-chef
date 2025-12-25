# lp-llm-chef

A personal meal planning system that uses linear programming to find the
minimum-cost combination of foods that satisfies your nutritional
constraints. Optionally integrates with Claude to generate recipes from
the optimized food list.

## Features

-   **Nutritional optimization**: Specify calorie targets, macro/micro
    nutrient minimums and maximums
-   **Cost minimization**: Finds the cheapest diet that meets all your
    constraints
-   **USDA nutrition data**: Uses the comprehensive FoodData Central
    database (\~10,000 foods)
-   **Constraint profiles**: Save and reuse different dietary goals
    (cutting, bulking, maintenance)
-   **LLM integration**: Export results as prompts for Claude to
    generate meal plans and recipes
-   **Food tagging**: Tag foods for filtering (breakfast, protein,
    exclude, etc.)

## Installation

Requires Python 3.9+.

``` bash
git clone https://github.com/yourusername/lp-llm-chef.git
cd lp-llm-chef
pip install -e .
```

For development:

``` bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Download USDA Data

Download the FoodData Central CSV files from
https://fdc.nal.usda.gov/download-datasets.html

Get the "Full Download of All Data Types" in CSV format and extract it.

### 2. Initialize the Database

``` bash
mealplan init ~/Downloads/FoodData_Central_csv_2024-04-18/
```

This loads \~10,000 foods from Foundation Foods and SR Legacy datasets
into a local SQLite database at `~/.mealplan/mealplan.db`.

### 3. Add Prices for Foods You Eat

The optimizer needs prices to minimize cost. Add them for foods you
actually eat:

``` bash
# Search for a food
mealplan search "chicken breast"
# Found: 171077 - Chicken, broilers or fryers, breast, skinless, boneless...

# Add price (per 100g)
mealplan prices add 171077 0.80 --source costco

# Or import from CSV
mealplan prices import my_prices.csv
```

Price CSV format:

``` csv
fdc_id,price_per_100g,price_source,notes
171077,0.80,costco,chicken breast bulk
170148,0.15,costco,brown rice
```

### 4. Run Optimization

``` bash
# With default constraints (2000 cal, basic macros)
mealplan optimize

# With a constraint profile
mealplan optimize --profile cutting

# Output as JSON or Markdown
mealplan optimize --output json
mealplan optimize --output markdown
```

### 5. Generate Recipes with Claude

``` bash
# Export the last optimization as an LLM prompt
mealplan export-for-llm latest --output meal_request.md

# Then paste the contents into Claude to get recipes
```

## Constraint Profiles

Create YAML files to define your dietary constraints:

``` yaml
# ~/.mealplan/constraints/cutting.yaml
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
  use_quadratic_penalty: true  # encourages variety
```

Save the profile:

``` bash
mealplan profile create cutting --from-file cutting.yaml
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
mealplan tags add 171077 protein
mealplan tags add 171077 meal-prep

# Exclude foods
mealplan tags add 12345 exclude

# List foods with a tag
mealplan tags list --tag protein
```

Use `exclude_tags` and `include_tags` in profiles to filter foods.

## Commands Reference

  ----------------------------------------------------------------------------
  Command                            Description
  ---------------------------------- -----------------------------------------
  `mealplan init <path>`             Initialize database from USDA CSV
                                     directory

  `mealplan search <query>`          Search foods by description

  `mealplan info <fdc_id>`           Show nutrients for a food

  `mealplan optimize`                Run optimization

  `mealplan export-for-llm <id>`     Generate LLM prompt (use `latest` for
                                     last run)

  `mealplan history`                 Show past optimization runs

  `mealplan prices add`              Add/update a food price

  `mealplan prices import`           Import prices from CSV

  `mealplan prices list`             List priced foods

  `mealplan prices list --missing`   List foods without prices

  `mealplan tags add`                Add tag to food

  `mealplan tags remove`             Remove tag from food

  `mealplan tags list`               List all tags or foods with tag

  `mealplan profile create`          Create profile from YAML

  `mealplan profile list`            List saved profiles

  `mealplan profile show`            Show profile details
  ----------------------------------------------------------------------------

## Example Workflow

``` bash
# 1. Set up (one time)
mealplan init ~/Downloads/FoodData_Central_csv/

# 2. Add prices for ~20-50 foods you commonly eat
mealplan search "eggs"
mealplan prices add 173424 0.40 --source costco --notes "large eggs"
mealplan search "salmon"
mealplan prices add 175167 1.20 --source costco
# ... repeat for your staple foods

# 3. Tag foods (optional)
mealplan tags add 173424 breakfast
mealplan tags add 175167 dinner

# 4. Create a constraint profile
cat > cutting.yaml << 'EOF'
calories:
  min: 1700
  max: 1900
nutrients:
  protein:
    min: 140
  fiber:
    min: 25
EOF
mealplan profile create cutting --from-file cutting.yaml

# 5. Run optimization
mealplan optimize --profile cutting

# 6. Get recipes from Claude
mealplan export-for-llm latest --days 7 --output ~/Desktop/meal_plan.md
# Open meal_plan.md and paste into Claude
```

## How It Works

The optimizer solves a linear program:

**Minimize**: Total daily cost = Σ (price per gram × grams of food)

**Subject to**:

-   Calorie target: min ≤ total calories ≤ max
-   Nutrient minimums: protein ≥ 150g, fiber ≥ 30g, etc.
-   Nutrient maximums: sodium ≤ 2300mg, etc.
-   Per-food limits: 0 ≤ each food ≤ 500g (configurable)

With `use_quadratic_penalty: true`, a small penalty for deviating from
"typical" portions is added, which encourages variety instead of extreme
solutions (e.g., eating 2kg of one cheap food).

## Configuration

Settings are stored in `~/.mealplan/`:

    ~/.mealplan/
    ├── config.yaml           # App settings
    ├── mealplan.db          # SQLite database
    ├── preferences.yaml     # LLM prompt preferences
    ├── llm_prompt_template.md  # Custom prompt template
    └── constraints/         # Your constraint profiles

## Development

``` bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/

# Type check
mypy src/mealplan/
```

## License

MIT
