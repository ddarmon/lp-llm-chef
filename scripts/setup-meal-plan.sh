#!/bin/bash
# setup-meal-plan.sh - Interactive setup for meal planning optimization
#
# This script walks you through:
# 1. Installing the mealplan CLI
# 2. Downloading and loading USDA nutrition data
# 3. Selecting your diet type and style
# 4. Creating a personalized constraint profile based on your goals
# 5. Tagging staple foods for your diet
# 6. Running your first optimization
# 7. Generating a prompt for Claude to create recipes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}${BOLD}  $1${NC}"
    echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}→${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Get script directory (where this script lives)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_DIR="${HOME}/.mealplan"
PROFILES_DIR="${DATA_DIR}/profiles"

print_header "Welcome to Meal Plan Optimizer Setup"

echo "This script will help you set up personalized meal planning based on your"
echo "health goals. It uses USDA nutrition data and quadratic programming to find"
echo "diverse food combinations that meet your nutritional needs."
echo ""
echo "At the end, you'll have a food list you can give to Claude to generate recipes."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Check/Install CLI
# ─────────────────────────────────────────────────────────────────────────────

print_header "Step 1: Checking Installation"

if command -v mealplan &> /dev/null; then
    print_success "mealplan CLI is installed"
else
    print_warning "mealplan CLI not found. Installing..."
    cd "$PROJECT_DIR"
    pip install -e ".[dev]"
    print_success "Installed mealplan CLI"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Check/Download USDA Data
# ─────────────────────────────────────────────────────────────────────────────

print_header "Step 2: USDA Nutrition Database"

# Check if database already exists
if [ -f "${DATA_DIR}/mealplan.db" ]; then
    print_success "Database already exists at ${DATA_DIR}/mealplan.db"
    echo ""
    read -p "Do you want to reinitialize the database? (y/N): " reinit
    if [[ ! "$reinit" =~ ^[Yy]$ ]]; then
        echo "Keeping existing database."
        SKIP_INIT=true
    fi
fi

if [ "${SKIP_INIT}" != "true" ]; then
    echo "The USDA FoodData Central database contains nutrition information for"
    echo "thousands of foods. You need to download it once."
    echo ""
    echo "Download from: https://fdc.nal.usda.gov/download-datasets.html"
    echo "Select: 'Full Download of All Data Types' → CSV format"
    echo ""

    read -p "Enter path to extracted USDA CSV directory: " usda_path

    if [ ! -d "$usda_path" ]; then
        print_error "Directory not found: $usda_path"
        exit 1
    fi

    if [ ! -f "$usda_path/food.csv" ]; then
        print_error "food.csv not found in $usda_path"
        print_error "Make sure you extracted the USDA CSV files"
        exit 1
    fi

    print_step "Initializing database (this takes 1-2 minutes)..."
    mealplan init "$usda_path"
    print_success "Database initialized with USDA nutrition data"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Diet Type Selection
# ─────────────────────────────────────────────────────────────────────────────

print_header "Step 3: Your Diet Type"

echo "What type of diet do you follow?"
echo "  1) Omnivore - eat everything"
echo "  2) Pescatarian - fish/seafood + eggs, no meat"
echo "  3) Vegetarian - eggs + dairy, no meat or fish"
echo "  4) Vegan - plant-based only"
echo ""
read -p "Enter choice (1-4) [1]: " diet_type_choice
diet_type_choice=${diet_type_choice:-1}

case $diet_type_choice in
    1)
        DIET_TYPE="omnivore"
        DIET_TYPE_DESC="omnivore"
        ;;
    2)
        DIET_TYPE="pescatarian"
        DIET_TYPE_DESC="pescatarian (fish + eggs, no meat)"
        ;;
    3)
        DIET_TYPE="vegetarian"
        DIET_TYPE_DESC="vegetarian (eggs + dairy, no meat/fish)"
        ;;
    4)
        DIET_TYPE="vegan"
        DIET_TYPE_DESC="vegan (plant-based)"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

print_step "Diet type: $DIET_TYPE_DESC"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Diet Style Selection
# ─────────────────────────────────────────────────────────────────────────────

print_header "Step 4: Your Diet Style"

echo "What eating style do you prefer?"
echo "  1) Standard - balanced macros, all food groups"
echo "  2) Slow-Carb (Tim Ferriss) - legumes as carbs, no white carbs/fruit"
echo "  3) Low-Carb - reduced carbohydrates, higher fat"
echo "  4) High-Protein - focus on protein for muscle building"
echo "  5) Mediterranean - olive oil, fish, vegetables, legumes"
echo ""
read -p "Enter choice (1-5) [1]: " diet_style_choice
diet_style_choice=${diet_style_choice:-1}

case $diet_style_choice in
    1)
        DIET_STYLE="standard"
        DIET_STYLE_DESC="standard balanced"
        EXCLUDE_CARBS=false
        LEGUME_FOCUS=false
        ;;
    2)
        DIET_STYLE="slowcarb"
        DIET_STYLE_DESC="slow-carb (Tim Ferriss style)"
        EXCLUDE_CARBS=true
        LEGUME_FOCUS=true
        ;;
    3)
        DIET_STYLE="lowcarb"
        DIET_STYLE_DESC="low-carb"
        EXCLUDE_CARBS=true
        LEGUME_FOCUS=false
        ;;
    4)
        DIET_STYLE="highprotein"
        DIET_STYLE_DESC="high-protein"
        EXCLUDE_CARBS=false
        LEGUME_FOCUS=false
        ;;
    5)
        DIET_STYLE="mediterranean"
        DIET_STYLE_DESC="Mediterranean"
        EXCLUDE_CARBS=false
        LEGUME_FOCUS=true
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

print_step "Diet style: $DIET_STYLE_DESC"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Health Goals
# ─────────────────────────────────────────────────────────────────────────────

print_header "Step 5: Your Health Goals"

echo "What is your primary goal?"
echo "  1) Weight Loss (cutting) - calorie deficit, high protein"
echo "  2) Muscle Gain (bulking) - calorie surplus, high protein & carbs"
echo "  3) Maintenance - balanced nutrition, maintain weight"
echo "  4) Custom - I'll specify my own constraints"
echo ""
read -p "Enter choice (1-4): " goal_choice

case $goal_choice in
    1)
        GOAL="cutting"
        DEFAULT_CAL_MIN=1600
        DEFAULT_CAL_MAX=1800
        DEFAULT_PROTEIN=150
        ;;
    2)
        GOAL="bulking"
        DEFAULT_CAL_MIN=2800
        DEFAULT_CAL_MAX=3200
        DEFAULT_PROTEIN=180
        ;;
    3)
        GOAL="maintenance"
        DEFAULT_CAL_MIN=2200
        DEFAULT_CAL_MAX=2400
        DEFAULT_PROTEIN=120
        ;;
    4)
        GOAL="custom"
        DEFAULT_CAL_MIN=2000
        DEFAULT_CAL_MAX=2500
        DEFAULT_PROTEIN=100
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# Adjust protein for high-protein style
if [ "$DIET_STYLE" = "highprotein" ]; then
    DEFAULT_PROTEIN=$((DEFAULT_PROTEIN + 30))
fi

echo ""
print_step "Selected goal: $GOAL"
echo ""

# Get calorie range
echo "Now let's set your daily calorie target."
echo "(Hint: Use a TDEE calculator to estimate your needs)"
echo ""
read -p "Minimum daily calories [$DEFAULT_CAL_MIN]: " cal_min
cal_min=${cal_min:-$DEFAULT_CAL_MIN}

read -p "Maximum daily calories [$DEFAULT_CAL_MAX]: " cal_max
cal_max=${cal_max:-$DEFAULT_CAL_MAX}

# Get protein target
echo ""
echo "Protein is crucial for muscle maintenance and satiety."
echo "(Recommended: 0.7-1g per pound of body weight)"
echo ""
read -p "Minimum daily protein in grams [$DEFAULT_PROTEIN]: " protein_min
protein_min=${protein_min:-$DEFAULT_PROTEIN}

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Tag Staple Foods
# ─────────────────────────────────────────────────────────────────────────────

print_header "Step 6: Setting Up Your Staple Foods"

echo "For best results, we'll tag foods you actually buy and eat."
echo "This prevents the optimizer from suggesting baby food or exotic meats!"
echo ""
echo "Based on your diet ($DIET_TYPE_DESC, $DIET_STYLE_DESC), we'll suggest"
echo "common staple foods to get you started."
echo ""
read -p "Would you like to auto-tag recommended staples? (Y/n): " auto_tag
auto_tag=${auto_tag:-Y}

if [[ "$auto_tag" =~ ^[Yy]$ ]]; then
    print_step "Tagging recommended staple foods..."
    echo ""

    # ─── PROTEINS ───
    # Fish/Seafood (for omnivore, pescatarian, mediterranean)
    if [ "$DIET_TYPE" != "vegetarian" ] && [ "$DIET_TYPE" != "vegan" ]; then
        mealplan tags add 175167 staple 2>/dev/null && echo "  + Salmon" || true
        mealplan tags add 175159 staple 2>/dev/null && echo "  + Tuna" || true
        mealplan tags add 171955 staple 2>/dev/null && echo "  + Cod" || true
        mealplan tags add 2684443 staple 2>/dev/null && echo "  + Shrimp" || true
        mealplan tags add 175139 staple 2>/dev/null && echo "  + Sardines" || true
    fi

    # Meat (omnivore only)
    if [ "$DIET_TYPE" = "omnivore" ]; then
        mealplan tags add 171077 staple 2>/dev/null && echo "  + Chicken breast" || true
        mealplan tags add 173110 staple 2>/dev/null && echo "  + Lean ground beef" || true
    fi

    # Eggs (not vegan)
    if [ "$DIET_TYPE" != "vegan" ]; then
        mealplan tags add 171287 staple 2>/dev/null && echo "  + Eggs (raw)" || true
        mealplan tags add 173424 staple 2>/dev/null && echo "  + Eggs (hard-boiled)" || true
    fi

    # ─── LEGUMES ───
    # Always add for slow-carb, mediterranean, or vegan/vegetarian
    if [ "$LEGUME_FOCUS" = true ] || [ "$DIET_TYPE" = "vegan" ] || [ "$DIET_TYPE" = "vegetarian" ]; then
        mealplan tags add 172421 staple 2>/dev/null && echo "  + Lentils" || true
        mealplan tags add 175187 staple 2>/dev/null && echo "  + Black beans" || true
        mealplan tags add 173757 staple 2>/dev/null && echo "  + Chickpeas" || true
        mealplan tags add 174289 staple 2>/dev/null && echo "  + Hummus" || true
        mealplan tags add 174286 staple 2>/dev/null && echo "  + Pinto beans" || true
    fi

    # ─── VEGETABLES ───
    mealplan tags add 168462 staple 2>/dev/null && echo "  + Spinach" || true
    mealplan tags add 169967 staple 2>/dev/null && echo "  + Broccoli" || true
    mealplan tags add 168421 staple 2>/dev/null && echo "  + Kale" || true
    mealplan tags add 169986 staple 2>/dev/null && echo "  + Cauliflower" || true
    mealplan tags add 168389 staple 2>/dev/null && echo "  + Asparagus" || true
    mealplan tags add 170108 staple 2>/dev/null && echo "  + Bell peppers" || true
    mealplan tags add 170457 staple 2>/dev/null && echo "  + Tomatoes" || true
    mealplan tags add 170000 staple 2>/dev/null && echo "  + Onions" || true
    mealplan tags add 169230 staple 2>/dev/null && echo "  + Garlic" || true
    mealplan tags add 169975 staple 2>/dev/null && echo "  + Cabbage" || true

    # ─── CARBS (if not slow-carb/low-carb) ───
    if [ "$EXCLUDE_CARBS" = false ]; then
        mealplan tags add 169704 staple 2>/dev/null && echo "  + Brown rice" || true
        mealplan tags add 168482 staple 2>/dev/null && echo "  + Sweet potato" || true
        mealplan tags add 171661 staple 2>/dev/null && echo "  + Oats" || true
        mealplan tags add 173944 staple 2>/dev/null && echo "  + Bananas" || true
    fi

    # ─── FATS ───
    mealplan tags add 171705 staple 2>/dev/null && echo "  + Avocado" || true
    mealplan tags add 748608 staple 2>/dev/null && echo "  + Olive oil" || true
    mealplan tags add 170567 staple 2>/dev/null && echo "  + Almonds" || true

    # ─── DAIRY (if not vegan) ───
    if [ "$DIET_TYPE" != "vegan" ]; then
        if [ "$DIET_STYLE" = "slowcarb" ]; then
            # Slow-carb allows cottage cheese only
            mealplan tags add 2346384 staple 2>/dev/null && echo "  + Cottage cheese" || true
        else
            mealplan tags add 170886 staple 2>/dev/null && echo "  + Yogurt" || true
        fi
    fi

    # ─── FERMENTED (slow-carb bonus) ───
    if [ "$DIET_STYLE" = "slowcarb" ]; then
        mealplan tags add 169279 staple 2>/dev/null && echo "  + Sauerkraut" || true
    fi

    echo ""
    print_success "Tagged recommended staples for $DIET_TYPE $DIET_STYLE diet"
fi

echo ""
echo "You can add more staples anytime with:"
echo "  ${BOLD}./scripts/tag-staple-foods.sh${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Build Profile
# ─────────────────────────────────────────────────────────────────────────────

print_header "Step 7: Creating Your Profile"

# Build the profile YAML
mkdir -p "$PROFILES_DIR"
PROFILE_NAME="${DIET_TYPE}_${DIET_STYLE}_${GOAL}"
PROFILE_FILE="${PROFILES_DIR}/${PROFILE_NAME}.yaml"

cat > "$PROFILE_FILE" << EOF
# Personal Meal Plan Profile
# Generated by setup-meal-plan.sh
# Diet: ${DIET_TYPE_DESC}, Style: ${DIET_STYLE_DESC}, Goal: ${GOAL}

name: ${PROFILE_NAME}
description: "${DIET_TYPE_DESC} ${DIET_STYLE_DESC} ${GOAL} - ${cal_min}-${cal_max} cal, ${protein_min}g protein"

calories:
  min: ${cal_min}
  max: ${cal_max}

nutrients:
  protein:
    min: ${protein_min}
  fiber:
    min: 30
  sodium:
    max: 2300
EOF

# Add carb limits for low-carb/slow-carb
if [ "$DIET_STYLE" = "lowcarb" ]; then
    cat >> "$PROFILE_FILE" << EOF
  carbohydrate:
    max: 100
EOF
fi

cat >> "$PROFILE_FILE" << EOF

# Use only foods you've tagged as staples
include_tags:
  - staple

exclude_tags:
  - exclude
  - junk_food

options:
  max_grams_per_food: 500
  use_quadratic_penalty: true
  lambda_deviation: 0.001
EOF

print_success "Created profile at: $PROFILE_FILE"
echo ""
echo "Your profile:"
echo "─────────────────────────────────────"
cat "$PROFILE_FILE"
echo "─────────────────────────────────────"

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Run Optimization
# ─────────────────────────────────────────────────────────────────────────────

print_header "Step 8: Finding Optimal Foods"

echo "Now we'll run the optimizer to find a diverse set of foods"
echo "that meets all your nutritional requirements."
echo ""

read -p "Press Enter to run optimization..."

print_step "Running optimization..."
echo ""

mealplan optimize --file "$PROFILE_FILE" --output table

print_success "Optimization complete!"

# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Export for Claude
# ─────────────────────────────────────────────────────────────────────────────

print_header "Step 9: Generate Recipe Prompt"

echo "Now let's export your food list so Claude can create recipes."
echo ""
read -p "How many days should the meal plan cover? [7]: " num_days
num_days=${num_days:-7}

OUTPUT_FILE="${DATA_DIR}/meal_plan_request.md"
mealplan export-for-llm latest --days "$num_days" --output "$OUTPUT_FILE"

print_success "Exported to: $OUTPUT_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# Done!
# ─────────────────────────────────────────────────────────────────────────────

print_header "Setup Complete!"

echo "Your personalized ${DIET_TYPE} ${DIET_STYLE} meal plan is ready!"
echo ""
echo "What to do next:"
echo ""
echo "  1. Open the generated prompt:"
echo "     ${BOLD}cat $OUTPUT_FILE${NC}"
echo ""
echo "  2. Copy it to Claude (claude.ai) and ask:"
echo "     ${BOLD}\"Create a ${num_days}-day meal plan with recipes using these foods.\"${NC}"
echo ""
echo "  3. Re-run optimization anytime:"
echo "     ${BOLD}mealplan optimize --file $PROFILE_FILE${NC}"
echo ""
echo "  4. Add more staple foods:"
echo "     ${BOLD}./scripts/tag-staple-foods.sh${NC}"
echo ""
echo "  5. Exclude foods you don't like:"
echo "     ${BOLD}mealplan search \"food name\"${NC}"
echo "     ${BOLD}mealplan tags add <fdc_id> exclude${NC}"
echo "     Then re-run optimization."
echo ""
echo "  6. Edit your profile:"
echo "     ${BOLD}$PROFILE_FILE${NC}"
echo ""
print_success "Enjoy your optimized meals!"
