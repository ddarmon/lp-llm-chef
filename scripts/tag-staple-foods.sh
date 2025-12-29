#!/bin/bash
# tag-staple-foods.sh - Quickly tag your grocery staples
#
# This script helps you build a curated list of foods you actually buy.
# Once tagged, you can use include_tags: [staple] in your profile to
# only optimize across these foods.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─────────────────────────────────────────────────────────────────────────────
# Food category definitions (FDC IDs for common foods)
# ─────────────────────────────────────────────────────────────────────────────

declare -A FISH_FOODS=(
    [175167]="Salmon, Atlantic, farmed, raw"
    [175159]="Tuna, yellowfin, raw"
    [171955]="Cod, Atlantic, raw"
    [175176]="Tilapia, raw"
    [175119]="Mackerel, Atlantic, raw"
    [175139]="Sardines, canned in oil"
    [2747652]="Anchovies, canned in olive oil"
    [173672]="Mackerel, Pacific, raw"
)

declare -A SEAFOOD_FOODS=(
    [2684443]="Shrimp, farm raised, raw"
    [171970]="Shrimp, cooked"
    [171972]="Shrimp, canned"
    [175180]="Shrimp, cooked"
)

declare -A MEAT_FOODS=(
    [171077]="Chicken breast, skinless, raw"
    [173110]="Ground beef, 93% lean, raw"
    [167762]="Turkey breast, raw"
    [168702]="Pork tenderloin, raw"
    [174756]="Ground turkey, 93% lean, raw"
)

declare -A LEGUMES_FOODS=(
    [172421]="Lentils, cooked"
    [175187]="Black beans, cooked"
    [173757]="Chickpeas, cooked"
    [174289]="Hummus, commercial"
    [174286]="Pinto beans, canned"
    [175201]="Kidney beans, cooked"
    [175196]="White beans, cooked"
)

declare -A VEGGIES_FOODS=(
    [168462]="Spinach, raw"
    [169967]="Broccoli, cooked"
    [168421]="Kale, raw"
    [169986]="Cauliflower, raw"
    [168389]="Asparagus, raw"
    [170108]="Bell peppers, red, raw"
    [170457]="Tomatoes, raw"
    [170000]="Onions, raw"
    [169230]="Garlic, raw"
    [169975]="Cabbage, raw"
    [170393]="Zucchini, raw"
    [169145]="Carrots, raw"
    [168483]="Celery, raw"
    [170407]="Cucumber, raw"
    [169248]="Mushrooms, white, raw"
    [169228]="Green beans, raw"
)

declare -A GREENS_FOODS=(
    [168462]="Spinach, raw"
    [168421]="Kale, raw"
    [168409]="Arugula, raw"
    [169247]="Lettuce, romaine, raw"
    [168417]="Chard, Swiss, raw"
    [168406]="Collard greens, raw"
)

declare -A CARBS_FOODS=(
    [169704]="Brown rice, long-grain, cooked"
    [168875]="Brown rice, medium-grain, cooked"
    [168482]="Sweet potato, raw"
    [171661]="Oats, instant, dry"
    [169761]="Quinoa, cooked"
    [170285]="Pasta, whole wheat, cooked"
    [168880]="Potatoes, russet, raw"
)

declare -A FRUITS_FOODS=(
    [173944]="Bananas, raw"
    [171688]="Apples, raw"
    [169097]="Blueberries, raw"
    [167762]="Strawberries, raw"
    [169910]="Oranges, raw"
    [169926]="Grapes, red, raw"
    [169124]="Mango, raw"
)

declare -A FATS_FOODS=(
    [171705]="Avocado, raw"
    [748608]="Olive oil, extra virgin"
    [170567]="Almonds, raw"
    [170158]="Almonds, dry roasted"
    [170187]="Walnuts, raw"
    [170178]="Cashews, raw"
    [170563]="Peanut butter"
    [171411]="Coconut oil"
)

declare -A DAIRY_FOODS=(
    [170886]="Yogurt, plain, low fat"
    [171284]="Yogurt, plain, whole milk"
    [2346384]="Cottage cheese, full fat"
    [173414]="Milk, whole"
    [170857]="Cheese, cheddar"
    [170904]="Cheese, mozzarella"
    [173410]="Greek yogurt, plain, nonfat"
)

declare -A EGGS_FOODS=(
    [171287]="Egg, whole, raw"
    [173424]="Egg, whole, hard-boiled"
    [173423]="Egg, whole, fried"
    [172187]="Egg, whole, scrambled"
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

show_category() {
    local -n foods=$1
    local category_name=$2

    echo ""
    echo -e "${CYAN}${BOLD}── $category_name ──${NC}"
    echo ""

    for fdc_id in "${!foods[@]}"; do
        echo "  [$fdc_id] ${foods[$fdc_id]}"
    done
    echo ""
}

tag_category() {
    local -n foods=$1
    local category_name=$2
    local count=0

    echo ""
    echo -e "${CYAN}Tagging $category_name...${NC}"

    for fdc_id in "${!foods[@]}"; do
        if llmn tags add "$fdc_id" staple 2>/dev/null; then
            echo -e "  ${GREEN}+${NC} ${foods[$fdc_id]}"
            ((count++))
        fi
    done

    echo ""
    echo -e "${GREEN}Tagged $count foods as staple${NC}"
}

show_help() {
    echo -e "${BLUE}${BOLD}Staple Foods Tagger${NC}"
    echo "Tag foods you regularly buy so optimization uses realistic ingredients."
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo "  s <query>     - Search for a food"
    echo "  t <fdc_id>    - Tag food as 'staple'"
    echo "  x <fdc_id>    - Tag food as 'exclude' (never use)"
    echo "  r <fdc_id>    - Remove 'staple' tag from food"
    echo "  l             - List current staples"
    echo "  q             - Quit"
    echo ""
    echo -e "${BOLD}Bulk Category Commands:${NC}"
    echo "  c fish        - Show/tag common fish"
    echo "  c seafood     - Show/tag shrimp, shellfish"
    echo "  c meat        - Show/tag chicken, beef, pork, turkey"
    echo "  c legumes     - Show/tag beans, lentils, chickpeas"
    echo "  c veggies     - Show/tag common vegetables"
    echo "  c greens      - Show/tag leafy greens"
    echo "  c carbs       - Show/tag rice, potatoes, oats"
    echo "  c fruits      - Show/tag common fruits"
    echo "  c fats        - Show/tag oils, nuts, avocado"
    echo "  c dairy       - Show/tag yogurt, cheese, milk"
    echo "  c eggs        - Show/tag egg preparations"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  > c fish           # Show fish options"
    echo "  > c fish tag       # Tag ALL fish as staples"
    echo "  > s salmon         # Search for salmon"
    echo "  > t 175167         # Tag FDC ID 175167 as staple"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

show_help

while true; do
    read -p "> " cmd args

    case $cmd in
        s|search)
            if [ -z "$args" ]; then
                echo "Usage: s <search query>"
                continue
            fi
            llmn search "$args"
            ;;

        t|tag)
            if [ -z "$args" ]; then
                echo "Usage: t <fdc_id>"
                continue
            fi
            llmn tags add "$args" staple
            echo -e "${GREEN}Tagged $args as staple${NC}"
            ;;

        x|exclude)
            if [ -z "$args" ]; then
                echo "Usage: x <fdc_id>"
                continue
            fi
            llmn tags add "$args" exclude
            echo -e "${GREEN}Tagged $args as exclude${NC}"
            ;;

        r|remove)
            if [ -z "$args" ]; then
                echo "Usage: r <fdc_id>"
                continue
            fi
            llmn tags remove "$args" staple
            echo -e "${YELLOW}Removed staple tag from $args${NC}"
            ;;

        l|list)
            echo ""
            echo "Current staple foods:"
            llmn tags list --tag staple 2>/dev/null || echo "(none yet)"
            echo ""
            ;;

        c|category)
            # Parse category name and optional "tag" action
            category=$(echo "$args" | awk '{print $1}')
            action=$(echo "$args" | awk '{print $2}')

            case $category in
                fish)
                    if [ "$action" = "tag" ]; then
                        tag_category FISH_FOODS "Fish"
                    else
                        show_category FISH_FOODS "Fish"
                        echo -e "${YELLOW}Use 'c fish tag' to tag all these as staples${NC}"
                    fi
                    ;;
                seafood)
                    if [ "$action" = "tag" ]; then
                        tag_category SEAFOOD_FOODS "Seafood"
                    else
                        show_category SEAFOOD_FOODS "Seafood"
                        echo -e "${YELLOW}Use 'c seafood tag' to tag all these as staples${NC}"
                    fi
                    ;;
                meat)
                    if [ "$action" = "tag" ]; then
                        tag_category MEAT_FOODS "Meat"
                    else
                        show_category MEAT_FOODS "Meat"
                        echo -e "${YELLOW}Use 'c meat tag' to tag all these as staples${NC}"
                    fi
                    ;;
                legumes|beans)
                    if [ "$action" = "tag" ]; then
                        tag_category LEGUMES_FOODS "Legumes"
                    else
                        show_category LEGUMES_FOODS "Legumes"
                        echo -e "${YELLOW}Use 'c legumes tag' to tag all these as staples${NC}"
                    fi
                    ;;
                veggies|vegetables)
                    if [ "$action" = "tag" ]; then
                        tag_category VEGGIES_FOODS "Vegetables"
                    else
                        show_category VEGGIES_FOODS "Vegetables"
                        echo -e "${YELLOW}Use 'c veggies tag' to tag all these as staples${NC}"
                    fi
                    ;;
                greens)
                    if [ "$action" = "tag" ]; then
                        tag_category GREENS_FOODS "Leafy Greens"
                    else
                        show_category GREENS_FOODS "Leafy Greens"
                        echo -e "${YELLOW}Use 'c greens tag' to tag all these as staples${NC}"
                    fi
                    ;;
                carbs)
                    if [ "$action" = "tag" ]; then
                        tag_category CARBS_FOODS "Carbs"
                    else
                        show_category CARBS_FOODS "Carbs"
                        echo -e "${YELLOW}Use 'c carbs tag' to tag all these as staples${NC}"
                    fi
                    ;;
                fruits|fruit)
                    if [ "$action" = "tag" ]; then
                        tag_category FRUITS_FOODS "Fruits"
                    else
                        show_category FRUITS_FOODS "Fruits"
                        echo -e "${YELLOW}Use 'c fruits tag' to tag all these as staples${NC}"
                    fi
                    ;;
                fats|oils|nuts)
                    if [ "$action" = "tag" ]; then
                        tag_category FATS_FOODS "Fats & Oils"
                    else
                        show_category FATS_FOODS "Fats & Oils"
                        echo -e "${YELLOW}Use 'c fats tag' to tag all these as staples${NC}"
                    fi
                    ;;
                dairy)
                    if [ "$action" = "tag" ]; then
                        tag_category DAIRY_FOODS "Dairy"
                    else
                        show_category DAIRY_FOODS "Dairy"
                        echo -e "${YELLOW}Use 'c dairy tag' to tag all these as staples${NC}"
                    fi
                    ;;
                eggs)
                    if [ "$action" = "tag" ]; then
                        tag_category EGGS_FOODS "Eggs"
                    else
                        show_category EGGS_FOODS "Eggs"
                        echo -e "${YELLOW}Use 'c eggs tag' to tag all these as staples${NC}"
                    fi
                    ;;
                *)
                    echo "Unknown category: $category"
                    echo "Available: fish, seafood, meat, legumes, veggies, greens, carbs, fruits, fats, dairy, eggs"
                    ;;
            esac
            ;;

        h|help|\?)
            show_help
            ;;

        q|quit|exit)
            echo "Goodbye!"
            exit 0
            ;;

        "")
            continue
            ;;

        *)
            echo "Unknown command: $cmd"
            echo "Type 'h' for help"
            ;;
    esac
done
