---
name: template
description: Template-based meal planning for realistic meals. Use when user wants practical meals with proper structure (protein + legume + vegetables), different foods at each meal, or human-like meal composition. Triggers on "realistic meals", "different foods each meal", "actual meals", "template", "meal structure", "not the same food".
---

# Template-Based Meal Composition

This skill uses template-based optimization to create realistic meal plans that look like what humans actually eat. Unlike Stigler-style optimization (which spreads across many foods), this approach selects one protein + one legume + vegetables per meal.

## Why Template-Based?

**Problem with Stigler-style optimization**:
```
Breakfast: Lettuce 325g, Onions 100g, Brussels Sprouts 100g, Kale 125g
Lunch:     Beans 200g, Onions 150g, Brussels Sprouts 150g, Lettuce 350g
Dinner:    Beans 200g, Onions 150g, Brussels Sprouts 150g, Lettuce 350g (IDENTICAL!)
```

**Template-based produces**:
```
Breakfast: Eggs 150g, Edamame 100g, Kale 100g
Lunch:     Cod 235g, Kidney beans 216g, Zucchini 161g
Dinner:    Salmon 170g, Lentils 113g, Carrots 149g
Snack:     Peanuts 30g
```

## Quick Start

```bash
# Basic template-based optimization
uv run mealplan optimize --pattern pescatarian --template --json

# With multiple patterns
uv run mealplan optimize --pattern pescatarian --pattern slow_carb --template --json

# With reproducible seed
uv run mealplan optimize --pattern pescatarian --template --seed 42 --json

# With goal-based targets
uv run mealplan optimize --pattern pescatarian --template --goal "fat_loss:185lbs:165lbs" --json
```

## Available Patterns

| Pattern | Protein Sources | Carb Sources | Notes |
|---------|-----------------|--------------|-------|
| `pescatarian` | Fish, eggs | Legumes, grains | No meat |
| `vegetarian` | Eggs, tofu, cheese | Legumes, grains | No meat/fish |
| `vegan` | Tofu, tempeh | Legumes, grains | Plant-based |
| `keto` | Fish, eggs, meat | Fats (not carbs!) | Very low carb |
| `mediterranean` | Fish, poultry | Grains, legumes | + olive oil |
| `paleo` | Meat, fish, eggs | Starchy veg, fruits | No legumes/grains |
| `slow_carb` | Any protein | Legumes only | No white carbs |

**Combine patterns**: `--pattern pescatarian --pattern slow_carb`

## Workflow

### Step 1: Identify User's Diet Type

Map to patterns:
- "I'm pescatarian" → `--pattern pescatarian`
- "I'm vegetarian" → `--pattern vegetarian`
- "I'm vegan" → `--pattern vegan`
- "I eat everything" → `--pattern mediterranean` or `--pattern paleo`

### Step 2: Identify Diet Style

- "Slow carb" → add `--pattern slow_carb`
- "Keto" or "low carb" → use `--pattern keto`
- "Mediterranean" → use `--pattern mediterranean`

### Step 3: Identify Goals

Use the `--goal` flag:
- Weight loss: `--goal "fat_loss:185lbs:165lbs"` (current:target)
- Maintenance: `--goal "maintenance:165lbs"`
- Muscle gain: `--goal "muscle_gain:165lbs:175lbs"`

### Step 4: Run Optimization

```bash
uv run mealplan optimize --pattern pescatarian --pattern slow_carb --template --goal "fat_loss:185lbs:165lbs" --json
```

### Step 5: Present Results

Template output has per-meal structure:

```json
{
  "meals": {
    "breakfast": {
      "foods": [
        {"description": "Egg, whole, cooked, fried", "grams": 150, "slot": "protein"},
        {"description": "Edamame, frozen, prepared", "grams": 100, "slot": "legume"},
        {"description": "Kale, raw", "grams": 100, "slot": "vegetable"}
      ],
      "totals": {"calories": 450, "protein": 35}
    },
    "lunch": {...},
    "dinner": {...},
    "snack": {...}
  },
  "daily_totals": {"calories": 1920, "protein": 175}
}
```

Present each meal with its foods and macros.

## How It Works

**Phase 1 - Selection (Discrete)**:
1. For each meal, select one food per slot (protein, legume, vegetables)
2. Enforce diversity: once a food is selected, exclude it from future meals
3. Selection is random with optional `--seed` for reproducibility

**Phase 2 - Optimization (Continuous)**:
1. Run small QP (~16-20 variables) with only selected foods
2. Use slot-specific target portions (protein=175g, legume=125g, vegetables=100g)
3. Optimize to meet calorie/protein constraints

## Tips

1. **Use `--seed` for reproducibility**: Same seed = same meal plan
2. **Run multiple times**: Different random selections give variety
3. **Combine patterns**: `--pattern pescatarian --pattern slow_carb` is powerful
4. **Check the slots**: Each food shows which slot it fills (protein/legume/vegetable)

## Comparison with Other Modes

| Feature | Template | Multi-period | Single-period |
|---------|----------|--------------|---------------|
| Meal structure | By design | Constrained | None |
| Different foods/meal | By design | Not guaranteed | N/A |
| Exact per-meal calories | Approximate | Exact | N/A |
| Food-meal affinity | Built-in | Manual config | N/A |
| Complexity | Simple | Complex | Simple |

## Example Session

User: "I want realistic meals, not the same vegetables everywhere"

1. Ask: Diet type? (pescatarian/vegetarian/vegan/omnivore)
2. Ask: Any diet style? (slow-carb/keto/mediterranean)
3. Ask: Goal? (weight loss/maintenance/muscle gain)
4. Run:
   ```bash
   uv run mealplan optimize --pattern pescatarian --pattern slow_carb --template --goal "fat_loss:185lbs:165lbs" --json
   ```
5. Show results:
   - Breakfast: Eggs 150g, Edamame 100g, Kale 100g
   - Lunch: Cod 235g, Kidney beans 216g, Zucchini 161g
   - Dinner: Salmon 170g, Lentils 113g, Carrots 149g
   - Snack: Peanuts 30g
6. Offer: Different seed for variety, adjust goal, generate recipes
