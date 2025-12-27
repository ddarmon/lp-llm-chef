"""SQLite database schema definitions."""

SCHEMA_SQL = """
-- Core food table (derived from USDA)
CREATE TABLE IF NOT EXISTS foods (
    fdc_id INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    data_type TEXT,
    food_category TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_foods_description ON foods(description);
CREATE INDEX IF NOT EXISTS idx_foods_data_type ON foods(data_type);
CREATE INDEX IF NOT EXISTS idx_foods_is_active ON foods(is_active);

-- Nutrient definitions
CREATE TABLE IF NOT EXISTS nutrients (
    nutrient_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    unit TEXT NOT NULL,
    display_name TEXT,
    is_macro BOOLEAN DEFAULT FALSE
);

-- Nutrient values per food (per 100g)
CREATE TABLE IF NOT EXISTS food_nutrients (
    fdc_id INTEGER,
    nutrient_id INTEGER,
    amount REAL NOT NULL,
    PRIMARY KEY (fdc_id, nutrient_id),
    FOREIGN KEY (fdc_id) REFERENCES foods(fdc_id),
    FOREIGN KEY (nutrient_id) REFERENCES nutrients(nutrient_id)
);

CREATE INDEX IF NOT EXISTS idx_food_nutrients_fdc ON food_nutrients(fdc_id);
CREATE INDEX IF NOT EXISTS idx_food_nutrients_nutrient ON food_nutrients(nutrient_id);

-- User-maintained price data
CREATE TABLE IF NOT EXISTS prices (
    fdc_id INTEGER PRIMARY KEY,
    price_per_100g REAL NOT NULL,
    price_source TEXT,
    price_date DATE,
    notes TEXT,
    FOREIGN KEY (fdc_id) REFERENCES foods(fdc_id)
);

-- Serving size information
CREATE TABLE IF NOT EXISTS servings (
    fdc_id INTEGER,
    portion_description TEXT,
    gram_weight REAL NOT NULL,
    PRIMARY KEY (fdc_id, portion_description),
    FOREIGN KEY (fdc_id) REFERENCES foods(fdc_id)
);

-- User food tags for filtering
CREATE TABLE IF NOT EXISTS food_tags (
    fdc_id INTEGER,
    tag TEXT,
    PRIMARY KEY (fdc_id, tag),
    FOREIGN KEY (fdc_id) REFERENCES foods(fdc_id)
);

CREATE INDEX IF NOT EXISTS idx_food_tags_tag ON food_tags(tag);

-- Saved constraint profiles
CREATE TABLE IF NOT EXISTS constraint_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    constraints_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization run history
CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT,
    total_cost REAL,
    result_json TEXT,
    FOREIGN KEY (profile_id) REFERENCES constraint_profiles(profile_id)
);

CREATE INDEX IF NOT EXISTS idx_optimization_runs_date ON optimization_runs(created_at);
"""


def get_schema_sql() -> str:
    """Return the complete schema SQL."""
    return SCHEMA_SQL
