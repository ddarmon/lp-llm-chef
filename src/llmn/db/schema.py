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

-- User profiles for weight tracking and TDEE learning
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    sex TEXT CHECK(sex IN ('male', 'female')),
    height_inches REAL,
    weight_lbs REAL,
    activity_level TEXT CHECK(activity_level IN ('sedentary', 'lightly_active', 'moderate', 'active', 'very_active')),
    goal TEXT,
    target_weight_lbs REAL,
    diet_type TEXT CHECK(diet_type IN ('omnivore', 'pescatarian', 'vegetarian', 'vegan') OR diet_type IS NULL),
    diet_style TEXT CHECK(diet_style IN ('standard', 'slow_carb', 'low_carb', 'keto', 'mediterranean', 'paleo') OR diet_style IS NULL),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Weight log with EMA trend (Hacker's Diet style)
CREATE TABLE IF NOT EXISTS weight_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    weight_lbs REAL NOT NULL,
    trend_lbs REAL NOT NULL,
    measured_at DATE NOT NULL,
    notes TEXT,
    UNIQUE(user_id, measured_at),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

CREATE INDEX IF NOT EXISTS idx_weight_log_user_date ON weight_log(user_id, measured_at);

-- Calorie log for TDEE learning
CREATE TABLE IF NOT EXISTS calorie_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date DATE NOT NULL,
    planned_calories REAL NOT NULL,
    notes TEXT,
    UNIQUE(user_id, date),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

CREATE INDEX IF NOT EXISTS idx_calorie_log_user_date ON calorie_log(user_id, date);

-- TDEE estimates from Kalman filter
CREATE TABLE IF NOT EXISTS tdee_estimates (
    estimate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    estimated_at DATE NOT NULL,
    mifflin_tdee REAL NOT NULL,
    tdee_bias REAL NOT NULL DEFAULT 0.0,
    variance REAL NOT NULL DEFAULT 10000.0,
    adjusted_tdee REAL NOT NULL,
    UNIQUE(user_id, estimated_at),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

CREATE INDEX IF NOT EXISTS idx_tdee_estimates_user_date ON tdee_estimates(user_id, estimated_at);
"""


def get_schema_sql() -> str:
    """Return the complete schema SQL."""
    return SCHEMA_SQL


def migrate_user_profiles_add_diet_columns(conn) -> bool:
    """
    Add diet_type and diet_style columns to user_profiles if they don't exist.

    This is a migration for existing databases that were created before
    these columns were added to the schema.

    Returns True if migration was performed, False if columns already existed.
    """
    import sqlite3

    cursor = conn.execute("PRAGMA table_info(user_profiles)")
    columns = {row[1] for row in cursor.fetchall()}

    migrated = False

    if "diet_type" not in columns:
        conn.execute(
            """
            ALTER TABLE user_profiles
            ADD COLUMN diet_type TEXT CHECK(diet_type IN ('omnivore', 'pescatarian', 'vegetarian', 'vegan') OR diet_type IS NULL)
            """
        )
        migrated = True

    if "diet_style" not in columns:
        conn.execute(
            """
            ALTER TABLE user_profiles
            ADD COLUMN diet_style TEXT CHECK(diet_style IN ('standard', 'slow_carb', 'low_carb', 'keto', 'mediterranean', 'paleo') OR diet_style IS NULL)
            """
        )
        migrated = True

    if migrated:
        conn.commit()

    return migrated
