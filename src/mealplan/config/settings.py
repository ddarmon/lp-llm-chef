"""Application settings and configuration management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


def _default_config_dir() -> Path:
    """Return the default configuration directory."""
    return Path.home() / ".mealplan"


def _default_db_path() -> Path:
    """Return the default database path."""
    return _default_config_dir() / "mealplan.db"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: Path = field(default_factory=_default_db_path)


@dataclass
class USDAConfig:
    """USDA data configuration."""

    data_path: Optional[Path] = None
    data_types: list[str] = field(
        default_factory=lambda: ["foundation_food", "sr_legacy_food"]
    )


@dataclass
class OptimizationConfig:
    """Optimization solver configuration."""

    solver: str = "slsqp"  # "linprog" or "slsqp"
    use_quadratic_penalty: bool = True
    lambda_cost: float = 1.0
    lambda_deviation: float = 0.001
    max_grams_per_food: float = 500.0


@dataclass
class DefaultsConfig:
    """Default values for various operations."""

    planning_days: int = 1
    output_format: str = "table"  # "table", "json", "markdown"


@dataclass
class Settings:
    """Main application settings."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    usda: USDAConfig = field(default_factory=USDAConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from YAML file or return defaults.

        Args:
            config_path: Path to config.yaml. If None, uses ~/.mealplan/config.yaml

        Returns:
            Settings instance
        """
        if config_path is None:
            config_path = _default_config_dir() / "config.yaml"

        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        settings = cls()

        # Parse database config
        if "database" in data:
            db_data = data["database"]
            if "path" in db_data:
                settings.database.path = Path(db_data["path"]).expanduser()

        # Parse USDA config
        if "usda" in data:
            usda_data = data["usda"]
            if "data_path" in usda_data:
                settings.usda.data_path = Path(usda_data["data_path"]).expanduser()
            if "data_types" in usda_data:
                settings.usda.data_types = usda_data["data_types"]

        # Parse optimization config
        if "optimization" in data:
            opt_data = data["optimization"]
            if "solver" in opt_data:
                settings.optimization.solver = opt_data["solver"]
            if "use_quadratic_penalty" in opt_data:
                settings.optimization.use_quadratic_penalty = opt_data[
                    "use_quadratic_penalty"
                ]
            if "lambda_cost" in opt_data:
                settings.optimization.lambda_cost = float(opt_data["lambda_cost"])
            if "lambda_deviation" in opt_data:
                settings.optimization.lambda_deviation = float(
                    opt_data["lambda_deviation"]
                )
            if "max_grams_per_food" in opt_data:
                settings.optimization.max_grams_per_food = float(
                    opt_data["max_grams_per_food"]
                )

        # Parse defaults
        if "defaults" in data:
            def_data = data["defaults"]
            if "planning_days" in def_data:
                settings.defaults.planning_days = int(def_data["planning_days"])
            if "output_format" in def_data:
                settings.defaults.output_format = def_data["output_format"]

        return settings

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save current settings to YAML file.

        Args:
            config_path: Path to save config.yaml. If None, uses ~/.mealplan/config.yaml
        """
        if config_path is None:
            config_path = _default_config_dir() / "config.yaml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "database": {
                "path": str(self.database.path),
            },
            "usda": {
                "data_path": str(self.usda.data_path) if self.usda.data_path else None,
                "data_types": self.usda.data_types,
            },
            "optimization": {
                "solver": self.optimization.solver,
                "use_quadratic_penalty": self.optimization.use_quadratic_penalty,
                "lambda_cost": self.optimization.lambda_cost,
                "lambda_deviation": self.optimization.lambda_deviation,
                "max_grams_per_food": self.optimization.max_grams_per_food,
            },
            "defaults": {
                "planning_days": self.defaults.planning_days,
                "output_format": self.defaults.output_format,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance, loading from disk if needed."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from disk."""
    global _settings
    _settings = Settings.load()
    return _settings
