"""Tests for CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from llmn.cli import app

runner = CliRunner()


class TestMainCommands:
    """Tests for main CLI commands."""

    def test_help(self):
        """Test that --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "llmn" in result.output.lower() or "nutrition" in result.output.lower()

    def test_init_requires_path(self):
        """Test that init command requires USDA path."""
        result = runner.invoke(app, ["init"])
        assert result.exit_code != 0

    def test_search_requires_query(self):
        """Test that search command requires query."""
        result = runner.invoke(app, ["search"])
        assert result.exit_code != 0


class TestPricesCommands:
    """Tests for prices subcommands."""

    def test_prices_help(self):
        """Test that prices --help works."""
        result = runner.invoke(app, ["prices", "--help"])
        assert result.exit_code == 0
        assert "prices" in result.output.lower()

    def test_prices_add_requires_args(self):
        """Test that prices add requires arguments."""
        result = runner.invoke(app, ["prices", "add"])
        assert result.exit_code != 0


class TestTagsCommands:
    """Tests for tags subcommands."""

    def test_tags_help(self):
        """Test that tags --help works."""
        result = runner.invoke(app, ["tags", "--help"])
        assert result.exit_code == 0

    def test_tags_add_requires_args(self):
        """Test that tags add requires arguments."""
        result = runner.invoke(app, ["tags", "add"])
        assert result.exit_code != 0


class TestProfileCommands:
    """Tests for profile subcommands."""

    def test_profile_help(self):
        """Test that profile --help works."""
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0

    def test_profile_create_requires_file(self):
        """Test that profile create requires --from-file."""
        result = runner.invoke(app, ["profile", "create", "test"])
        assert result.exit_code != 0
