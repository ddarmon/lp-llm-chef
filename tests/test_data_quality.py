"""Tests for data quality detection."""

from __future__ import annotations

from llmn.data.quality import calculate_fallback_energy, detect_incomplete_foods


class TestCalculateFallbackEnergy:
    """Tests for Atwater factor energy calculation."""

    def test_protein_only(self):
        """Test energy calculation with only protein."""
        energy = calculate_fallback_energy(protein_g=25, carbs_g=0, fat_g=0)
        assert energy == 100  # 25 * 4 = 100 kcal

    def test_carbs_only(self):
        """Test energy calculation with only carbs."""
        energy = calculate_fallback_energy(protein_g=0, carbs_g=50, fat_g=0)
        assert energy == 200  # 50 * 4 = 200 kcal

    def test_fat_only(self):
        """Test energy calculation with only fat."""
        energy = calculate_fallback_energy(protein_g=0, carbs_g=0, fat_g=10)
        assert energy == 90  # 10 * 9 = 90 kcal

    def test_mixed_macros(self):
        """Test energy calculation with all macros."""
        # Chicken breast: 31g protein, 0g carbs, 3.6g fat
        # Expected: 31*4 + 0*4 + 3.6*9 = 124 + 0 + 32.4 = 156.4 kcal
        energy = calculate_fallback_energy(protein_g=31, carbs_g=0, fat_g=3.6)
        assert abs(energy - 156.4) < 0.1


class TestDetectIncompleteFoods:
    """Tests for detect_incomplete_foods function."""

    def test_complete_foods_may_have_minor_issues(self, sample_foods):
        """Test detection on complete foods (may have minor inconsistencies)."""
        with sample_foods.get_connection() as conn:
            issues = detect_incomplete_foods(conn, [1, 2, 3])

        # Sample foods are complete but may have minor energy inconsistencies
        # (stored energy vs calculated from Atwater factors)
        # Any issues should only be warnings, not errors
        for issue in issues:
            assert issue.severity == "warning"

    def test_detect_zero_energy_with_macros(self, sample_foods):
        """Test detection of foods with zero energy but non-zero macros."""
        with sample_foods.get_connection() as conn:
            # Insert a food with zero energy but has macros
            conn.execute(
                "INSERT INTO foods (fdc_id, description, data_type, is_active) VALUES (?, ?, ?, ?)",
                (100, "Broken food entry", "sr_legacy_food", True),
            )
            # Add nutrients with zero energy but non-zero protein
            conn.execute("INSERT INTO food_nutrients VALUES (?, ?, ?)", (100, 1008, 0))  # Energy = 0
            conn.execute("INSERT INTO food_nutrients VALUES (?, ?, ?)", (100, 1003, 25))  # Protein = 25g
            conn.execute("INSERT INTO food_nutrients VALUES (?, ?, ?)", (100, 1004, 5))  # Fat = 5g
            conn.execute("INSERT INTO food_nutrients VALUES (?, ?, ?)", (100, 1005, 10))  # Carbs = 10g

            issues = detect_incomplete_foods(conn, [1, 2, 100])

        # Should detect the broken food
        assert len(issues) >= 1

        # Find the issue for food 100
        food_100_issues = [i for i in issues if i.fdc_id == 100]
        assert len(food_100_issues) >= 1
        assert food_100_issues[0].issue_type in ("missing_energy", "calculated_energy")

    def test_empty_fdc_list(self, sample_foods):
        """Test with empty list of food IDs."""
        with sample_foods.get_connection() as conn:
            issues = detect_incomplete_foods(conn, [])

        assert len(issues) == 0
