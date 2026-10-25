import unittest
from pathlib import Path
import pandas as pd
from analyzer import (
    drop_flagged_outlier_rows,
    iqr_outlier_report,
    missing_value_summary,
    numeric_correlations,
    row_has_iqr_outlier,
)
SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "student_scores.csv"
class AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.read_csv(SAMPLE)
    def test_missing_values_are_counted(self) -> None:
        summary = missing_value_summary(self.df)
        missing = summary.set_index("column")["missing_count"]
        self.assertEqual(int(missing["hours_studied"]), 1)
        self.assertEqual(int(missing["practice_tests"]), 1)
        self.assertEqual(int(missing["survey_rating"]), 1)
        self.assertEqual(int(missing["exam_score"]), 0)
    def test_iqr_flags_extreme_hours_and_reports_thresholds(self) -> None:
        report = iqr_outlier_report(self.df).set_index("column")
        hours = report.loc["hours_studied"]
        self.assertGreaterEqual(int(hours["outlier_count"]), 1)
        self.assertIn(40.0, hours["outlier_values"])
        self.assertLess(hours["lower_threshold"], hours["upper_threshold"])
        self.assertEqual(
            round(hours["iqr"], 4),
            round(hours["q3"] - hours["q1"], 4),
        )
    def test_correlations_use_numeric_columns_only(self) -> None:
        corr = numeric_correlations(self.df)
        self.assertIsNotNone(corr)
        assert corr is not None
        self.assertNotIn("student_id", corr.columns)
        self.assertIn("exam_score", corr.columns)
        self.assertIn("hours_studied", corr.columns)
    def test_sensitivity_drops_flagged_rows(self) -> None:
        flagged = row_has_iqr_outlier(self.df)
        reduced = drop_flagged_outlier_rows(self.df)
        self.assertEqual(len(reduced), len(self.df) - int(flagged.sum()))
        self.assertTrue((reduced["hours_studied"] != 40.0).all())
if __name__ == "__main__":
    unittest.main()