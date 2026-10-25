"""Starter harness: Python tools first, model explanation second.

Next hackathon enhancement: model-selected tool calls and session memory.
"""
from __future__ import annotations
import pandas as pd
from analyzer import (
    drop_flagged_outlier_rows,
    iqr_outlier_report,
    missing_value_summary,
    numeric_correlations,
    row_has_iqr_outlier,
)
from skill_loader import load_skill
from ai_explainer import explain


def analyze(df: pd.DataFrame, x: str | None = None, y: str | None = None) -> dict:
    missing = missing_value_summary(df)
    outliers = iqr_outlier_report(df)
    correlations = numeric_correlations(df)
    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": missing.to_dict(orient="records"),
        "iqr_outliers": outliers.to_dict(orient="records"),
        "rows_with_outlier_flags": int(row_has_iqr_outlier(df).sum()),
        "correlations": correlations.to_dict() if correlations is not None else None,
        "caveats": [
            "IQR outlier flags are investigation clues, not proven errors.",
            "Correlation does not establish causation.",
        ],
    }
    if x and y and x != y:
        if x not in df.columns or y not in df.columns:
            raise ValueError("Both selected columns must exist in the dataset.")
        if x not in df.select_dtypes(include="number") or y not in df.select_dtypes(include="number"):
            raise ValueError("Correlation sensitivity requires two numeric columns.")
        reduced = drop_flagged_outlier_rows(df)
        full_corr = df[[x, y]].corr(min_periods=3).iloc[0, 1]
        reduced_corr = reduced[[x, y]].corr(min_periods=3).iloc[0, 1]
        report["correlation_sensitivity"] = {
            "x": x,
            "y": y,
            "correlation_all_rows": None if pd.isna(full_corr) else float(full_corr),
            "correlation_without_flagged_rows": None if pd.isna(reduced_corr) else float(reduced_corr),
            "rows_all": len(df),
            "rows_without_flagged_rows": len(reduced),
        }
    return report


def explain_report(report: dict) -> str:
    return explain(report, load_skill())
