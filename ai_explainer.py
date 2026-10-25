"""Turn computed DataTrap results into a plain-English summary.
This layer only restates numbers the analyzer already produced. It does
not fetch outside evidence and should not be treated as a statistical audit.
"""
from __future__ import annotations
import pandas as pd
def summarize_findings(
    missing: pd.DataFrame,
    outliers: pd.DataFrame,
    correlations: pd.DataFrame | None,
    n_rows: int,
    n_outlier_rows: int,
) -> str:
    parts: list[str] = [
        f"This scan looked at {n_rows} rows.",
        "The notes below only describe what was computed from this CSV.",
    ]
    missing_cols = missing[missing["missing_count"] > 0]
    if missing_cols.empty:
        parts.append("No missing values were counted in the uploaded columns.")
    else:
        examples = ", ".join(
            f"{row.column} ({int(row.missing_count)} missing, {row.missing_pct}%)"
            for row in missing_cols.head(5).itertuples()
        )
        parts.append(
            f"Missing values were found in {len(missing_cols)} column(s): {examples}."
        )
        parts.append(
            "Missingness can change averages and correlations, so check how those "
            "fields were collected before trusting a chart."
        )
    if outliers.empty:
        parts.append("No numeric columns were available for an IQR outlier scan.")
    else:
        flagged = outliers[outliers["outlier_count"] > 0]
        if flagged.empty:
            parts.append(
                "No values fell outside the IQR fences (Q1 − 1.5×IQR to Q3 + 1.5×IQR)."
            )
        else:
            details = "; ".join(
                f"{row.column}: {int(row.outlier_count)} value(s) outside "
                f"[{row.lower_threshold}, {row.upper_threshold}]"
                for row in flagged.itertuples()
            )
            parts.append(
                f"{n_outlier_rows} row(s) had at least one IQR-flagged value. {details}."
            )
            parts.append(
                "An IQR flag is not proof of an error. Compare results with and "
                "without those rows as a sensitivity check, then investigate the source."
            )
    if correlations is None:
        parts.append(
            "Correlations were skipped because fewer than two numeric columns were present."
        )
    else:
        parts.append(
            "Numeric correlations were computed from overlapping non-missing values. "
            "Correlation is not causation."
        )
    return " ".join(parts)