"""Data-quality and statistics helpers for DataTrap.
These functions flag questions for investigation. They do not decide
whether an outlier is an error, and they do not establish causation.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
def validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if df.empty:
        raise ValueError("The dataset has no rows.")
    if df.shape[1] == 0:
        raise ValueError("The dataset has no columns.")
def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()
def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing values per column."""
    validate_dataframe(df)
    n_rows = len(df)
    counts = df.isna().sum()
    summary = pd.DataFrame(
        {
            "column": counts.index,
            "missing_count": counts.to_numpy(),
            "missing_pct": (counts / n_rows * 100).round(2).to_numpy(),
        }
    )
    return summary.sort_values(
        ["missing_count", "column"], ascending=[False, True]
    ).reset_index(drop=True)
def iqr_outlier_report(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Flag potential outliers with the interquartile-range method.
    A value is flagged when it is below Q1 - factor*IQR or above Q3 + factor*IQR.
    The report includes the thresholds so users can see how the flag was computed.
    """
    validate_dataframe(df)
    rows: list[dict[str, Any]] = []
    for column in numeric_columns(df):
        series = df[column].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        if iqr == 0:
            mask = series != series.median()
        else:
            mask = (series < lower) | (series > upper)
        flagged = series[mask]
        rows.append(
            {
                "column": column,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_threshold": round(lower, 4),
                "upper_threshold": round(upper, 4),
                "outlier_count": int(mask.sum()),
                "outlier_values": flagged.tolist(),
            }
        )
    return pd.DataFrame(rows)
def row_has_iqr_outlier(df: pd.DataFrame, factor: float = 1.5) -> pd.Series:
    """Return a boolean mask of rows that contain at least one IQR outlier."""
    validate_dataframe(df)
    mask = pd.Series(False, index=df.index)
    report = iqr_outlier_report(df, factor=factor)
    for _, row in report.iterrows():
        column = row["column"]
        lower = row["lower_threshold"]
        upper = row["upper_threshold"]
        series = df[column]
        column_mask = series.notna() & ((series < lower) | (series > upper))
        if row["iqr"] == 0:
            median = df[column].median()
            column_mask = series.notna() & (series != median)
        mask = mask | column_mask
    return mask
def drop_flagged_outlier_rows(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Sensitivity check: rows without IQR-flagged values.
    This is an exploration aid, not a recommendation to drop those rows.
    """
    mask = row_has_iqr_outlier(df, factor=factor)
    return df.loc[~mask].copy()
def numeric_correlations(
    df: pd.DataFrame, min_periods: int = 3
) -> pd.DataFrame | None:
    """Pearson correlations among numeric columns with enough overlapping values."""
    validate_dataframe(df)
    numeric = df[numeric_columns(df)]
    if numeric.shape[1] < 2:
        return None
    return numeric.corr(min_periods=min_periods)


def _json_number(value: Any) -> float | int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    number = float(value)
    if not pd.notna(number) or number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, 4)


def correlation_sensitivity(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    """Compare a pairwise correlation with IQR-flagged rows excluded.

    This is a sensitivity check, not evidence that the excluded rows are errors.
    """
    validate_dataframe(df)
    if x == y or x not in df.columns or y not in df.columns:
        raise ValueError("Choose two different existing columns.")
    pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        return {
            "n": len(pair),
            "correlation": None,
            "filtered_n": None,
            "filtered_correlation": None,
            "removed": 0,
        }
    corr = pair[x].corr(pair[y])
    flagged = row_has_iqr_outlier(df).reindex(pair.index).fillna(False).astype(bool)
    filtered = pair.loc[~flagged]
    corr2 = filtered[x].corr(filtered[y]) if len(filtered) >= 3 else None
    return {
        "n": int(len(pair)),
        "correlation": _json_number(corr),
        "filtered_n": int(len(filtered)),
        "filtered_correlation": _json_number(corr2),
        "removed": int(int(flagged.sum())),
    }


def scan(df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact, JSON-safe report of computed checks for the agent harness."""
    validate_dataframe(df)
    missing = missing_value_summary(df)
    outliers = iqr_outlier_report(df)
    outlier_rows = row_has_iqr_outlier(df)
    corr = numeric_correlations(df)
    outlier_payload: list[dict[str, Any]] = []
    if not outliers.empty:
        for rec in outliers.to_dict(orient="records"):
            values = rec.get("outlier_values") or []
            outlier_payload.append(
                {
                    "column": str(rec["column"]),
                    "q1": _json_number(rec["q1"]),
                    "q3": _json_number(rec["q3"]),
                    "iqr": _json_number(rec["iqr"]),
                    "lower_threshold": _json_number(rec["lower_threshold"]),
                    "upper_threshold": _json_number(rec["upper_threshold"]),
                    "outlier_count": int(rec["outlier_count"]),
                    "outlier_values": [_json_number(v) for v in values[:12]],
                }
            )
    correlations = None
    if corr is not None:
        correlations = {
            str(col): {str(other): _json_number(val) for other, val in row.items()}
            for col, row in corr.to_dict().items()
        }
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "n_outlier_rows": int(outlier_rows.sum()),
        "missing": [
            {
                "column": str(row["column"]),
                "missing_count": int(row["missing_count"]),
                "missing_pct": _json_number(row["missing_pct"]),
            }
            for row in missing.to_dict(orient="records")
        ],
        "iqr_outliers": outlier_payload,
        "correlations": correlations,
    }
