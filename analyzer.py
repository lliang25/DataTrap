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