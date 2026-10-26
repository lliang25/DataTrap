"""Deterministic data-quality checks; flags are not proof of errors or causation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if df.shape[1] == 0:
        raise ValueError("The dataset has no columns.")
    if len(df) == 0:
        raise ValueError("The dataset has no rows.")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    validate_dataframe(df)
    counts = df.isna().sum()
    return pd.DataFrame({
        "column": counts.index,
        "missing_count": counts.to_numpy(),
        "missing_pct": (counts / len(df) * 100).round(2).to_numpy(),
    }).sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def _iqr_checks(df: pd.DataFrame, factor: float = 1.5):
    """Share full-precision fences and masks between reports and row filtering."""
    validate_dataframe(df)
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("IQR factor must be finite and non-negative.")
    for column in numeric_columns(df):
        series = df[column].replace([np.inf, -np.inf], np.nan)
        finite = series.dropna()
        if finite.empty:
            continue
        q1, q3 = float(finite.quantile(.25)), float(finite.quantile(.75))
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask = (series.notna() & ((series < lower) | (series > upper))).fillna(False)
        yield column, q1, q3, iqr, lower, upper, mask


def iqr_outlier_report(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    rows = []
    for column, q1, q3, iqr, lower, upper, mask in _iqr_checks(df, factor):
        rows.append(dict(column=column, q1=q1, q3=q3, iqr=iqr,
                         lower_threshold=lower, upper_threshold=upper,
                         outlier_count=int(mask.sum()), outlier_values=df.loc[mask, column].tolist()))
    return pd.DataFrame(rows, columns=["column", "q1", "q3", "iqr", "lower_threshold",
                                       "upper_threshold", "outlier_count", "outlier_values"])


def row_has_iqr_outlier(df: pd.DataFrame, factor: float = 1.5) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for *_, column_mask in _iqr_checks(df, factor):
        mask |= column_mask
    return mask


def drop_flagged_outlier_rows(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    return df.loc[~row_has_iqr_outlier(df, factor)].copy()


def numeric_correlations(df: pd.DataFrame, min_periods: int = 3) -> pd.DataFrame | None:
    validate_dataframe(df)
    numeric = df[numeric_columns(df)].replace([np.inf, -np.inf], np.nan)
    return None if numeric.shape[1] < 2 else numeric.corr(min_periods=min_periods)


def pair_statistics(pair: pd.DataFrame) -> dict[str, Any]:
    complete = pair.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(complete)
    r = None
    reason = None
    if n < 3:
        reason = "Fewer than 3 complete finite observations."
    elif (complete.nunique() < 2).any():
        reason = "At least one selected column is constant."
    else:
        value = float(complete.iloc[:, 0].corr(complete.iloc[:, 1]))
        if math.isfinite(value):
            r = value
        else:
            reason = "Correlation is numerically undefined."
    return {"n": n, "pearson_r": r, "unavailable_reason": reason}


def correlation_sensitivity(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    validate_dataframe(df)
    if x == y or x not in numeric_columns(df) or y not in numeric_columns(df):
        raise ValueError("Select two distinct numeric columns.")
    pair = df[[x, y]]
    flags = row_has_iqr_outlier(pair)
    before, after = pair_statistics(pair), pair_statistics(pair.loc[~flags])
    return {"filter_scope": "IQR flags in the two selected columns only",
            "flagged_rows": int(flags.sum()), "before": before, "after": after,
            "delta_r": None if before["pearson_r"] is None or after["pearson_r"] is None
            else after["pearson_r"] - before["pearson_r"]}


def scan(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate-only model context: no raw cells, indices, or outlier values."""
    validate_dataframe(df)
    outliers = iqr_outlier_report(df)
    numeric = df[numeric_columns(df)]
    return {"n_rows": len(df), "n_columns": df.shape[1],
            "missing": missing_value_summary(df).to_dict(orient="records"),
            "nonfinite_count": int(numeric.isin([np.inf, -np.inf]).sum().sum()),
            "outliers": outliers.drop(columns="outlier_values").to_dict(orient="records"),
            "n_outlier_rows": int(row_has_iqr_outlier(df).sum())}
