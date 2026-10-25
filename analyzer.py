"""Deterministic, auditable CSV checks; no LLM calculations."""
from __future__ import annotations
import math
import pandas as pd


def missing_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df.isna().sum().items()}


def iqr_outliers(df: pd.DataFrame) -> dict[str, dict]:
    result = {}
    for col in df.select_dtypes(include='number').columns:
        values = df[col].dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        if not math.isfinite(iqr) or iqr == 0:
            continue
        lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        count = int(((values < lower) | (values > upper)).sum())
        result[str(col)] = {'count': count, 'lower': round(lower, 4), 'upper': round(upper, 4)}
    return result


def correlation_sensitivity(df: pd.DataFrame, x: str, y: str) -> dict:
    if x == y or x not in df or y not in df:
        raise ValueError('Choose two different existing columns')
    pair = df[[x, y]].dropna().copy()
    if len(pair) < 3:
        return {'n': len(pair), 'correlation': None, 'filtered_n': None, 'filtered_correlation': None, 'removed': 0}
    corr = pair[x].corr(pair[y])
    mask = pd.Series(True, index=pair.index)
    for col in (x, y):
        q1, q3 = pair[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            mask &= pair[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    filtered = pair[mask]
    corr2 = filtered[x].corr(filtered[y]) if len(filtered) >= 3 else float('nan')
    def safe(value):
        return round(float(value), 4) if pd.notna(value) and math.isfinite(float(value)) else None
    return {'n': len(pair), 'correlation': safe(corr), 'filtered_n': len(filtered),
            'filtered_correlation': safe(corr2), 'removed': len(pair) - len(filtered)}


def scan(df: pd.DataFrame) -> dict:
    return {'rows': len(df), 'columns': len(df.columns),
            'missing': missing_counts(df), 'iqr_outliers': iqr_outliers(df)}
