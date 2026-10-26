"""Read-only, allowlisted analysis tools. Model text is never executed as code."""
from __future__ import annotations

import pandas as pd

from analyzer import (missing_value_summary, iqr_outlier_report, numeric_columns,
                      row_has_iqr_outlier, pair_statistics, correlation_sensitivity)

TOOL_DESCRIPTIONS = {
    'missing_value_detection': 'Count missing values. Arguments: columns (optional list of up to 20 column names). Omit columns to inspect all columns, if at most 20.',
    'outlier_detection': 'Count IQR flags and show fences, without raw values. Arguments: columns (optional list of up to 20 numeric column names).',
    'correlation_analysis': 'Pearson r, effective n, and unavailable reason for one pair. Arguments: x and y (distinct numeric column names). Does not filter outliers.',
    'sensitivity_analysis': 'Compare Pearson r and n before/after IQR flags in the selected pair; includes delta and flagged row count. Arguments: x and y (distinct numeric column names).',
}


def _columns(df: pd.DataFrame, args: dict, numeric: bool) -> list[str]:
    if set(args) - {'columns'}:
        raise ValueError('Only the columns argument is allowed for this tool.')
    allowed = numeric_columns(df) if numeric else df.columns.tolist()
    columns = args.get('columns', allowed)
    if not isinstance(columns, list) or not columns or len(columns) > 20:
        raise ValueError('Select between 1 and 20 columns.')
    if any(not isinstance(c, str) or c not in allowed for c in columns):
        raise ValueError('Unknown column or unsupported column type. Use the dataset schema.')
    if len(set(columns)) != len(columns):
        raise ValueError('Column names must be unique.')
    return columns


def _pair(df: pd.DataFrame, args: dict) -> tuple[str, str]:
    if set(args) != {'x', 'y'}:
        raise ValueError('Exactly x and y are required.')
    x, y = args['x'], args['y']
    allowed = numeric_columns(df)
    if not isinstance(x, str) or not isinstance(y, str) or x == y or x not in allowed or y not in allowed:
        raise ValueError('x and y must be distinct numeric columns from the dataset schema.')
    return x, y


def _missing(df, args):
    columns = _columns(df, args, numeric=False)
    return {'n_rows': len(df), 'columns': missing_value_summary(df[columns]).to_dict(orient='records')}


def _outliers(df, args):
    columns = _columns(df, args, numeric=True)
    subset = df[columns]
    return {'n_rows': len(df), 'columns': iqr_outlier_report(subset).drop(columns='outlier_values').to_dict(orient='records'),
            'flagged_rows': int(row_has_iqr_outlier(subset).sum()),
            'method': '1.5 × IQR; finite observations only; flags are not proven errors'}


def _correlation(df, args):
    x, y = _pair(df, args)
    return {'x': x, 'y': y, **pair_statistics(df[[x, y]])}


def _sensitivity(df, args):
    x, y = _pair(df, args)
    return {'x': x, 'y': y, **correlation_sensitivity(df, x, y)}


# Resolve exclusively through this registry: never eval, exec, shell, or dynamic import.
TOOLS = {
    'missing_value_detection': _missing,
    'outlier_detection': _outliers,
    'correlation_analysis': _correlation,
    'sensitivity_analysis': _sensitivity,
}


def execute_tool(name: str, arguments: dict, df: pd.DataFrame) -> dict:
    if not isinstance(name, str) or name not in TOOLS:
        raise ValueError('Unknown tool. Only the four listed analysis tools are allowed.')
    if not isinstance(arguments, dict):
        raise ValueError('Tool arguments must be a JSON object.')
    return TOOLS[name](df, arguments)
