"""Starter harness: Python tools first, model explanation second.

Next hackathon enhancement: model-selected tool calls and session memory.
"""
from __future__ import annotations
import pandas as pd
from analyzer import scan, correlation_sensitivity
from skill_loader import load_skill
from ai_explainer import explain


def analyze(df: pd.DataFrame, x: str | None = None, y: str | None = None) -> dict:
    report = scan(df)
    if x and y and x != y:
        report['correlation_sensitivity'] = {'x': x, 'y': y, **correlation_sensitivity(df, x, y)}
    return report


def explain_report(report: dict) -> str:
    return explain(report, load_skill())
