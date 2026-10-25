"""Explain computed DataTrap findings with a local Ollama model."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from ollama import Client


def explain(report: dict[str, Any], skill: str) -> str:
    """Ask Ollama to explain a statistics report, without sending source rows."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("DATATRAP_MODEL", "llama3.2:1b")
    client = Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": skill},
            {
                "role": "user",
                "content": (
                    "Explain these precomputed findings for a beginner. "
                    "Do not invent statistics, claim causation, or treat outlier "
                    "flags as proven errors. Give practical questions to investigate.\n\n"
                    + json.dumps(report, indent=2, default=str)
                ),
            },
        ],
    )
    return response["message"]["content"].strip()


def summarize_findings(
    missing: pd.DataFrame,
    outliers: pd.DataFrame,
    correlations: pd.DataFrame | None,
    n_rows: int,
    n_outlier_rows: int,
) -> str:
    """Create a deterministic recap that works even when Ollama is unavailable."""
    parts = [f"This scan looked at {n_rows} rows."]
    missing_cols = missing[missing["missing_count"] > 0]
    if missing_cols.empty:
        parts.append("No missing values were counted in the uploaded columns.")
    else:
        examples = ", ".join(
            f"{row.column} ({int(row.missing_count)} missing, {row.missing_pct}%)"
            for row in missing_cols.head(5).itertuples()
        )
        parts.append(f"Missing values were found in {len(missing_cols)} column(s): {examples}.")
    if outliers.empty:
        parts.append("No numeric columns were available for an IQR outlier scan.")
    else:
        flagged = outliers[outliers["outlier_count"] > 0]
        if flagged.empty:
            parts.append("No values fell outside the IQR outlier thresholds.")
        else:
            parts.append(f"{n_outlier_rows} row(s) had at least one IQR-flagged value.")
            parts.append("An IQR flag is a clue to investigate, not proof of an error.")
    if correlations is None:
        parts.append("Correlations were skipped because fewer than two numeric columns were present.")
    else:
        parts.append("Numeric correlations were computed from overlapping non-missing values; correlation is not causation.")
    return " ".join(parts)
