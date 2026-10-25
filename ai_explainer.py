"""Explain computed DataTrap findings with a local Ollama model."""
from __future__ import annotations

import json
import os
from typing import Any

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
