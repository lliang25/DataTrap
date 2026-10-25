"""Local open-weight LLM explanation: not used for arithmetic."""
import json
import os


def explain(report: dict, skill: str) -> str:
    try:
        from ollama import Client
        client = Client(host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'), timeout=60)
        response = client.chat(model=os.getenv('DATATRAP_MODEL', 'llama3.2:1b'), messages=[
            {'role': 'system', 'content': skill + '\nOnly use values in the report. Explain uncertainty and give one useful next check.'},
            {'role': 'user', 'content': 'Explain this computed data-quality report:\n' + json.dumps(report, indent=2)},
        ])
        return response['message']['content']
    except Exception as exc:
        raise RuntimeError('Local Ollama explanation unavailable. Install Ollama, pull llama3.2:1b, and ensure it is running. Details: ' + str(exc)) from exc
