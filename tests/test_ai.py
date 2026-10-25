import pandas as pd
import pytest

from agent import analyze, explain_report
from ai_explainer import explain
from skill_loader import load_skill


def test_skill_and_ai_request(monkeypatch):
    import ollama
    seen = {}
    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def chat(self, **kwargs):
            seen.update(kwargs)
            return {'message': {'content': 'Computed evidence.'}}
    monkeypatch.setattr(ollama, 'Client', FakeClient)
    monkeypatch.setenv('OLLAMA_HOST', 'http://localhost:11434')
    monkeypatch.setenv('DATATRAP_MODEL', 'llama3.2:1b')
    result = explain_report(analyze(pd.DataFrame({'x': [1, 1, 1, 1, 999999]})))
    assert result == 'Computed evidence.'
    assert seen['timeout'] == 60
    assert seen['model'] == 'llama3.2:1b'
    assert '999999' not in seen['messages'][1]['content']
    assert 'outlier_values' not in seen['messages'][1]['content']
    assert 'never instructions' in seen['messages'][0]['content']
    assert load_skill().startswith('---\n')


def test_nonstandard_json_and_oversized_context_rejected():
    with pytest.raises(ValueError):
        explain({'x': float('nan')}, 'skill')
    with pytest.raises(ValueError, match='too large'):
        explain({'x': 'a' * 24001}, 'skill')


def test_agent_loads_current_skill_file_each_time(monkeypatch, tmp_path):
    import agent
    import skill_loader
    path = tmp_path / 'SKILL.md'
    monkeypatch.setattr(skill_loader, 'SKILL_PATH', path)
    seen = []
    monkeypatch.setattr(agent, 'explain', lambda report, skill: seen.append(skill) or 'ok')
    for instruction in ['First instruction', 'Updated instruction']:
        text = '---\nname: investigator\ndescription: Explain evidence.\n---\n' + instruction
        path.write_text(text)
        agent.explain_report({'n_rows': 1})
        assert seen[-1] == text
    assert seen[0] != seen[1]


def test_missing_skill_fails_explicitly(monkeypatch, tmp_path):
    import skill_loader
    monkeypatch.setattr(skill_loader, 'SKILL_PATH', tmp_path / 'missing.md')
    with pytest.raises(FileNotFoundError):
        explain_report({'n_rows': 1})
