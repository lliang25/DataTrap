"""Opt-in tests that call the real locally installed Ollama model.

Run: DATATRAP_TEST_OLLAMA=1 python -m pytest -q -s tests/test_ollama_integration.py
Only bundled synthetic data is used. No model is downloaded by these tests.
"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from agent import analyze, explain_report
from skill_loader import load_skill


@pytest.mark.skipif(os.getenv('DATATRAP_TEST_OLLAMA') != '1', reason='Set DATATRAP_TEST_OLLAMA=1 to call local Ollama')
def test_real_model_generates_different_responses_and_receives_skill(monkeypatch):
    import ollama
    real_client = ollama.Client
    exchanges = []

    class AuditedClient:
        def __init__(self, **kwargs):
            # Explicitly keep this test local even if a user's normal host is remote.
            assert kwargs['host'] == 'http://localhost:11434'
            self.client = real_client(**kwargs)

        def chat(self, **kwargs):
            response = self.client.chat(**kwargs)  # Real HTTP request, not a mock response.
            exchanges.append((kwargs, response))
            return response

    monkeypatch.setenv('OLLAMA_HOST', 'http://localhost:11434')
    monkeypatch.setenv('DATATRAP_MODEL', 'llama3.2:1b')
    monkeypatch.setattr(ollama, 'Client', AuditedClient)
    df = pd.read_csv(Path(__file__).parents[1] / 'sample_data/student_scores.csv')
    reports = [analyze(df, 'hours_studied', 'exam_score'),
               analyze(df.iloc[:10], 'hours_studied', 'exam_score')]
    outputs = [explain_report(report) for report in reports]
    assert outputs[0] != outputs[1]
    assert len(exchanges) == 2
    skill = load_skill()
    evidence = []
    for request, response in exchanges:
        assert request['messages'][0]['content'].startswith(skill)
        assert request['model'] == 'llama3.2:1b'
        assert response.done
        assert response.eval_count > 0
        assert response.prompt_eval_count > 0
        assert response.message.content.strip()
        evidence.append({'model': response.model, 'created_at': str(response.created_at),
                         'done': response.done, 'eval_count': response.eval_count,
                         'prompt_eval_count': response.prompt_eval_count,
                         'skill_sha256': hashlib.sha256(skill.encode()).hexdigest(),
                         'output': response.message.content})
    assert exchanges[0][0]['messages'][1]['content'] != exchanges[1][0]['messages'][1]['content']
    print('\nREAL_OLLAMA_EVIDENCE=' + json.dumps(evidence, ensure_ascii=False))


@pytest.mark.skipif(os.getenv('DATATRAP_TEST_OLLAMA') != '1', reason='Set DATATRAP_TEST_OLLAMA=1 to call local Ollama')
def test_real_agent_selects_tools_and_skill_changes_first_choice(monkeypatch, tmp_path):
    import agent
    import skill_loader
    import ollama
    real_client = ollama.Client
    generation_counts = []
    class AuditedClient:
        def __init__(self):
            self.client = real_client(host='http://localhost:11434', timeout=60)
        def chat(self, **kwargs):
            response = self.client.chat(**kwargs)
            assert response.done and response.eval_count > 0
            generation_counts.append(response.eval_count)
            return response
    monkeypatch.setenv('DATATRAP_AGENT_MODEL', 'qwen3:1.7b')
    df = pd.read_csv(Path(__file__).parents[1] / 'sample_data/student_scores.csv')
    result = agent.run_agent(df, 'Which columns have missing values?', client=AuditedClient())
    assert result['stop_reason'] == 'completed'
    assert [t['tool'] for t in result['trace']] == ['missing_value_detection']
    result = agent.run_agent(df, 'Do outliers affect the correlation between hours_studied and exam_score?', client=AuditedClient())
    assert result['stop_reason'] == 'completed'
    assert [t['tool'] for t in result['trace']] == ['correlation_analysis', 'sensitivity_analysis']
    # Counterfactual: same dataset and question, only the loaded Skill policy differs.
    original_skill = skill_loader.load_skill()
    first_tools = []
    skill_path = tmp_path / 'SKILL.md'
    monkeypatch.setattr(skill_loader, 'SKILL_PATH', skill_path)
    for first_tool in ['missing_value_detection', 'outlier_detection']:
        policy = original_skill + f'\nFor this audit, override the broad-audit starting rule: your FIRST action MUST be {first_tool} with empty arguments.\n'
        skill_path.write_text(policy, encoding='utf-8')
        result = agent.run_agent(df, 'Please audit the quality of this dataset.', max_steps=1, client=AuditedClient())
        assert result['trace'][0]['status'] == 'ok'
        first_tools.append(result['trace'][0]['tool'])
    assert first_tools == ['missing_value_detection', 'outlier_detection']
    print('\nREAL_AGENT_EVIDENCE=' + json.dumps({'generated_tokens_per_decision': generation_counts,
          'skill_policy_first_tools': first_tools}))
