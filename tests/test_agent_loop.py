import copy
import json

import pandas as pd
import pytest

import agent
from agent_tools import execute_tool


@pytest.fixture
def df():
    return pd.DataFrame({'hours': [1, 2, 3, 4, 50], 'score': [2, 4, 6, 8, -50], 'name': ['a', None, 'b', 'c', 'd']})


def action(name, arguments=None, answer=''):
    return {'action': name, 'arguments': arguments or {}, 'answer': answer}


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return {'message': {'content': value if isinstance(value, str) else json.dumps(value)}}


def test_only_selected_tool_runs_and_finish_stops(df, monkeypatch):
    calls = []
    original = agent.execute_tool
    def spy(name, args, frame):
        calls.append(name)
        return original(name, args, frame)
    monkeypatch.setattr(agent, 'execute_tool', spy)
    client = ScriptedClient([action('missing_value_detection'), action('finish', answer='One name is missing.')])
    result = agent.run_agent(df, 'Which columns are missing values?', client=client)
    assert calls == ['missing_value_detection']
    assert result['stop_reason'] == 'completed'
    assert result['steps'] == 2
    assert result['trace'][0]['output']['columns'][0]['missing_count'] == 1
    assert 'missing_count' in client.requests[1]['messages'][-1]['content']
    assert 'outlier_values' not in json.dumps(client.requests)


def test_multiple_steps_use_previous_observation(df):
    class AdaptiveClient(ScriptedClient):
        def chat(self, **kwargs):
            if len(self.requests) == 1:
                observation = kwargs['messages'][-1]['content']
                assert 'pearson_r' in observation and 'correlation_analysis' in observation
            return super().chat(**kwargs)
    client = AdaptiveClient([action('correlation_analysis', {'x': 'hours', 'y': 'score'}),
                             action('sensitivity_analysis', {'x': 'hours', 'y': 'score'}),
                             action('finish', answer='The relationship changes after filtering.')])
    result = agent.run_agent(df, 'Do outliers change the correlation?', client=client)
    assert [t['tool'] for t in result['trace']] == ['correlation_analysis', 'sensitivity_analysis']
    assert result['trace'][1]['output']['after']['pearson_r'] == pytest.approx(1)
    assert result['steps'] == 3


def test_bad_column_can_be_corrected(df):
    client = ScriptedClient([action('correlation_analysis', {'x': 'not_a_column', 'y': 'score'}),
                             action('correlation_analysis', {'x': 'hours', 'y': 'score'}),
                             action('finish', answer='Here is the observed correlation.')])
    result = agent.run_agent(df, 'Relationship?', client=client)
    assert [t['status'] for t in result['trace']] == ['tool_error', 'ok']
    assert 'error' in client.requests[1]['messages'][-1]['content']


@pytest.mark.parametrize('bad', ['not json', '[1,2]', '{"action":"exec","arguments":{},"answer":""}',
                                '{"action":"finish","arguments":{},"answer":""}',
                                '{"action":"finish","action":"finish","arguments":{},"answer":"x"}'])
def test_malformed_or_unsafe_action_is_never_executed(df, bad, monkeypatch):
    def forbidden(*args):
        pytest.fail('Invalid action was executed')
    monkeypatch.setattr(agent, 'execute_tool', forbidden)
    client = ScriptedClient([bad] * 5)
    result = agent.run_agent(df, 'Analyze', client=client)
    assert result['stop_reason'] == 'step_limit'
    assert len(client.requests) == result['steps'] == 5
    assert all(t['status'] == 'invalid_action' for t in result['trace'])


def test_five_step_cap_includes_tools_and_no_extra_summary_call(df):
    client = ScriptedClient([action('missing_value_detection', {'columns': ['name']}),
                             action('outlier_detection', {'columns': ['hours']}),
                             action('outlier_detection', {'columns': ['score']}),
                             action('correlation_analysis', {'x': 'hours', 'y': 'score'}),
                             action('sensitivity_analysis', {'x': 'hours', 'y': 'score'})])
    result = agent.run_agent(df, 'Inspect the data', client=client)
    assert result['stop_reason'] == 'step_limit'
    assert len(result['trace']) == len(client.requests) == 5
    assert all(t['status'] == 'ok' for t in result['trace'])


def test_duplicate_tool_not_executed_twice(df):
    client = ScriptedClient([action('missing_value_detection')] * 5)
    result = agent.run_agent(df, 'Missing?', client=client)
    assert [t['status'] for t in result['trace']] == ['ok'] + ['tool_error'] * 4


def test_transport_error_stops_and_preserves_evidence(df):
    client = ScriptedClient([action('missing_value_detection'), TimeoutError('Ollama timeout')])
    result = agent.run_agent(df, 'Missing?', client=client)
    assert result['stop_reason'] == 'model_error'
    assert result['trace'][0]['status'] == 'ok'
    assert result['trace'][1]['status'] == 'model_error'
    assert len(client.requests) == 2


def test_finish_without_tools_is_marked_no_evidence(df):
    result = agent.run_agent(df, 'What can you do?', client=ScriptedClient([action('finish', answer='Please choose a column.')]))
    assert result['stop_reason'] == 'no_evidence'
    assert result['trace'] == []


@pytest.mark.parametrize('limit', [0, 6, True, -1])
def test_invalid_budget_rejected(df, limit):
    with pytest.raises(ValueError):
        agent.run_agent(df, 'Question', max_steps=limit)


def test_skill_is_in_decision_prompt_and_changes_next_run(df, monkeypatch):
    for skill in ['Start with missing_value_detection.', 'Start with outlier_detection.']:
        monkeypatch.setattr(agent, 'load_skill', lambda: skill)
        client = ScriptedClient([action('finish', answer='Clarification needed.')])
        agent.run_agent(df, 'Audit', client=client)
        assert client.requests[0]['messages'][0]['content'].startswith(skill)
        # No precomputed scan is passed to the planner.
        assert 'missing_count' not in client.requests[0]['messages'][1]['content']


@pytest.mark.parametrize('tool,args', [
    ('python', {'code': 'print(1)'}),
    ('missing_value_detection', {'path': '/etc/passwd'}),
    ('outlier_detection', {'columns': ['name']}),
    ('correlation_analysis', {'x': 'hours', 'y': 'hours'}),
    ('sensitivity_analysis', {'x': 'hours', 'y': 'score', 'code': 'anything'}),
])
def test_tool_allowlist_and_arguments(df, tool, args):
    before = df.copy(deep=True)
    with pytest.raises(ValueError):
        execute_tool(tool, args, df)
    pd.testing.assert_frame_equal(df, before)


def test_all_tools_read_only_and_no_raw_outlier_values(df):
    before = df.copy(deep=True)
    for tool, args in [('missing_value_detection', {}), ('outlier_detection', {}),
                       ('correlation_analysis', {'x': 'hours', 'y': 'score'}),
                       ('sensitivity_analysis', {'x': 'hours', 'y': 'score'})]:
        output = execute_tool(tool, args, df)
        assert 'outlier_values' not in json.dumps(output, allow_nan=False)
        pd.testing.assert_frame_equal(df, before)
