from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / 'app.py'


def test_default_sample_starts():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert not app.error
    assert {box.label for box in app.selectbox} == {"Sample dataset", "X numeric column", "Y numeric column"}
    assert any(m.label == 'Pearson r' for m in app.metric)


@pytest.mark.parametrize('frame', [pd.DataFrame({'value': [1, 2, 3]}),
                                   pd.DataFrame({'text': ['a', 'b']}),
                                   pd.DataFrame({'a': [None, None], 'b': [None, None]})])
def test_small_input_shapes(monkeypatch, frame):
    monkeypatch.setattr(pd, 'read_csv', lambda *args, **kwargs: frame.copy())
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert not app.error


@pytest.mark.parametrize('error', [pd.errors.EmptyDataError('No columns'),
                                  pd.errors.ParserError('Invalid CSV'),
                                  UnicodeError('Invalid encoding')])
def test_bad_csv_has_friendly_error(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(pd, 'read_csv', fail)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert 'Could not read CSV' in app.error[0].value


def test_ai_failure_keeps_statistics(monkeypatch):
    import agent
    def fail(report):
        raise ConnectionError('Ollama offline')
    monkeypatch.setattr(agent, 'explain_report', fail)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    next(b for b in app.button if b.label == 'Explain with local AI').click().run(timeout=20)
    assert not app.exception
    assert 'Local AI unavailable' in app.error[0].value
    assert any(m.label == 'Pearson r' for m in app.metric)


def test_ai_result_persists_but_is_hidden_when_evidence_changes(monkeypatch):
    import agent
    monkeypatch.setattr(agent, 'explain_report', lambda report: 'Verified mocked explanation')
    app = AppTest.from_file(str(APP)).run(timeout=20)
    next(b for b in app.button if b.label == 'Explain with local AI').click().run(timeout=20)
    assert any(m.value == 'Verified mocked explanation' for m in app.markdown)
    app.run(timeout=20)
    assert any(m.value == 'Verified mocked explanation' for m in app.markdown)
    next(b for b in app.selectbox if b.label == 'Y numeric column').select('attendance_pct').run(timeout=20)
    assert not app.exception
    assert not any(m.value == 'Verified mocked explanation' for m in app.markdown)


def test_student_scores_upload_bytes_are_parsed(monkeypatch):
    from io import BytesIO
    import streamlit as st
    upload = BytesIO((APP.parent / 'sample_data/student_scores.csv').read_bytes())
    upload.name = 'student_scores.csv'
    # Inject at the widget boundary; CSV parsing and all analysis remain real.
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: upload)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert not app.error
    assert app.checkbox[0].value is False
    assert [m.value for m in app.metric if m.label == 'Complete pairs (n)'] == ['18', '17']
    assert [m.value for m in app.metric if m.label == 'Pearson r'] == ['0.2325', '0.9666']
    missing = app.dataframe[1].value.set_index('column')
    assert missing.loc['exam_score', 'missing_count'] == 0
    assert missing.loc['hours_studied', 'missing_count'] == 1
    assert missing.loc['practice_tests', 'missing_count'] == 1
    outliers = app.dataframe[2].value.set_index('column')
    assert outliers.loc['hours_studied', 'outlier_count'] == 1


def test_all_nonfinite_recap_does_not_claim_no_numeric_columns(monkeypatch):
    frame = pd.DataFrame({'x': [float('inf'), float('nan')]})
    monkeypatch.setattr(pd, 'read_csv', lambda *args, **kwargs: frame.copy())
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert any('No finite numeric observations' in m.value for m in app.markdown)


def test_ask_agent_renders_trace_without_breaking_dashboard(monkeypatch):
    import agent
    seen = []
    def fake_run(frame, question):
        seen.append(question)
        return {'answer': 'One value is missing.', 'steps': 2, 'stop_reason': 'completed',
                'model': 'test-model', 'skill_sha256': 'test-skill',
                'trace': [{'step': 1, 'tool': 'missing_value_detection', 'input': {},
                           'status': 'ok', 'output': {'missing_count': 1}}]}
    monkeypatch.setattr(agent, 'run_agent', fake_run)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.text_input[0].set_value('Which values are missing?')
    next(b for b in app.button if b.label == 'Run Agent').click().run(timeout=20)
    assert not app.exception
    assert seen == ['Which values are missing?']
    assert any(m.value == 'One value is missing.' for m in app.markdown)
    assert any('missing_value_detection' in e.label for e in app.expander)
    assert any(m.label == 'Pearson r' for m in app.metric)


def test_ask_agent_empty_question_has_friendly_error():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    next(b for b in app.button if b.label == 'Run Agent').click().run(timeout=20)
    assert not app.exception
    assert 'Enter a question' in app.error[0].value


def test_detective_caches_results_and_clears_changed_dataset(monkeypatch):
    import detective
    calls = []
    original = detective.investigate_dataset
    class Offline:
        def chat(self, **kwargs):
            raise ConnectionError('Offline')
    def investigate(frame):
        calls.append(frame.copy())
        return original(frame, client=Offline())
    monkeypatch.setattr(detective, 'investigate_dataset', investigate)
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not calls
    next(b for b in app.button if b.label == 'Investigate Dataset').click().run(timeout=20)
    assert not app.exception
    assert len(calls) == 1
    assert any('Partial report' in w.value for w in app.warning)
    app.run(timeout=20)
    next(b for b in app.button if b.label == 'Investigate Dataset').click().run(timeout=20)
    assert len(calls) == 1
    assert any(b.label == 'Run Agent' for b in app.button)
    next(b for b in app.button if b.label == 'Clear Detective report / enable retry').click().run(timeout=20)
    assert 'detective_result' not in app.session_state
    next(b for b in app.button if b.label == 'Investigate Dataset').click().run(timeout=20)
    assert len(calls) == 2
    monkeypatch.setattr(pd, 'read_csv', lambda *args, **kwargs: pd.DataFrame({'changed': [1, 2, 3]}))
    app.run(timeout=20)
    assert not app.exception
    assert 'detective_result' not in app.session_state
    assert len(calls) == 2


def test_lavender_hero_and_sample_selector():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert any('data-mascot-state="surprised"' in m.value for m in app.markdown)
    next(b for b in app.selectbox if b.label == 'Sample dataset').select('Student scores · student_scores.csv').run(timeout=20)
    assert not app.exception
    assert any('data-mascot-state="confused"' in m.value for m in app.markdown)
    assert any(m.label == 'Rows' and m.value == '20' for m in app.metric)


@pytest.mark.parametrize('frame,expected', [
    (pd.DataFrame({'x': [1, 2, 3, 4]}), 'happy'),
    (pd.DataFrame({'x': [1, 2, 3], 'y': [1, 1, 1]}), 'neutral'),
    (pd.DataFrame({'text': ['a', 'b']}), 'neutral'),
    (pd.DataFrame({'x': [1, 2, float('inf')]}), 'neutral'),
    (pd.DataFrame({'x': [1, None, 3]}), 'confused'),
])
def test_hero_uses_verified_analysis_state(monkeypatch, frame, expected):
    monkeypatch.setattr(pd, 'read_csv', lambda *args, **kwargs: frame.copy())
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert any(f'data-mascot-state="{expected}"' in m.value for m in app.markdown)


def test_detective_report_is_readable_and_technical_evidence_retained(monkeypatch):
    import detective
    import json
    original = detective.investigate_dataset
    class Model:
        def chat(self, **kwargs):
            candidate = json.loads(kwargs['messages'][1]['content'])['candidates'][0]
            return {'message': {'content': json.dumps({'hypothesis_id': candidate['id'],
                    'tool': candidate['tool'], 'arguments': candidate['arguments']})}}
    monkeypatch.setattr(detective, 'investigate_dataset', lambda frame: original(frame, Model()))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    next(b for b in app.button if b.label == 'Investigate Dataset').click().run(timeout=20)
    assert not app.exception
    rendered = '\n'.join(m.value for m in app.markdown)
    for title in ['Scan Summary', 'Selected Hypothesis', 'Investigation Result', 'Verified Evidence', 'Suggested Next Steps', 'Limitations']:
        assert title in rendered
    assert any('full report JSON' in e.label for e in app.expander)
    assert any(e.label == 'Detective Tool Execution Trace' for e in app.expander)
    assert any(b.label == 'Run Agent' for b in app.button)


def test_custom_html_escapes_dataset_labels():
    # Exercise the reusable card renderer through Streamlit's test runtime.
    code = '''
from ui_components import card
card('<script>alert(1)</script>', '<img src=x onerror=alert(1)>')
'''
    app = AppTest.from_string(code).run(timeout=20)
    assert not app.exception
    markup = app.markdown[0].value
    assert '<script>' not in markup and '<img src=x' not in markup
    assert '&lt;script&gt;' in markup
