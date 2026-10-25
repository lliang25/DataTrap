from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / 'app.py'


def test_default_sample_starts():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert not app.error
    assert len(app.selectbox) == 2
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
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert 'Local AI unavailable' in app.error[0].value
    assert any(m.label == 'Pearson r' for m in app.metric)


def test_ai_result_persists_but_is_hidden_when_evidence_changes(monkeypatch):
    import agent
    monkeypatch.setattr(agent, 'explain_report', lambda report: 'Verified mocked explanation')
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.button[0].click().run(timeout=20)
    assert any(m.value == 'Verified mocked explanation' for m in app.markdown)
    app.run(timeout=20)
    assert any(m.value == 'Verified mocked explanation' for m in app.markdown)
    app.selectbox[1].select('attendance_pct').run(timeout=20)
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
    assert [m.value for m in app.metric if m.label == 'Complete pairs (n)'] == ['11', '10']
    assert [m.value for m in app.metric if m.label == 'Pearson r'] == ['-0.6443', '1.0000']
    missing = app.dataframe[1].value.set_index('column')
    assert missing.loc['exam_score', 'missing_count'] == 1
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
