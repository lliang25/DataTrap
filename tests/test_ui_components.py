from pathlib import Path
import base64
import hashlib
import math

import pytest

from ui_components import (mascot_state, verified_sign_reversal, mascot_html,
                           verified_sensitivities_from_trace)


def sensitivity(a, b, flagged=1):
    return {'before': {'pearson_r': a}, 'after': {'pearson_r': b}, 'flagged_rows': flagged}


def test_happy_requires_completed_clean_scan():
    assert mascot_state(missing_count=0, flagged_rows=0, complete=True) == 'happy'
    assert mascot_state(missing_count=0, flagged_rows=0, complete=False) == 'neutral'
    assert mascot_state(missing_count=0, flagged_rows=0, complete=True, warning=True) == 'neutral'
    assert mascot_state() == 'neutral'


@pytest.mark.parametrize('missing,flagged', [(1, 0), (0, 2), (2, 3)])
def test_confused_when_scan_detects_issues(missing, flagged):
    assert mascot_state(missing_count=missing, flagged_rows=flagged, complete=True) == 'confused'


@pytest.mark.parametrize('a,b', [(-.6, .9), (.5, -.2)])
def test_surprised_has_priority_over_known_issues(a, b):
    assert mascot_state(missing_count=2, flagged_rows=3, complete=True,
                        sensitivities=[sensitivity(a, b)]) == 'surprised'


@pytest.mark.parametrize('a,b,flagged', [
    (0, .8, 1), (-.3, 0, 1), (.2, .8, 1), (-.8, -.2, 1),
    (-.3, .8, 0), (-.3, .8, -1), (-.3, .8, None),
    (None, .8, 1), (-.3, None, 1), (math.nan, .8, 1),
    (-.3, math.inf, 1), (-math.inf, .8, 1), ('-.3', .8, 1),
    (-.3, True, 1), (-.3, .8, math.nan),
])
def test_no_false_sign_reversal(a, b, flagged):
    assert not verified_sign_reversal(sensitivity(a, b, flagged))
    assert mascot_state(missing_count=1, flagged_rows=1, sensitivities=[sensitivity(a, b, flagged)]) == 'confused'


@pytest.mark.parametrize('value', [None, {}, 'sign reversal!', {'before': None, 'after': {}}])
def test_malformed_sensitivity_is_not_surprised(value):
    assert not verified_sign_reversal(value)


def test_model_prose_and_failed_tools_cannot_control_mascot():
    trace = [
        {'tool': 'sensitivity_analysis', 'status': 'error', 'output': sensitivity(-.4, .8)},
        {'tool': 'correlation_analysis', 'status': 'ok', 'output': sensitivity(-.4, .8)},
        {'tool': None, 'status': 'ok', 'output': {'answer': 'There is a sign reversal!'}},
    ]
    assert verified_sensitivities_from_trace(trace) == []
    trace.append({'tool': 'sensitivity_analysis', 'status': 'ok', 'output': sensitivity(-.4, .8)})
    assert verified_sensitivities_from_trace(trace) == [sensitivity(-.4, .8)]


def test_assets_are_original_bytes_and_markup_preserves_aspect_ratio():
    hashes = {
        'happy': '12349102e6cf4c8f5eb8e6ff03a3cf9722d77dd98c36addc11f0cce84d29e2ef',
        'confused': '5283fb4b52a388c8624ef3c9d0153efec0ed099b50c6bcdad863e1706c3200ed',
        'surprised': '4ddb9069ce6f1e68b6b69e8383e3ff1fd660b830668355ae19580b8d71b9b152',
    }
    for state, expected in hashes.items():
        markup = mascot_html(state)
        data = base64.b64decode(markup.split('base64,')[1].split('"')[0])
        assert hashlib.sha256(data).hexdigest() == expected
        assert f'alt="DataTrap mascot: {state}"' in markup
    css = (Path(__file__).parents[1] / 'assets/lavender.css').read_text()
    assert 'prefers-reduced-motion:reduce' in css
    assert 'animation:none !important' in css
    assert 'object-fit:contain' in css


def test_neutral_and_missing_asset_fallback(tmp_path):
    assert '<img' not in mascot_html('neutral')
    assert 'image unavailable' in mascot_html('happy', asset_dir=tmp_path)
    assert '<img' not in mascot_html('happy', asset_dir=tmp_path)
