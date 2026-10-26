"""Lavender Lab presentation components. No statistical calculations or model calls."""
from __future__ import annotations

import base64
import math
from functools import lru_cache
from html import escape
from numbers import Real
from pathlib import Path

import streamlit as st

ASSET_DIR = Path(__file__).parent / 'assets' / 'mascots'
STATES = {
    'happy': ('Looking good', 'No missing values or IQR flags detected in this completed scan. Keep the collection context in mind.'),
    'confused': ('A few clues to investigate', 'Missing values or unusual observations deserve a closer look. A flag is a question, not a verdict.'),
    'surprised': ('A plot twist in the evidence', 'A verified sensitivity comparison reversed the correlation’s sign after flagged observations were excluded.'),
    'neutral': ('Investigation in progress', 'Add a dataset or review the analysis limitations. There is not enough complete evidence to give an all-clear.'),
}


def finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def verified_sign_reversal(sensitivity):
    """Accept computed sensitivity results only; never inspect model prose."""
    if not isinstance(sensitivity, dict):
        return False
    before, after = sensitivity.get('before'), sensitivity.get('after')
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    a, b, count = before.get('pearson_r'), after.get('pearson_r'), sensitivity.get('flagged_rows')
    return (finite_number(a) and finite_number(b) and finite_number(count) and count > 0
            and ((a < 0 < b) or (b < 0 < a)))


def mascot_state(*, missing_count=None, flagged_rows=None, sensitivities=(), complete=False, warning=False):
    if any(verified_sign_reversal(item) for item in sensitivities):
        return 'surprised'
    if ((finite_number(missing_count) and missing_count > 0)
            or (finite_number(flagged_rows) and flagged_rows > 0)):
        return 'confused'
    if (not complete or warning or not finite_number(missing_count) or not finite_number(flagged_rows)
            or missing_count < 0 or flagged_rows < 0):
        return 'neutral'
    return 'happy'


def verified_sensitivities_from_trace(trace):
    """Only executed, successful sensitivity tools qualify; model answer is ignored."""
    return [entry['output'] for entry in trace
            if entry.get('tool') == 'sensitivity_analysis' and entry.get('status') == 'ok'
            and isinstance(entry.get('output'), dict)]


@lru_cache(maxsize=3)
def _image_data(path, modified):
    return base64.b64encode(Path(path).read_bytes()).decode('ascii')


def mascot_html(state, asset_dir=ASSET_DIR):
    state = state if state in STATES else 'neutral'
    if state == 'neutral':
        return '<div class="dt-neutral" role="status"><span class="dt-orbit" aria-hidden="true">◎</span><span>Ready to investigate</span></div>'
    path = Path(asset_dir) / f'{state}.png'
    try:
        data = _image_data(str(path), path.stat().st_mtime_ns)
    except OSError:
        return '<div class="dt-neutral" role="status">Mascot image unavailable · analysis is still available.</div>'
    return f'<img class="dt-mascot dt-{state}" src="data:image/png;base64,{data}" alt="DataTrap mascot: {state}" />'


def render_header(state):
    title, description = STATES[state]
    st.markdown(f'''<div class="dt-brand"><span class="dt-logo" aria-hidden="true">D<span>?</span></span><strong>DataTrap</strong><span class="dt-brand-note">LAVENDER LAB</span><span class="dt-local">Local AI · Open weights</span></div>
<section class="dt-hero" aria-label="DataTrap overview"><div class="dt-hero-copy"><span class="dt-eyebrow">YOUR AI DATA DETECTIVE</span><h1>Every dataset<br>has a story.</h1><p class="dt-intro">Find the missing pieces. Question the unexpected.<br>Let the evidence lead the way.</p><div class="dt-state" data-mascot-state="{state}"><strong>{escape(title)}</strong><p>{escape(description)}</p></div><nav class="dt-nav" aria-label="Dashboard sections"><a href="#detective-mode">Investigate dataset ↗</a><a href="#ask-datatrap">Ask a question →</a></nav></div><div class="dt-hero-art">{mascot_html(state)}</div></section>''', unsafe_allow_html=True)


def inject_styles():
    st.markdown('<style>' + (Path(__file__).parent / 'assets' / 'lavender.css').read_text() + '</style>', unsafe_allow_html=True)


def section(number, title, subtitle, anchor=None):
    identifier = f' id="{escape(anchor, quote=True)}"' if anchor else ''
    st.markdown(f'<div class="dt-section"{identifier}><span class="dt-section-number">{escape(number)}</span><div><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div></div>', unsafe_allow_html=True)


def card(title, body, label='VERIFIED EVIDENCE'):
    st.markdown(f'<article class="dt-card"><span class="dt-eyebrow">{escape(label)}</span><h3>{escape(str(title))}</h3><p>{escape(str(body))}</p></article>', unsafe_allow_html=True)


def format_value(value):
    if value is None or (isinstance(value, Real) and not finite_number(value)):
        return 'Unavailable'
    if finite_number(value):
        return str(int(value)) if value == int(value) else f'{value:.4f}'
    return str(value)


def facts(items):
    """Escape every data label; presentation rounding never alters stored evidence."""
    cells = ''.join(f'<div><span>{escape(str(label))}</span><strong>{escape(format_value(value))}</strong></div>' for label, value in items)
    st.markdown(f'<div class="dt-facts">{cells}</div>', unsafe_allow_html=True)


def render_tool_evidence(tool, evidence):
    if tool == 'sensitivity_analysis':
        st.caption(f"Selected pair: {evidence.get('x', '')} / {evidence.get('y', '')}")
        before, after = evidence.get('before', {}), evidence.get('after', {})
        facts([('Original Pearson r', before.get('pearson_r')), ('Filtered Pearson r', after.get('pearson_r')),
               ('Original complete pairs', before.get('n')), ('Filtered complete pairs', after.get('n')),
               ('Flagged rows', evidence.get('flagged_rows')), ('Change in r', evidence.get('delta_r'))])
        for stats in (before, after):
            if stats.get('unavailable_reason'):
                st.info(stats['unavailable_reason'])
    elif tool == 'correlation_analysis':
        st.caption(f"Selected pair: {evidence.get('x', '')} / {evidence.get('y', '')}")
        facts([('Pearson r', evidence.get('pearson_r')), ('Complete pairs', evidence.get('n'))])
        if evidence.get('unavailable_reason'):
            st.info(evidence['unavailable_reason'])
    else:
        st.json(evidence)


def render_detective_report(report):
    st.caption(f"Status: {report['status']} · Stop: {report['stop_reason']} · Model decisions: {report['model_decisions']} · Tool calls: {report['tool_calls']}")
    if report['status'] == 'partial':
        st.warning('Partial report — AI investigation did not succeed. Only completed scan results are verified.')
    elif report['status'] == 'scan_only':
        st.info('Scan-only report — no AI follow-up was performed. See the scope and limitations.')
    for finding in report['findings']:
        result_state = mascot_state(
            sensitivities=[finding['evidence']] if finding['tool_used'] == 'sensitivity_analysis' else [],
            missing_count=1 if any(signal['kind'] == 'missing_values' for signal in report['signals']) else 0,
            flagged_rows=1 if any(signal['kind'] == 'iqr_flags' for signal in report['signals']) else 0,
            complete=False)
        result_copy, result_art = st.columns([4, 1])
        with result_copy:
            card('Investigation Result', finding['observation'])
        with result_art:
            st.markdown(f'<div class="dt-result-mascot">{mascot_html(result_state)}</div>', unsafe_allow_html=True)
        st.write('Verified Evidence')
        render_tool_evidence(finding['tool_used'], finding['evidence'])
        card('Suggested Next Steps', finding['recommendation'], 'INVESTIGATE IN CONTEXT')
        st.caption(finding['limitation'])
    scope = report['scope']
    card('Scan Summary', f"Inspected {scope['scanned_rows']} of {scope['total_rows']} rows, {len(scope['scanned_columns'])} of {scope['total_columns']} columns, and {len(report['scan'].get('correlations', []))} numeric pairs. {scope['policy']}")
    for signal in report['signals']:
        evidence = signal['evidence']
        if signal['kind'] == 'missing_values':
            card(evidence['column'], f"{evidence['missing_count']} missing values ({evidence['missing_pct']}%). Review how this field was collected.", 'MISSING PIECES')
        elif signal['kind'] == 'iqr_flags':
            card(evidence['column'], f"{evidence['outlier_count']} IQR-flagged values. Flags are clues, not proven errors.", 'UNUSUAL OBSERVATIONS')
        else:
            card(f"{evidence['x']} / {evidence['y']}", f"Observed Pearson r: {format_value(evidence['pearson_r'])}; complete pairs: {evidence['n']}. Correlation does not establish causation.", 'RELATIONSHIP TO EXPLORE')
    if not report['signals']:
        st.caption('No signals listed in completed scan results. Review the scope and status before drawing conclusions.')
    selected = report.get('selected_hypothesis')
    if selected:
        card('Selected Hypothesis', selected['hypothesis'], 'MODEL-SELECTED QUESTION · NOT A CONCLUSION')
        st.caption(f"Follow-up tool: {selected['tool']} · Variables: {selected['arguments']['x']} / {selected['arguments']['y']}")
    st.write('Limitations')
    for limitation in report['limitations']:
        st.text(limitation)
    with st.expander('Advanced Details · investigation data', expanded=False):
        with st.expander('All verified scan results · full report JSON'):
            st.json(report)
        with st.expander('Detective Tool Execution Trace'):
            st.json(report['trace'])
