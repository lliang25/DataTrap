"""CSV quality exploration with optional local AI explanations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from agent import analyze, explain_report, run_agent
from ai_explainer import summarize_findings
from analyzer import (missing_value_summary, iqr_outlier_report, numeric_columns,
                      numeric_correlations, row_has_iqr_outlier, validate_dataframe)

st.set_page_config(page_title="DataTrap", layout="wide")
st.title("DataTrap")
st.caption("Question the data, not just the answers.")
st.info("Flags invite investigation. They do not prove errors, selection bias, or causation.")
uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
use_sample = st.checkbox("Or load the bundled sample dataset", value=uploaded is None)
if uploaded is None and not use_sample:
    st.stop()
try:
    source = uploaded if uploaded is not None else Path(__file__).parent / "sample_data" / "demo.csv"
    df = pd.read_csv(source)
    validate_dataframe(df)
except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

st.subheader("Preview")
st.write(f"{len(df)} rows × {df.shape[1]} columns")
st.dataframe(df.head(25), use_container_width=True)
missing = missing_value_summary(df)
outliers = iqr_outlier_report(df)
corr = numeric_correlations(df)
flags = row_has_iqr_outlier(df)
st.subheader("Missing values")
st.dataframe(missing, use_container_width=True)
st.subheader("IQR outlier flags")
st.caption("Fences: Q1 − 1.5 × IQR and Q3 + 1.5 × IQR. Calculations use full precision. Flags are not a recommendation to delete rows.")
if outliers.empty:
    st.info("No finite numeric observations available for IQR analysis.")
else:
    st.dataframe(outliers.drop(columns="outlier_values"), use_container_width=True)
    st.write(f"{int(flags.sum())} rows have at least one flag across all numeric columns.")
    with st.expander("Flagged values (displayed locally; excluded from AI report)"):
        st.dataframe(outliers[["column", "outlier_values"]], use_container_width=True)

st.subheader("Correlation exploration")
if corr is None:
    st.info("Need at least two numeric columns to compute correlations.")
else:
    st.dataframe(corr, use_container_width=True)
    finite = df[numeric_columns(df)].replace([np.inf, -np.inf], np.nan).notna().astype(int)
    st.caption("Pairwise complete finite observation counts. Correlations require n ≥ 3; constant columns have undefined correlation.")
    st.dataframe(finite.T.dot(finite), use_container_width=True)

numeric = numeric_columns(df)
x = y = None
if len(numeric) >= 2:
    st.subheader("Chart and outlier sensitivity")
    x = st.selectbox("X numeric column", numeric)
    y = st.selectbox("Y numeric column", [c for c in numeric if c != x])
report = analyze(df, x, y)
if report["nonfinite_count"]:
    st.warning(f'{report["nonfinite_count"]} infinite values excluded from numeric calculations and plots; these are separate from missing values.')
if x is not None and y is not None:
    result = report["correlation_sensitivity"]
    st.caption("Filtering uses only the selected X and Y columns. Fences use each column's finite observations; correlation and plots use complete pairs. Other columns cannot remove rows from this comparison.")
    pair = df[[x, y]]
    pair_flags = row_has_iqr_outlier(pair)
    st.write(f'{result["flagged_rows"]} rows flagged in the selected columns.')
    for panel, title, frame, stats in zip(st.columns(2), ["All rows", "Without selected-column flags"],
                                          [pair, pair.loc[~pair_flags]], [result["before"], result["after"]]):
        with panel:
            st.write(title)
            st.metric("Complete pairs (n)", stats["n"])
            st.metric("Pearson r", "Unavailable" if stats["pearson_r"] is None else f'{stats["pearson_r"]:.4f}')
            if stats["unavailable_reason"]:
                st.info(stats["unavailable_reason"])
            plot = frame.replace([np.inf, -np.inf], np.nan).dropna()
            if not plot.empty:
                st.scatter_chart(plot, x=x, y=y)
    if result["delta_r"] is not None:
        st.write(f'Change in r (filtered − original): {result["delta_r"]:+.4f}')
elif numeric:
    st.subheader("Single numeric column")
    st.line_chart(df[numeric].replace([np.inf, -np.inf], np.nan))
    st.caption("X-axis is row index. Add another numeric column for correlation sensitivity.")

st.subheader("Rule-based recap")
st.write(summarize_findings(missing, outliers, corr, len(df), int(flags.sum())))
with st.expander("Auditable AI input (aggregate statistics and column names)"):
    st.json(report)
st.subheader("Optional local AI explanation")
st.caption("Uses the configured Ollama endpoint. Only the aggregate report above is sent, including column names. Keep OLLAMA_HOST local for local processing. AI text may contain mistakes; compare it with the evidence.")
# Key explanations to their exact evidence so reruns cannot display stale results.
report_key = json.dumps(report, sort_keys=True, allow_nan=False)
if st.button("Explain with local AI"):
    st.session_state.pop("ai_explanation", None)
    try:
        with st.spinner("Waiting for Ollama (up to 60 seconds)…"):
            explanation = explain_report(report)
        st.session_state["ai_explanation"] = (report_key, explanation)
    except Exception as exc:
        st.error(f"Local AI unavailable: {exc}. Start Ollama and run `ollama pull llama3.2:1b`. The numeric report remains available.")
if "ai_explanation" in st.session_state:
    key, explanation = st.session_state["ai_explanation"]
    if key == report_key:
        st.warning("AI-generated interpretation is unverified. Small models can misstate counts or reverse comparisons; use the computed tables as the source of truth.")
        st.write(explanation)

st.subheader("Ask DataTrap")
st.caption("Ask a question about this CSV. The local model chooses from four read-only analysis tools, with at most 5 decisions. Uses DATATRAP_AGENT_MODEL (default qwen3:1.7b). Your question, column names and aggregate tool results are sent to the configured Ollama endpoint.")
# Fingerprint the actual dataset, not just its aggregate report.
dataset_key = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()
                             + str(list(zip(df.columns, df.dtypes.astype(str)))).encode()).hexdigest()
with st.form("ask_datatrap"):
    question = st.text_input("Ask DataTrap", placeholder="Do outliers affect the relationship between hours_studied and exam_score?", max_chars=2000)
    ask = st.form_submit_button("Run Agent")
if ask:
    st.session_state.pop("agent_result", None)
    try:
        with st.spinner("Agent is choosing and running tools (maximum 5 decisions)…"):
            agent_result = run_agent(df, question)
        st.session_state["agent_result"] = (dataset_key, question, agent_result)
    except Exception as exc:
        st.error(f"Could not start agent: {exc}")
if "agent_result" in st.session_state:
    saved_key, saved_question, agent_result = st.session_state["agent_result"]
    if saved_key == dataset_key:
        st.write(f"Question: {saved_question}")
        st.caption(f"Model: {agent_result['model']} · Decisions: {agent_result['steps']}/5 · Stop: {agent_result['stop_reason']}")
        if agent_result['stop_reason'] != 'completed':
            st.warning("The agent stopped without a completed evidence-based answer. Inspect the stop reason and trace.")
        st.write(agent_result['answer'])
        st.caption("Model interpretation is unverified. Use computed tool results as the source of truth.")
        st.write("Tool Execution Trace")
        if not agent_result['trace']:
            st.info("No tools were executed.")
        for entry in agent_result['trace']:
            with st.expander(f"Step {entry['step']} · {entry['tool'] or 'Model response'} · {entry['status']}", expanded=True):
                st.write("Input")
                st.json(entry['input'])
                st.write("Output summary (aggregate evidence)")
                st.json(entry['output'])
        st.caption(f"Loaded Skill SHA-256: {agent_result['skill_sha256']}")

st.subheader("DataTrap Detective Mode")
st.caption("Investigate this dataset with a bounded scan and one local-model-selected follow-up. The scope, verified evidence, and tool calls are shown below.")
from detective import investigate_dataset
if "detective_result" in st.session_state and st.session_state["detective_result"][0] != dataset_key:
    del st.session_state["detective_result"]
if st.button("Investigate Dataset"):
    # Repeated clicks and ordinary widget reruns reuse the same dataset result.
    if "detective_result" not in st.session_state:
        try:
            with st.spinner("Scanning the dataset and asking Ollama to select one investigation…"):
                investigation = investigate_dataset(df)
            st.session_state["detective_result"] = (dataset_key, investigation)
        except (TypeError, ValueError) as exc:
            st.error(f"Could not investigate this dataset: {exc}")
if "detective_result" in st.session_state:
    _, investigation = st.session_state["detective_result"]
    st.caption(f"Status: {investigation['status']} · Stop: {investigation['stop_reason']} · Model decisions: {investigation['model_decisions']} · Tool calls: {investigation['tool_calls']}")
    if investigation['status'] == 'partial':
        st.warning("Partial report — AI investigation did not succeed. Only completed scan results are verified.")
    elif investigation['status'] == 'scan_only':
        st.info("Scan-only report — no AI follow-up was performed. See the scope and limitations.")
    st.write("Scan scope")
    st.json(investigation['scope'])
    st.write("Scan summary: verified signals")
    st.json(investigation['signals'])
    with st.expander("All verified scan results"):
        st.json(investigation['scan'])
    if investigation['selected_hypothesis']:
        st.write("Selected hypothesis (a question to test, not a conclusion)")
        st.json(investigation['selected_hypothesis'])
    for finding in investigation['findings']:
        st.write(finding['title'])
        st.text(finding['hypothesis'])
        st.json(finding['evidence'])
        st.text(finding['observation'])
        st.text(finding['recommendation'])
        st.text(finding['limitation'])
    st.write("Limitations")
    for limitation in investigation['limitations']:
        st.text(limitation)
    with st.expander("Detective Tool Execution Trace"):
        st.json(investigation['trace'])
    if st.button("Clear Detective report / enable retry"):
        del st.session_state["detective_result"]
        st.rerun()
