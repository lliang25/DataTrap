"""DataTrap Streamlit UI — upload a CSV, preview it, and inspect data-quality flags."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from ai_explainer import summarize_findings
from analyzer import (
    drop_flagged_outlier_rows,
    iqr_outlier_report,
    missing_value_summary,
    numeric_columns,
    numeric_correlations,
    row_has_iqr_outlier,
)
st.set_page_config(page_title="DataTrap", layout="wide")
st.title("DataTrap")
st.caption("Question the data, not just the answers.")
st.info(
    "A CSV can hide missing values and extreme observations behind a clean chart. "
    "DataTrap flags those issues for investigation. It does not prove selection bias, "
    "causation, or that an outlier is a mistake."
)
uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
use_sample = st.checkbox("Or load the bundled sample dataset", value=uploaded is None)
df: pd.DataFrame | None = None
source_label = ""
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_label = uploaded.name
elif use_sample:
    df = pd.read_csv("sample_data/student_scores.csv")
    source_label = "sample_data/student_scores.csv"
if df is None:
    st.stop()
st.subheader("Preview")
st.write(f"**Source:** {source_label} · **{len(df)} rows × {df.shape[1]} columns**")
st.dataframe(df.head(25), use_container_width=True)
try:
    missing = missing_value_summary(df)
    outliers = iqr_outlier_report(df)
    corr = numeric_correlations(df)
    outlier_rows = row_has_iqr_outlier(df)
    sensitivity_df = drop_flagged_outlier_rows(df)
except (TypeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
st.subheader("Missing values")
st.dataframe(missing, use_container_width=True)
if missing["missing_count"].sum() == 0:
    st.success("No missing values counted.")
else:
    st.warning(
        "Missing values can quietly change totals, averages, and correlations. "
        "Ask how those gaps were produced before drawing a conclusion."
    )
st.subheader("IQR outlier flags")
st.write(
    "A value is flagged when it falls below **Q1 − 1.5×IQR** or above **Q3 + 1.5×IQR**. "
    "Thresholds are shown so you can see the evidence, not just the warning."
)
if outliers.empty:
    st.info("No numeric columns were available for an IQR scan.")
else:
    display_outliers = outliers.drop(columns=["outlier_values"])
    st.dataframe(display_outliers, use_container_width=True)
    flagged = outliers[outliers["outlier_count"] > 0]
    if flagged.empty:
        st.success("No IQR-flagged values in the numeric columns.")
    else:
        st.warning(
            f"{int(outlier_rows.sum())} row(s) contain at least one IQR-flagged value. "
            "Use the comparison below as a sensitivity check — not a recommendation to delete data."
        )
        with st.expander("Flagged values by column"):
            for _, row in flagged.iterrows():
                st.write(f"**{row['column']}:** {row['outlier_values']}")
st.subheader("Correlation exploration")
st.caption("Correlation ≠ causation. A strong number can still be an artifact of missing data or a few extreme rows.")
if corr is None:
    st.info("Need at least two numeric columns to compute correlations.")
else:
    st.dataframe(corr.round(3), use_container_width=True)
st.subheader("Chart and outlier sensitivity")
numeric = numeric_columns(df)
if not numeric:
    st.info("Add a numeric column to plot a chart.")
else:
    left, right = st.columns(2)
    with left:
        y_col = st.selectbox("Numeric column to plot", numeric)
    with right:
        x_candidates = [c for c in df.columns if c != y_col]
        x_col = st.selectbox("Column on the x-axis", x_candidates, index=0)
    view = st.radio(
        "Which rows should the chart use?",
        ["All uploaded rows", "Rows without IQR-flagged values"],
        horizontal=True,
    )
    chart_df = df if view == "All uploaded rows" else sensitivity_df
    if chart_df.empty:
        st.warning("No rows remain after removing IQR-flagged observations.")
    else:
        plot_df = chart_df[[x_col, y_col]].dropna()
        st.write(f"Plotting **{len(plot_df)}** rows ({view.lower()}).")
        st.bar_chart(plot_df, x=x_col, y=y_col)
    compare_left, compare_right = st.columns(2)
    with compare_left:
        st.metric("Rows in full dataset", len(df))
        if y_col in df.columns:
            st.metric(f"Mean {y_col} (all rows)", f"{pd.to_numeric(df[y_col], errors='coerce').mean():.2f}")
    with compare_right:
        st.metric("Rows without IQR flags", len(sensitivity_df))
        if y_col in sensitivity_df.columns and not sensitivity_df.empty:
            st.metric(
                f"Mean {y_col} (without flags)",
                f"{pd.to_numeric(sensitivity_df[y_col], errors='coerce').mean():.2f}",
            )
st.subheader("Plain-English recap")
st.write(
    summarize_findings(
        missing=missing,
        outliers=outliers,
        correlations=corr,
        n_rows=len(df),
        n_outlier_rows=int(outlier_rows.sum()),
    )
)
st.caption(
    "This recap only restates calculated results. It is not independently verified statistical evidence."
)