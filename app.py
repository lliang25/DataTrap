"""Streamlit UI for DataTrap hackathon prototype."""
from pathlib import Path
import pandas as pd
import streamlit as st
from agent import analyze, explain_report

st.set_page_config(page_title='DataTrap', page_icon='🔎', layout='wide')
st.title('🔎 DataTrap')
st.caption('Question the data, not just the answers. Statistical checks are calculated in Python.')
st.warning('Outlier flags are not evidence of errors. Correlation does not imply causation.')

sample = Path(__file__).resolve().parent / 'sample_data' / 'demo.csv'
uploaded = st.file_uploader('Upload a CSV', type=['csv'])
try:
    df = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(sample)
except Exception as exc:
    st.error(f'Cannot read this CSV: {exc}')
    st.stop()

st.caption('Using uploaded CSV' if uploaded else 'Using synthetic example CSV')
if df.empty or not len(df.columns):
    st.error('No data found. Upload a non-empty CSV.')
    st.stop()
st.subheader('Preview')
st.dataframe(df.head(20), use_container_width=True)

numeric = list(df.select_dtypes(include='number').columns)
x = y = None
if len(numeric) >= 2:
    left, right = st.columns(2)
    x = left.selectbox('X column', numeric, index=0)
    y = right.selectbox('Y column', numeric, index=1 if len(numeric) > 1 else 0)
report = analyze(df, x, y)

st.subheader('Data-quality scan')
a, b = st.columns(2)
a.metric('Rows', report['rows'])
b.metric('Columns', report['columns'])
st.write('**Missing values per column**')
st.dataframe(pd.DataFrame([{'column': k, 'missing': v} for k, v in report['missing'].items()]), hide_index=True)
st.write('**Potential IQR outliers (numeric columns)**')
if report['iqr_outliers']:
    st.dataframe(pd.DataFrame([{'column': k, **v} for k, v in report['iqr_outliers'].items()]), hide_index=True)
else:
    st.info('No eligible numeric columns for IQR scan.')

if x and y and x != y:
    st.subheader('Correlation sensitivity')
    st.json(report['correlation_sensitivity'])
    st.scatter_chart(df, x=x, y=y)
else:
    st.info('Choose two different numeric columns to examine correlation.')

st.subheader('Optional local AI explanation')
st.caption('Loads an Agent Skill and asks the local Ollama model to summarize only computed statistics.')
if st.button('Explain with local AI'):
    with st.spinner('Asking local open-weight model…'):
        try:
            st.write(explain_report(report))
        except RuntimeError as exc:
            st.error(str(exc))
