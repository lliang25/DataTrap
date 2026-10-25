# DataTrap — Question the Data, Not Just the Answers

> Most AI tools help you analyze data. DataTrap helps you question the analysis.

**DataTrap** is a hackathon prototype for inspecting CSV datasets, flagging potential quality issues, and using a **locally run open-weight AI model** to explain real, computed statistical evidence. A demonstration prototype; verify any event submission rules separately.

## What works in this starter

- Ask natural-language questions in **Ask DataTrap**: a local model chooses safe analysis tools and shows a Tool Execution Trace.
- Upload a CSV or use the included synthetic sample dataset.
- Preview rows and count missing values per column.
- Flag potential numeric outliers using the 1.5×IQR rule.
- Explore correlations between pairs of numeric columns and compare them with flagged rows excluded.
- View a scatter plot and an auditable, rule-based findings report.
- Optionally ask an **Ollama-hosted Llama 3.2 1B** model to explain a structured analysis summary, guided by an Agent Skill in `skills/data-quality-investigator/SKILL.md`.

**Ask DataTrap now supports model-selected tools and a bounded multi-step agent loop.** The existing deterministic dashboard and optional Llama recap are preserved. Durable memory, advanced charting, and statistical significance tests are not implemented.

## Proposed hackathon requirement mapping

This is a project checklist, not confirmation against official event rules.

| Requirement | This starter |
| --- | --- |
| Open-weight model | Ollama + `qwen3:1.7b` for the agent; `llama3.2:1b` for the optional recap. Run an actual model demo. |
| Agent Skill | `skills/data-quality-investigator/SKILL.md`, loaded at runtime |
| Harness | `agent.py` runs a model-directed loop with four allowlisted tools, observations, a trace, and a five-decision cap. |
| Public, open-source submission | Publish this repository publicly with `LICENSE`, README, and a live demo |

Using Cursor to *write code* doesn't itself satisfy the open-weight requirement; the **app must run** an open-weight model.

## Quick start (Windows / Cursor)

1. Install Python 3.10+, Git, and Cursor. Open the repository folder in Cursor.
2. In Cursor's terminal:

   ```powershell
   py -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   .venv\Scripts\python.exe -m streamlit run app.py
   ```

3. Open the Streamlit local URL, typically http://localhost:8501.
4. For AI explanations, install [Ollama](https://ollama.com/download) and run:

   ```powershell
   ollama pull llama3.2:1b
   ollama pull qwen3:1.7b
   ```

   Open/keep Ollama running. Use **Run Agent** in Ask DataTrap, or **Explain with local AI** for the original recap. The app never sends uploaded CSV rows to a remote AI service by design; it passes a compact statistical report to the configured Ollama endpoint. Keep `OLLAMA_HOST` pointing at a trusted local endpoint if data privacy matters.

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Repo map

```text
app.py                   Streamlit interface
analyzer.py              Deterministic stats and checks
agent.py                 Model-directed agent loop + original dashboard helpers
agent_tools.py           Four read-only tools with argument validation
ai_explainer.py          Ollama model call
skill_loader.py          Agent Skill file loading
skills/data-quality-investigator/SKILL.md
sample_data/demo.csv     Synthetic data for demo
sample_data/student_scores.csv  Same synthetic fixture, for upload verification
tests/                   Statistics, AI boundary, and Streamlit regression tests
.github/pull_request_template.md
.github/workflows/tests.yml
CONTRIBUTING.md
LICENSE
.env.example
.gitignore
```

## Architecture

The deterministic dashboard remains `CSV → pandas analyzer → tables/charts/recap`.

The independent question-answering path is:

```text
CSV + question → dataset schema + runtime-loaded Skill → local Ollama model
                    ↑                                      ↓
            tool observation ← safe tool registry ← validated JSON action
                    └──────── next model decision ─────────┘
                                      ↓
                         final answer + execution trace
```

The agent starts with the question and column names/types, **not a precomputed full scan**.
It chooses the next tool after each observation. The dashboard still computes its usual
statistics on page reruns; those results are not used to predetermine the agent's tools.

Python performs all arithmetic. The model is instructed to explain only computed findings, but its output is not automatically fact-checked. AI can be unavailable and the numeric scan still works. Outliers are *flags*, not proven errors; correlations are not causal evidence. Missingness alone does not establish selection bias.

## Analysis behavior and configuration

- Correlation comparisons filter IQR flags in the **two selected columns only**.
- Both charts show complete finite pairs, with Pearson r and effective sample size n.
- Correlations need at least 3 complete observations and two nonconstant columns; otherwise the UI explains why the value is unavailable.
- IQR fences retain full precision. Infinite values are excluded from numeric calculations and counted separately from missing values.
- A single numeric column is plotted against row index. Text-only data still supports missingness inspection.
- The synthetic demo includes missing values and one influential observation: filtering it changes the study-hours/score relationship substantially. It is not real student data.
- AI receives the displayed aggregate report, including column names, counts and thresholds; raw rows and flagged values are excluded. Column names can themselves contain sensitive information.
- The 1B model can misstate counts or reverse before/after comparisons, even with explicit evidence labels. Always check its prose against the displayed tables.
- AI output is optional and may be incorrect. The deterministic report remains available when Ollama is offline or a request fails.
- AI requests use a 60-second HTTP timeout and reject reports larger than 24,000 characters; use fewer columns for the small-model demo.

`.env.example` documents supported environment variables; `.env` is **not automatically loaded**. Defaults work without configuration. To override them before starting Streamlit:

```bash
export OLLAMA_HOST=http://localhost:11434
export DATATRAP_MODEL=llama3.2:1b
export DATATRAP_AGENT_MODEL=qwen3:1.7b
```

PowerShell equivalents:

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
$env:DATATRAP_MODEL = "llama3.2:1b"
$env:DATATRAP_AGENT_MODEL = "qwen3:1.7b"
```

A hosted Streamlit app needs access to its own Ollama server; its `localhost` is the server, not the visitor's laptop. A working local demo does not imply a hosted AI demo is configured.

## Tool-using agent workflow

Ask questions such as:

- “Which columns have missing values?”
- “Do outliers affect the correlation between hours_studied and exam_score?”
- “学习时长和成绩的相关性是否受到异常值影响？”

The default agent model is `qwen3:1.7b`, already available on the development machine.
The smaller Llama 1B model remains available for the original recap; local testing
found its tool selection less reliable. Set `DATATRAP_AGENT_MODEL` to change the agent
model. Qwen3 runs with thinking disabled so its bounded response contains the action.
No paid API is used and the app never automatically downloads models.

| Tool | Allowed inputs | Returned evidence |
| --- | --- | --- |
| `missing_value_detection` | Optional `columns` | Missing counts and percentages |
| `outlier_detection` | Optional numeric `columns` | IQR fences, counts, distinct flagged rows; no raw outlier values |
| `correlation_analysis` | Numeric `x`, `y` | Pearson r, complete finite n, unavailable reason |
| `sensitivity_analysis` | Numeric `x`, `y` | Before/after r and n, flagged rows, delta; selected-pair filtering |

The model returns one schema-constrained JSON action per turn. This is a tool-use
protocol implemented by the application, not a claim of native Ollama function-call
support for every model. Python validates the action and dispatches it only through
`agent_tools.TOOLS`. There is no model-generated Python execution, shell access,
file-path argument, arbitrary import, or dataset mutation.

**Budget and errors:** a maximum of five model decisions per question, including
failed actions and the final answer. Each decision executes at most one tool. Invalid
JSON/arguments and tool errors are returned as observations for correction. Repeated
successful calls are blocked. Transport/model failures stop immediately. At the budget
limit, collected results remain in the trace and the app explicitly reports that no
final model answer was produced; there is no sixth summarization call. A finish with
no successful tool is labeled `no_evidence` rather than presented as verified analysis.
Each HTTP request has a 60-second timeout; a multi-step run can take longer overall.

**Skill influence:** `skills/data-quality-investigator/SKILL.md` is read for every
question and placed in the system prompt before the first tool selection. Its policy
maps missingness questions to missing detection, relationship questions to correlation,
and outlier-impact follow-ups to sensitivity analysis. It also supplies scientific
interpretation constraints. The model still makes each tool choice; Python does not
route by question keywords. Tests verify prompt inclusion and, with real Ollama,
change only the Skill's starting policy to check that the first tool changes.

**Trace:** displays each attempted tool, validated input, aggregate output or error,
step number, stop reason, model, and loaded Skill hash. It does not expose hidden model
reasoning. Results persist across UI reruns and are hidden when the dataset changes.

**Limits:** questions are at most 2,000 characters; agent mode accepts up to 100 uniquely
named columns; one missing/outlier call can inspect up to 20 columns. Schema and output
size limits reject oversized requests. The model may still choose unnecessary tools or
misinterpret results; the trace is the evidence. Tool selection and prompt compliance
can vary between model versions. There is no persistent conversational memory.

## Team roles (suggested)

- **Analysis:** `analyzer.py` and tests.
- **Frontend:** `app.py` and charts.
- **AI:** `agent.py`, `ai_explainer.py`, `skill_loader.py`, `skills/`.
- **Integration/demo:** README, CI, fixture CSV, end-to-end testing and pitch.

See [CONTRIBUTING.md](CONTRIBUTING.md) for Git branches, PRs, and team commands.

## Run tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

On macOS/Linux use `.venv/bin/python -m pytest -q`. CI runs the suite on Python 3.10 and 3.12. AI tests use a mock client; to verify the actual model, start Ollama, pull the model, and click the AI button locally.

### Optional real-model integration check

With Ollama running and both `llama3.2:1b` and `qwen3:1.7b` already installed:

```bash
DATATRAP_TEST_OLLAMA=1 .venv/bin/python -m pytest -q -s tests/test_ollama_integration.py
```

This sends only bundled synthetic data summaries to localhost. It checks real generation metadata, different outputs for different reports, the exact Skill text in the system message, model-selected tool sequences, and a Skill-policy A/B test. The agent tests use the installed Qwen model; the original recap test uses Llama. Regular tests skip this check so CI does not require a model. Passing it confirms integration, not factual correctness of model prose.

`sample_data/student_scores.csv` is a copy of the synthetic demo, not a real student dataset.

## Suggested 2-minute demo

1. Upload `sample_data/demo.csv`.
2. Show missingness, outliers, correlation table, and the plot.
3. Choose two numeric columns to compare sensitivity to flagged outliers.
4. Turn on Ollama and generate an explanation grounded in calculated values.
5. Show the skill file and `agent.py` source to demonstrate where AI behavior comes from.

## Security and scientific caveats

- Never commit real private datasets, keys, or `.env`.
- Uploaded data are processed in the Streamlit app's Python process; only aggregate stats are sent to Ollama.
- A local default endpoint does not guarantee privacy if `OLLAMA_HOST` is changed to a remote server.
- The IQR method is a heuristic and may overflag skewed distributions.
- Correlations use pairwise complete numeric observations; filtering may change sample sizes.
- Outliers should not automatically be deleted. Interpret sensitivity results in context.

## License

Original starter source is provided under the MIT License. **Model weights have their own license**; review Ollama's Llama model listing and Meta's terms separately before distributing model weights.
