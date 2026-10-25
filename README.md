# DataTrap — Question the Data, Not Just the Answers

> Most AI tools help you analyze data. DataTrap helps you question the analysis.

**DataTrap** is a hackathon prototype for inspecting CSV datasets, flagging potential quality issues, and using a **locally run open-weight AI model** to explain real, computed statistical evidence. Built for Hacktoberfest 2026.

## What works in this starter

- Upload a CSV or use the included synthetic sample dataset.
- Preview rows and count missing values per column.
- Flag potential numeric outliers using the 1.5×IQR rule.
- Explore correlations between pairs of numeric columns and compare them with flagged rows excluded.
- View a scatter plot and an auditable, rule-based findings report.
- Optionally ask a locally hosted open-weight model (**Qwen3 1.7B** by default, or Llama 3.2 1B) to explain a structured analysis summary, guided by an Agent Skill in `skills/data-quality-investigator/SKILL.md`.

**Not yet implemented:** autonomous model-selected tools, multi-step agent planning, durable memory, advanced charting, or statistical significance tests. `agent.py` is a *basic deterministic harness*: it calls Python analysis tools before passing their outputs to the model. The team can expand it to add model-driven tool selection and memory for the hackathon.

## Hackathon requirement mapping

| Requirement | This starter |
| --- | --- |
| Open-weight model | Ollama + `qwen3:1.7b` (or `llama3.2:1b`; optional at runtime; **run AI demo to satisfy requirement**) |
| Agent Skill | `skills/data-quality-investigator/SKILL.md`, loaded at runtime |
| Harness | `agent.py` orchestrates Python statistics, skill loading, and LLM explanations; **expand for meaningful agentic behavior** |
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
   ollama pull qwen3:1.7b
   ```

   Open/keep Ollama running, then click **Explain with local AI** in the app. The app never sends uploaded CSV rows to a remote AI service by design; it passes a compact statistical report to the configured Ollama endpoint. Keep `OLLAMA_HOST` pointing at a trusted local endpoint if data privacy matters.

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
agent.py                 Scan orchestration + safe LLM context
ai_explainer.py          Ollama model call
skill_loader.py          Agent Skill file loading
skills/data-quality-investigator/SKILL.md
sample_data/student_scores.csv  Synthetic data for demo
test/test_analyzer.py    Initial unit tests
.github/pull_request_template.md
.github/workflows/tests.yml
CONTRIBUTING.md
LICENSE
.env.example
.gitignore
```

## Architecture

`CSV → pandas analyzer → structured findings → agent harness → Agent Skill + local Ollama model → explanation`.

Python performs all arithmetic. The model explains only computed findings. AI can be unavailable and the numeric scan still works. Outliers are *flags*, not proven errors; correlations are not causal evidence. Missingness alone does not establish selection bias.

## Team roles (suggested)

- **Analysis:** `analyzer.py` and tests.
- **Frontend:** `app.py` and charts.
- **AI:** `agent.py`, `ai_explainer.py`, `skill_loader.py`, `skills/`.
- **Integration/demo:** README, CI, fixture CSV, end-to-end testing and pitch.

See [CONTRIBUTING.md](CONTRIBUTING.md) for Git branches, PRs, and team commands.

## Run tests

```bash
.venv/bin/python -m pytest -q
```

## Suggested 2-minute demo

1. Load `sample_data/student_scores.csv`.
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
