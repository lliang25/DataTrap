# Tool-using agent verification — 2026-10-25

## Automated verification

`DATATRAP_TEST_OLLAMA=1 .venv/bin/python -m pytest -q`

**56 passed in 24.47s**, including real local model integration tests, with no skips.
The regular offline suite skips the two opt-in Ollama tests.

Coverage includes selective dispatch, observation-driven follow-up, invalid JSON,
unknown tools, invalid arguments, correcting a bad column, duplicate calls,
transport failure with retained evidence, finish/no-evidence handling, the hard
five-decision cap with no extra summary request, read-only data handling, and UI
answer/trace rendering. Earlier statistical and dashboard regression tests pass.

## Actual model behavior

No paid API or newly downloaded model was used. Both models were already installed:

- `llama3.2:1b`: preserved for the existing optional recap. In exploratory agent tests
  it selected irrelevant tools and sometimes misstated missingness, so it is not the
  default agent model.
- `qwen3:1.7b`: default tool-selection model, with thinking disabled and constrained
  JSON action output. Missingness query: missing tool → finish. Outlier-impact query:
  correlation tool → sensitivity tool → finish.

The agent integration test observes nonzero generated-token metadata from actual
Ollama responses. With the same question and dataset, it writes two different starting
policies to a temporary Skill file. Changing only that file changes the first selected
tool from `missing_value_detection` to `outlier_detection`. The repository Skill is
not modified by this counterfactual test.

The loop accepts only the four registry tools; it cannot execute model-generated
Python, shell commands, file paths, or arbitrary imports. Its JSON protocol is an
application-level tool-use interface, not dependent on native function calling.

## Streamlit verification

Restarted the server on `http://127.0.0.1:8501` to clear the previous process's cached
Python imports. Before restart, that process still held the old `agent` module and
reported an import error; the fresh process loads the new code correctly.

Real browser question:

> 学习时长 hours_studied 和成绩 exam_score 的相关性是否受到异常值影响？

Observed:

1. `correlation_analysis`, inputs `hours_studied`, `exam_score`: n=11, r=-0.6442771031252582.
2. `sensitivity_analysis`, same inputs: one flagged row, after n=10 and r approximately 1.
3. Finish: UI displayed `Decisions: 3/5 · Stop: completed`, answer, and both execution records.

The original preview, missingness and outlier tables, correlation matrix, scatter plots,
and rule-based recap remained visible and correct. No repository commit or push was made.

## Limits

The model can still choose unnecessary tools or misinterpret computed evidence. An
observed English output described a numeric increase as a decrease; a Chinese question
was understood and correctly routed but answered in English. These are model-compliance
limitations, not incorrect tool calculations. The UI identifies prose as unverified and
shows the exact tool evidence. There is no persistent memory or statistical significance
testing. The five-decision limit includes errors and finish; at the limit the UI retains
the trace and explicitly reports an incomplete answer.
