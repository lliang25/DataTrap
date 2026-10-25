# Contributing to DataTrap

## Working as a four-person team

Keep `main` as the stable demonstration branch. Create a feature branch for each task and open a pull request (PR).

```bash
git switch main
git pull origin main
git switch -c feature/my-task
# Edit and test files
git status
git add path/to/changed_file.py
git commit -m "Describe the working feature"
git push -u origin feature/my-task
```

Open a PR from your branch into `main`; have another teammate review, test, and merge. Afterward run `git switch main && git pull origin main` before starting the next feature.

## Team ownership

- Analysis: `analyzer.py`, tests.
- UI: `app.py`.
- AI: `agent.py`, `ai_explainer.py`, `skill_loader.py`, `skills/`.
- Integration: demo datasets, README, GitHub Actions, final test and demo.

Changes touching shared interfaces should be agreed upon in team chat before implementation. Prefer small commits and PRs. Avoid force-pushing shared branches. Never commit API keys, uploaded/private data, or local model files.

## Definition of done

- Functionality works with the included sample CSV.
- `python -m pytest -q` succeeds.
- UI starts with `python -m streamlit run app.py`.
- README matches behavior and warns about statistical limitations.
- A teammate can reproduce the result on another machine.
