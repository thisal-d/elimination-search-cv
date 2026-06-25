# Contributing to EliminationSearchCV

Thank you for taking the time to contribute! 🎉  
This is an early-stage project and contributions of **any size** are welcome — from a typo fix to a new feature.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Reporting Bugs](#reporting-bugs)
3. [Suggesting Features](#suggesting-features)
4. [Development Setup](#development-setup)
5. [Branch & Commit Conventions](#branch--commit-conventions)
6. [Pull Request Checklist](#pull-request-checklist)
7. [Running Benchmarks](#running-benchmarks)
8. [Code Style](#code-style)

---

## Getting Started

1. **Star the repo** ⭐ — it helps the project grow and signals to others that it's worth checking out.
2. **Read the [CHANGELOG](./CHANGELOG.md)** to understand what has been done and what is planned.
3. **Check open [Issues](https://github.com/thisal-d/elimination-search-cv/issues)** before starting new work — your idea may already be tracked.
4. **Open an issue** before starting significant work (new features, refactors) to align on design and avoid duplicated effort.

---

## Reporting Bugs

Use the **Bug Report** issue template on GitHub. Please include:

- A minimal, reproducible example (MRE)
- Python version, OS, and scikit-learn version (`pip show scikit-learn`)
- Expected vs actual behavior
- Full traceback if applicable

---

## Suggesting Features

Use the **Feature Request** issue template. Describe:

- What problem you are trying to solve
- How you expect the API to look (pseudocode is fine)
- Whether you are willing to implement it yourself

---

## Development Setup

```bash
# 1. Fork and clone your fork
git clone https://github.com/<your-username>/elimination-search-cv.git
cd elimination-search-cv

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install in editable mode
pip install -e .

# 4. Install benchmark dependencies (optional)
pip install scikit-learn numpy pandas
```

---

## Branch & Commit Conventions

| Type | Branch prefix | Example |
|---|---|---|
| New feature | `feat/` | `feat/n-jobs-parallel` |
| Bug fix | `fix/` | `fix/single-param-grid-crash` |
| Documentation | `docs/` | `docs/update-readme-benchmarks` |
| Refactor | `refactor/` | `refactor/scoring-utils` |
| Benchmark | `bench/` | `bench/add-svm-model` |

Commit messages should be short and imperative, e.g.:
- `feat: add best_score_ and best_estimator_ attributes`
- `fix: handle cv=1 edge case in create_cv_data_sets`
- `docs: update benchmark table with v0.0.1 results`

---

## Pull Request Checklist

Before opening a PR, make sure:

- [ ] Your branch is up to date with `main`
- [ ] The code is clean and follows existing style (see [Code Style](#code-style))
- [ ] Docstrings are updated for any changed/added functions or classes
- [ ] If you added a public attribute or parameter, the README `Result Attributes` / `Constructor Parameters` table is updated
- [ ] If this is a bug fix, a note is included in `CHANGELOG.md`
- [ ] If this is a significant new feature, a note is included in `CHANGELOG.md`

---

## Running Benchmarks

```bash
# Quick configurable benchmark (recommended for development)
python benchmarks/benchmark_fast.py

# Full benchmark (5 models × 3 datasets × Light+Full grids)
python benchmarks/benchmark.py
```

Benchmark results are saved to `benchmarks/marks/`. If your change affects search speed or score quality, please include benchmark output in your PR description.

---

## Code Style

- **Python 3.8+ compatible** — avoid f-strings with `=` (3.8 syntax), walrus operator, and other 3.9+ features unless adding a version guard.
- **Type hints** on all public functions and `__init__` parameters.
- **Docstrings** in Google-style for all public classes and functions.
- **Comments** — prefer block comments above non-obvious logic over inline comments.
- **No external dependencies** beyond `scikit-learn` and `numpy` in the core `src/` package.

---

*By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).*
