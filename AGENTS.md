# Repository Guidelines

Before touching the codebase, read `README.md` (including the Timeline) so you have the current project context.

## Project Structure & Module Organization
Core source code resides in `source/`, with subpackages for `models/`, `pipelines/`, `datasets/`, and shared utilities in `utils/`. Configuration dataclasses live in `source/configs.py`; extend them when introducing new tunables so pipelines stay declarative. Tests and sample assets live in `tests/`; keep additional fixtures lightweight and under version control. Use `archive/` strictly for reference artefacts that are not imported by runtime code.

## Build, Test, and Development Commands
Activate the project environment before running anything: `conda activate ML`. Sync dependencies declared in the project docs with `conda env update --file environment.yml --name ML` (add missing libraries there rather than ad-hoc installs). Launch the decoder training experiment with `python -m source.pipelines.train_decoder`, which streams LAION data and reports metrics to Weights & Biases. Prototype dataset tweaks by running `python -m source.datasets.dataset_hf` inside an interactive session and logging overrides.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and lowercase, underscored module names (e.g., `image_decoder_transpose.py`). Prefer dataclasses for structured configs and keep function signatures annotated with types. Group imports as standard library, third-party, then local modules, and reuse the `config_logging` helper to keep log formatting consistent. Device selection must remain configuration-driven; never hardcode `cuda`, `mps`, or paths in code.

## Testing Guidelines
Tests are written for `pytest`; execute them with `pytest tests -k dinov3 --maxfail=1` before opening a PR. Name new suites `test_<feature>.py`, and co-locate small (<1 MB) media fixtures in `tests/`. Mock or gate calls touching external services (Hugging Face streaming, Modal, W&B) so default test runs remain offline. Add assertions on tensor shapes, dtypes, and numerical tolerances when modifying model logic.

## Commit & Pull Request Guidelines
Craft concise, imperative commit messages mirroring the existing history (e.g., `better streaming dataset`). A pull request should explain the problem, summarize the approach, highlight architectural decisions, and list validation steps (pytest results, manual experiment notes, screenshots). Document required config changes, new environment variables, or checkpoint updates so teammates and agents can reproduce your results. Link issues or experiment dashboards when relevant.

## Security & External Services
Store credentials for Hugging Face, W&B, or Modal in environment variables or Conda secrets; never commit them to the repo. Review timeout and SSL parameters in `HFStreamConfig` before streaming datasets, and justify any relaxed settings in code review. Confirm that outbound integrations (Modal functions, W&B runs) are disabled or stubbed in CI contexts to avoid leaking sensitive data.
