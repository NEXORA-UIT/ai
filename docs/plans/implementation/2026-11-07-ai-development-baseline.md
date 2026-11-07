# AI Development Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default pytest invocation reliable and give the AI repository one reproducible Python 3.12 baseline for local development and CI.

**Architecture:** Configure discovery in pytest itself. Run the existing offline test suite with the same locked environment and commands locally and in GitHub Actions. Tests that need live services or model weights keep their existing markers and remain separately runnable.

**Tech Stack:** Python 3.12, uv 0.13.6, pytest, Ruff, GitHub Actions.

**Spec:** [AI maintainability review, step 1](../../refactor/ai-maintainability-review.md)

## Global Constraints

- Work on the current AI feature branch and preserve the existing working tree changes.
- Preserve runtime behavior, public DTOs, CLI entrypoints, and the dependency lock.
- The CI job requires no database, provider key, Tika, or Hugging Face weights.
- Leave the changes in the working tree for review.

## Review Focus

- Pytest discovers API and generation tests with the default command.
- A fresh Hugging Face cache does not prevent the offline suite from passing.
- The lock remains consistent with the project manifest.
- CI never selects tests marked `integration`, `real_model`, or `database`.
- Existing feature branch changes do not get mixed into baseline edits.

## Task 1: Fix discovery and exercise the suite with an empty model cache

**Files:** Modify `pyproject.toml` and `tests/ingestion/workflow/test_pipeline.py`. Preserve explicit real tokenizer coverage in tests marked `integration`.

**Interfaces:** `pytest --collect-only -q` and `pytest -q -m "not integration and not real_model and not database"`.

- [x] Run default collection. Reproduced the duplicate `test_answer` module error on the original working tree.
- [x] Run the offline tests with `--import-mode=importlib`, `HF_HUB_OFFLINE=1` and a new empty `HF_HUB_CACHE`. The preview chunk test and pipeline embedding test unexpectedly loaded the BGE-M3 tokenizer.
- [x] Add `addopts = "--import-mode=importlib"` under `[tool.pytest.ini_options]`. Select the existing `--whitespace-tokenizer` for the preview test and pass `WhitespaceTokenCounter()` to the pipeline test.
- [x] Rerun default collection and the empty cache suite without passing `--import-mode` on the command line. Collected 408 tests and ran 377 successfully with 31 tests deselected by marker.

## Task 2: Align local setup and CI

**Files:** Create `.python-version` and `.github/workflows/ai-ci.yml`. Update `README.md`.

**Interfaces:** `uv sync --locked`, `uv run --locked ruff check .`, and `uv run --locked pytest -q -m "not integration and not real_model and not database"`.

- [x] Pin `3.12` in `.python-version`. Use the checkout and official setup-uv actions pinned by commit, use uv `0.13.6` to match the demo image, and run the three commands above in a read-only-token Linux workflow on push and pull request. Enable Hugging Face and Transformers offline mode for pytest.
- [x] Document the locked local install, offline checks, and separate integration markers in README. Keep the FastAPI module and application name.
- [x] Check `uv lock --check --offline`, the workflow YAML, Ruff, default pytest collection, the empty cache suite, and `git diff --check`. The GitHub-hosted workflow itself is verified after the changes are pushed.

Fresh Windows install: uv resolved all 66 packages and downloaded the locked
PyTorch CPU wheel. Windows failed while installing that wheel with resource
error 1450, using both hardlinks and copy mode. The 377 passing tests ran on
Python 3.12.16 with the dependencies already available on the workstation.
