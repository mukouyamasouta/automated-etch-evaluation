---
name: gan-method-b-trace
description: Resume, inspect, configure, and safely run the GaN etching method-B trace using Det163, Seg204, and the curated End-To-End pipeline in the automated-etch-evaluation repository. Use when continuing this experiment from a fresh session, checking current progress, preparing local smoke tests, training weights, or recording results.
---

# GaN Method-B Trace

Use the repository files as the source of truth. Do not rely on conversation memory for current experiment state.

## Start or resume work

1. Locate the repository root with `git rev-parse --show-toplevel`.
2. Read `AGENTS.md` completely and follow it.
3. Read `docs/PROJECT_STATE.md` completely.
4. Run `git status --short --branch` and `git log -3 --oneline --decorate`.
5. Preserve all existing uncommitted changes unless the user explicitly assigns them.
6. Summarize the current state, the next recorded step, and any blocker before performing a long-running or mutating action.

Read `docs/FILE_INVENTORY.md` when selecting code or explaining why a version is used. Read `docs/Dataset_ファイル構成ガイド.html` when Dataset roles or generation relationships matter. Read `docs/EXPERIMENT_LOG.md` before running or comparing experiments.

## Choose the operating mode

- **Status or explanation:** inspect only; do not modify files or start training.
- **Local smoke test:** use a separate clearly named local configuration or script, a small subset, one epoch, batch size one, and low worker count. State that its accuracy is not comparable to the thesis.
- **Full training:** verify the intended device, free storage, Dataset paths, exact commit, and output location. Start only when the user explicitly requests the full run.
- **End-To-End:** first verify that the selected Det163 and Seg204 weight files exist and match the epochs recorded in `docs/PROJECT_STATE.md`.

## Preserve reproducibility

Before an experiment, record or confirm the commit, script, Dataset, split sizes, hyperparameters, image sizing, device, and intended output. After it finishes or fails, append a factual entry to `docs/EXPERIMENT_LOG.md` and update the state document if the next step changed.

Keep Dataset, weights, generated outputs, and archive history out of Git. Never modify the `Model_backup` source. Check staged files for large or sensitive data before committing or pushing.

If repository documentation conflicts with code, stop before a costly run, report the exact conflict, and resolve which source is authoritative with the user.
