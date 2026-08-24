# Preference Alignment Lab: DPO CPU Implementation

Production-style preference-alignment lab using `prompt`, `chosen`, and `rejected`
pairs. This implementation selects the DPO objective and provides a reproducible
CPU-only mock training and evaluation workflow. It demonstrates the alignment
pipeline without claiming to fine-tune a language model.

## Learning goals

- Validate and load preference pairs (`prompt`, `chosen`, `rejected`).
- Implement or wrap DPO/ORPO training logic.
- Build evaluation metrics for pairwise preference and regression prompts.
- Practice production habits: typed code, configs, tests, Makefile, CI, docs.

## Quickstart (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

On Linux or macOS, activate with `source .venv/bin/activate` and use single quotes
around `.[dev]` if required by the shell.

## Run the lab

```powershell
pref-lab validate data/sample_preferences.jsonl
pref-lab train --config configs/local.yaml
pref-lab evaluate --config configs/local.yaml
Get-Content outputs/training_metrics.json
Get-Content outputs/metrics.json
```

The checked configuration produces a deterministic 19/5 train/validation split.
Generated metrics are written under `outputs/`, which is intentionally ignored by Git.

## Current results

| Result | Value |
|---|---:|
| Valid preference pairs | 24 |
| Train / validation examples | 19 / 5 |
| Mock DPO loss | 0.6931 -> 0.6455 |
| Validation pairwise accuracy | 1.0000 |
| Tests | 45 passed |

Pairwise accuracy is produced by a deterministic heuristic that rewards informative
length, lexical coverage, and explanatory connectives. It is label-independent but
can favor longer answers. It is not equivalent to human evaluation or model judging.

## Quality checks

```powershell
python -m ruff check src tests
python -m mypy src
python -m pytest -q
```

These are the same checks executed by GitHub Actions.

## Optional real-model dependencies

The `train` extra is reserved for extending this lab with a TRL-backed trainer:

```powershell
python -m pip install -e ".[dev,train]"
```

This installs PyTorch, Transformers, Datasets, TRL, and PEFT and is not required for
the CPU workflow documented above.

## Lab scope

- Robust JSONL validation with line-numbered errors, duplicate checks, and optional
  PII patterns.
- Deterministic splitting by normalized prompt to prevent leakage.
- Numerically stable DPO loss implemented with `numpy.logaddexp`.
- CPU mock optimization that writes explicit training metrics but no checkpoint.
- Deterministic pairwise evaluation with explicit tie handling.
- Strict typing, linting, tests, configuration, and documentation.

## Milestones

| Time | Goal | Command |
|---|---|---|
| 0-30 min | Setup and inspect sample data | `pytest -q` |
| 30-50 min | Implement dataset validation/collator | `pytest tests/test_data.py` |
| 50-70 min | (Optional) Generate synthetic data | `python scripts/generate_data.py` |
| 70-100 min | Implement DPO objective | `pytest tests/test_losses.py` |
| 100-115 min | Train mock and evaluate | `pref-lab evaluate --config configs/local.yaml` |
| 115-120 min | Review metrics and report | `Get-Content outputs/metrics.json` |

## Repository layout

```text
src/preference_lab/     Python package
data/                   Small sample preference dataset
configs/                YAML configs for local experiments
docs/                   Lab guide, completed report, and data card
scripts/                Utility entrypoints
tests/                  Unit tests for student work
```

## Production checklist

- [x] Dataset schema validated.
- [x] Train/eval split by prompt, not by row.
- [x] Config committed; generated artifacts ignored.
- [x] Training and evaluation metrics saved as JSON.
- [x] Data card and experiment report completed.
- [ ] Model safety regression prompts run before/after training. The CPU mock trainer
  does not generate responses, so no model-safety claim is made.

See [the experiment report](docs/REPORT.md) and [the data card](docs/DATA_CARD.md)
for assumptions, metrics, and limitations.
