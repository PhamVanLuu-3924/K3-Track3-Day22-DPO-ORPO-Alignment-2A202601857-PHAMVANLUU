# Preference Alignment Experiment Report

## 1. Dataset analysis and cleaning

The final dataset contains 24 English preference pairs in the machine-learning
education domain. Two source issues were corrected:

- Line 1 contained unescaped quotation marks around `self-attention`, making the
  JSONL invalid.
- Line 18 used `recursive neural network` for RNN; this was corrected to
  `recurrent neural network`.

The loader now reports file and line information for JSON and schema failures. It
normalizes Unicode, case, and whitespace for duplicate checks, rejects identical or
near-identical responses, and offers an opt-in regex PII guard. The current dataset
loads all 24 records and produces no PII-pattern matches when the guard is enabled.

The split groups normalized prompts before deterministic shuffling with seed 42.
Using a validation ratio of 0.2 produces 19 training and 5 validation examples, with
zero prompt overlap between the two sets.

## 2. DPO implementation

DPO was selected because its pairwise objective is explicit and can be demonstrated
on CPU without the additional reference-free odds calculation required by ORPO. For
each pair, the implementation computes:

```text
policy_ratio = policy_chosen_logp - policy_rejected_logp
reference_ratio = reference_chosen_logp - reference_rejected_logp
logit = beta * (policy_ratio - reference_ratio)
loss = mean(softplus(-logit))
```

`numpy.logaddexp(0, -logit)` implements the softplus term without overflow for large
positive or negative logits. Inputs are checked for equal shape, empty arrays,
non-finite values, and invalid beta.

Configuration:

| Parameter | Value |
|---|---:|
| Method | DPO |
| Backend | `mock_cpu` |
| Beta | 0.1 |
| Steps | 20 |
| Learning rate | 1.0 |
| Batch size (documented config) | 2 |
| Maximum length (documented config) | 512 |

The mock trainer optimizes one shared scalar preference margin with the analytical
DPO gradient. It demonstrates the objective direction and writes metrics, but it does
not update a language model or produce a checkpoint.

## 3. Results

### Training

| Metric | Value |
|---|---:|
| Training examples | 19 |
| Initial DPO loss | 0.693147 |
| Final DPO loss | 0.645508 |
| Final preference margin | 0.976611 |

### Evaluation

| Metric | Value |
|---|---:|
| Validation examples | 5 |
| Pairwise accuracy | 1.0000 |
| Ties | 0 |
| Mean chosen score | 0.625015 |
| Mean rejected score | 0.446000 |

The evaluator is a deterministic, label-independent CPU heuristic. It scores the text
of each response using informative length, lexical coverage, and explanatory
connectives. It does not read whether a response came from the `chosen` or `rejected`
field.

### Qualitative review

Prompt: `What is the difference between precision and recall?`

- Chosen: “Precision measures the accuracy of positive predictions (how many
  predicted positives are actually positive), while recall measures the ability to
  find all positive instances (how many actual positives were found).”
- Rejected: “Precision measures the overall accuracy of the model, while recall
  measures the model's ability to handle imbalanced data.”
- Scores: chosen 0.716667; rejected 0.532500.
- Preference: chosen, which agrees with the dataset label.

## 4. Discussion and failure modes

The completed pipeline is deterministic, runs without GPU dependencies, prevents
prompt leakage, and is covered by linting, strict type checking, and 45 tests. The
DPO implementation remains stable on preference logits as large as +/-10,000.

Important limitations:

- The dataset is small, English-only, and restricted to introductory ML questions.
- Many rejected answers are obviously incorrect and shorter than chosen answers,
  making the evaluation task artificially easy.
- The heuristic evaluator can reward verbosity and explanatory words even when a
  response is factually wrong. Its 100% accuracy must not be interpreted as model
  quality.
- The mock trainer optimizes a scalar margin, not model parameters. There is no
  before/after text generation comparison and no deployable checkpoint.
- Near-duplicate and PII detection use heuristics and can produce false positives or
  miss obfuscated content.
- The four regression prompts were not run against a model because this CPU mock does
  not generate responses. Therefore, this experiment makes no safety-improvement
  claim.

## 5. Reproduction

```powershell
python -m pip install -e ".[dev]"
python -m ruff check src tests
python -m mypy src
python -m pytest -q
pref-lab validate data/sample_preferences.jsonl
pref-lab train --config configs/local.yaml
pref-lab evaluate --config configs/local.yaml
```

Expected artifacts are `outputs/training_metrics.json` and `outputs/metrics.json`.
