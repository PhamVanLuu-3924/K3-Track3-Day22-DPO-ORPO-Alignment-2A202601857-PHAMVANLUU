# Data Card: ML Education Preference Pairs

## Dataset summary

- Name: ML Education Preference Pairs
- Version: cleaned lab dataset, 24 August 2026
- Size: 24 preference pairs
- Language: English
- Domain: introductory machine learning education
- Source: sample data included in this repository
- License/permission: not specified in the repository; redistribution and production
  use require confirmation from the dataset owner

## Schema

Each JSONL record contains:

| Field | Type | Description |
|---|---|---|
| `prompt` | string | Question or instruction shown to the model |
| `chosen` | string | Preferred response |
| `rejected` | string | Lower-quality response |
| `metadata` | object | Domain and labeling-rubric metadata |

All current records use `domain: education` and `rubric: accuracy`.

## Collection and labeling

The repository does not document the original authoring process. The pairs appear to
be educational examples in which the chosen answer is intended to be accurate and
explanatory while the rejected answer contains a factual error or misconception. No
claim is made that labels were independently reviewed by subject-matter experts.

## Cleaning and validation

- Corrected malformed JSON caused by unescaped quotation marks on line 1.
- Corrected `recursive neural network` to `recurrent neural network` on line 18.
- Enforced non-empty prompt and responses.
- Rejected normalized duplicate prompts and identical/near-identical response pairs.
- Verified all 24 records parse against the Pydantic schema.
- Ran the optional PII-pattern scan; no email, phone, or supported API-key pattern was
  detected.

The PII scan is regex-based and is not a guarantee that the dataset contains no
personal or sensitive information.

## Split strategy

Normalized prompts are grouped before shuffling with seed 42. A 0.2 validation ratio
produces:

| Split | Examples |
|---|---:|
| Train | 19 |
| Validation | 5 |
| Test | 0 |

There is no normalized prompt overlap between train and validation.

## Intended uses

- Teaching preference-data validation and leakage prevention.
- Unit testing a DPO objective.
- Demonstrating a reproducible CPU mock alignment workflow.

## Out-of-scope uses

- Production model training or benchmarking.
- Claims about general language-model helpfulness, factuality, or safety.
- Medical, legal, financial, or other high-stakes evaluation.

## Biases and limitations

- Small dataset with only 24 examples and no independent test set.
- English-only and concentrated on introductory ML concepts.
- Rejected responses are frequently obviously wrong and shorter than chosen answers.
- The accuracy rubric does not cover style, harmlessness, cultural context, or user
  preference diversity.
- Some labels simplify nuanced topics; for example, architectural comparisons can
  depend on task and implementation details.
- Unknown source license and undocumented annotator demographics or review process.

## Maintenance

New records should be reviewed for JSON validity, factual accuracy, duplicate prompts,
near-duplicate responses, licensing, and PII before inclusion. Changes to the dataset
count or split seed must be reflected in tests and the experiment report.
