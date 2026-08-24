from __future__ import annotations

import numpy as np


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute the mean DPO loss from sequence log probabilities.

    The loss is ``-log(sigmoid(beta * (policy_log_ratio - ref_log_ratio)))``.
    ``logaddexp`` evaluates the equivalent softplus expression without overflowing
    for large positive or negative preference logits.
    """
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a positive finite number")

    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )
    )
    expected_shape = arrays[0].shape
    if any(values.shape != expected_shape for values in arrays[1:]):
        raise ValueError("all log-probability arrays must have the same shape")
    if arrays[0].size == 0:
        raise ValueError("log-probability arrays must not be empty")
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("log-probability arrays must contain only finite values")

    policy_chosen, policy_rejected, ref_chosen, ref_rejected = arrays
    policy_log_ratio = policy_chosen - policy_rejected
    reference_log_ratio = ref_chosen - ref_rejected
    preference_logits = beta * (policy_log_ratio - reference_log_ratio)
    losses = np.logaddexp(0.0, -preference_logits)
    return float(np.mean(losses))


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    TODO(student): implement SFT loss + odds-ratio preference penalty.
    """
    raise NotImplementedError("TODO(student): implement ORPO loss")
