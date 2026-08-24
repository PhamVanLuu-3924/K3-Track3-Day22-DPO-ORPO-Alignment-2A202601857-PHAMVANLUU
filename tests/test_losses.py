import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_matches_expected_value() -> None:
    loss = dpo_loss(
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-0.6]),
        np.array([-1.0]),
        beta=0.1,
    )

    # Preference logit = 0.1 * ((-0.5 + 1.5) - (-0.6 + 1.0)) = 0.06.
    expected = np.logaddexp(0.0, -0.06)
    assert loss == pytest.approx(expected)


def test_dpo_loss_uses_batch_mean() -> None:
    loss = dpo_loss(
        np.array([-0.5, -1.5]),
        np.array([-1.5, -0.5]),
        np.array([-1.0, -1.0]),
        np.array([-1.0, -1.0]),
        beta=1.0,
    )

    expected = np.mean(np.logaddexp(0.0, np.array([-1.0, 1.0])))
    assert loss == pytest.approx(expected)


def test_dpo_loss_is_stable_for_extreme_logits() -> None:
    preferred_loss = dpo_loss(
        np.array([0.0]),
        np.array([-10_000.0]),
        np.array([-1.0]),
        np.array([-1.0]),
        beta=1.0,
    )
    rejected_loss = dpo_loss(
        np.array([-10_000.0]),
        np.array([0.0]),
        np.array([-1.0]),
        np.array([-1.0]),
        beta=1.0,
    )

    assert np.isfinite(preferred_loss)
    assert np.isfinite(rejected_loss)
    assert preferred_loss == pytest.approx(0.0)
    assert rejected_loss == pytest.approx(10_000.0)


@pytest.mark.parametrize("beta", [0.0, -0.1, np.inf, np.nan])
def test_dpo_loss_rejects_invalid_beta(beta: float) -> None:
    values = np.array([-1.0])
    with pytest.raises(ValueError, match="beta must be a positive finite number"):
        dpo_loss(values, values, values, values, beta=beta)


def test_dpo_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must have the same shape"):
        dpo_loss(
            np.array([-1.0, -2.0]),
            np.array([-1.0]),
            np.array([-1.0, -2.0]),
            np.array([-1.0, -2.0]),
            beta=0.1,
        )


@pytest.mark.parametrize("values", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_dpo_loss_rejects_empty_or_non_finite_arrays(values: np.ndarray) -> None:
    message = "must not be empty" if values.size == 0 else "must contain only finite values"
    with pytest.raises(ValueError, match=message):
        dpo_loss(values, values, values, values, beta=0.1)


def test_orpo_loss_todo() -> None:
    with pytest.raises(NotImplementedError):
        # Using negative logprobs
        orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
