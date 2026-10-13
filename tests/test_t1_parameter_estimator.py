import math

import pytest
import torch

pytest.importorskip("mrpro")

from parameters_estimation.T1ParameterEstimator import T1ParameterEstimator

# sequence timings as in examples/scripts/main.py
N_DYNAMICS = 54
SAMPLING_TIME = torch.linspace(0, 2.09375, N_DYNAMICS)
REPETITION_TIME = torch.tensor(0.0049)
TAU = torch.tensor(0.028)
NOMINAL_FA = math.radians(5)


def _build_estimator() -> T1ParameterEstimator:
    return T1ParameterEstimator(
        sampling_time=SAMPLING_TIME,
        repetition_time=REPETITION_TIME,
        nominal_flip_angle=NOMINAL_FA,
        tau=TAU,
        opt_type='lbfgs',
        device='cpu',
    )


def _simulate(estimator: T1ParameterEstimator, ny: int = 4, nx: int = 4):
    """Noise-free dynamics (t, 1, 1, ny, nx) from known parameters, using the estimator's own signal model."""
    t1 = torch.linspace(0.6, 1.8, ny * nx).reshape(ny, nx)
    flip_angle = torch.full((ny, nx), math.radians(5.5))
    m0 = torch.full((ny, nx), 2.0)
    (x,) = estimator.model_op_full_standard(m0=m0, t1=t1, flip_angle=flip_angle)
    x = x.reshape(N_DYNAMICS, 1, 1, ny, nx)
    return x, m0, flip_angle, t1


@pytest.mark.parametrize("use_mask", [False, True])
def test_fit_recovers_known_parameters(use_mask: bool) -> None:
    estimator = _build_estimator()
    x, m0, flip_angle, t1 = _simulate(estimator)

    mask = None
    if use_mask:
        mask = torch.ones(N_DYNAMICS, dtype=torch.bool)
        mask[::3] = False  # e.g. systolic frames excluded

    p, qp = estimator.fit(x, mask=mask)
    m0_fit, flip_angle_fit, t1_fit = estimator.get_params(p)

    assert p.shape == (3, 1, 1, 1, *x.shape[-2:])
    torch.testing.assert_close(t1_fit, t1, rtol=1e-2, atol=0)
    torch.testing.assert_close(flip_angle_fit, flip_angle, rtol=1e-2, atol=0)
    # m0 is fitted on images normalized by the maximum of the selected dynamics
    x_selected = x if mask is None else x[mask]
    torch.testing.assert_close(m0_fit, m0 / x_selected.abs().max(), rtol=1e-2, atol=0)
    # qp is rescaled to the input intensity and covers all dynamics, also the masked ones
    torch.testing.assert_close(qp.reshape(x.shape), x, rtol=0, atol=1e-2 * x.abs().max().item())


def test_fit_warm_start_at_constraint_bounds_stays_finite() -> None:
    """A warm start with parameters exactly on the constraint bounds must not produce NaN or inf.

    In the outer loop of main.py, each fit is initialized with the parameters of the previous fit
    (``initial_p``). These are outputs of the sigmoid constraint, so in saturated pixels they can
    be exactly equal to a bound. ``fit`` maps the initial values back with
    ``ConstraintsOp.inverse``, i.e. a logit, and logit(1) = inf. With this test we check that such
    an initialization still gives finite parameters and model images, so the next outer
    iteration does not break.

    Note: this only checks that the output is finite. Pixels initialized exactly on a bound stay
    there (the sigmoid is saturated, so the gradient is ~0).
    """
    estimator = _build_estimator()
    x, *_ = _simulate(estimator)

    p_init, _ = estimator.fit(x)
    p_init[1] = math.radians(9)  # upper flip angle bound
    p_init[2] = 6.0  # upper T1 bound

    p, qp = estimator.fit(x, initial_p=p_init)

    assert torch.isfinite(p).all()
    assert torch.isfinite(qp).all()
