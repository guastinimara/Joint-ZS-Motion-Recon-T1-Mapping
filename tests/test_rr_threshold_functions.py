import torch

from utils.rr_threshold_functions import (
    compute_rr_mask,
    find_valid_rr_threshold,
    get_next_expansion,
    update_rr_expansion_and_compute_rr_mask,
)

# RR index of each dynamic, wrapping around from 10 back to 0
RR_IDX = torch.tensor([3, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 0, 1, 2])


class _FakeEstimator:
    """Stand-in for T1ParameterEstimator: returns NaN until the mask selects at least ``min_frames`` dynamics."""

    def __init__(self, min_frames: int):
        self.min_frames = min_frames
        self.masks = []

    def fit(self, x, mask=None, recon_params=None):
        self.masks.append(mask)
        x_ref = torch.zeros(1) if mask.sum() >= self.min_frames else torch.full((1,), float('nan'))
        return torch.zeros(3), x_ref


def test_get_next_expansion_alternates_positive_and_negative() -> None:
    expansion = [0]
    sequence = []
    for _ in range(4):
        expansion = get_next_expansion(expansion)
        sequence.append(expansion)

    assert sequence == [[0, 1], [0, 1, -1], [0, 1, -1, 2], [0, 1, -1, 2, -2]]


def test_compute_rr_mask_selects_offsets_with_wrap_around() -> None:
    mask = compute_rr_mask(RR_IDX, base_idx=0, expansion_offsets=[0, -1])

    assert mask.dtype == torch.bool
    assert torch.equal(mask, (RR_IDX == 0) | (RR_IDX == 10))


def test_update_rr_expansion_matches_compute_rr_mask() -> None:
    mask, expansion = update_rr_expansion_and_compute_rr_mask(RR_IDX, base_idx=3, expansion_offsets=[0, 1])

    assert expansion == [0, 1, -1]
    assert torch.equal(mask, compute_rr_mask(RR_IDX, 3, [0, 1, -1]))


def test_find_valid_rr_threshold_expands_until_fit_succeeds() -> None:
    # base phase 3: offsets [0] -> 3 frames, [0, 1] -> 5 frames, [0, 1, -1] -> 7 frames
    estimator = _FakeEstimator(min_frames=7)

    _, x_ref, expansion = find_valid_rr_threshold(
        RR_IDX, x_m=None, params_estim=estimator, initial_expansion=[0]
    )

    assert expansion == [0, 1, -1]
    assert len(estimator.masks) == 3
    assert not torch.isnan(x_ref).any()
    assert torch.equal(estimator.masks[-1], compute_rr_mask(RR_IDX, 3, [0, 1, -1]))


def test_find_valid_rr_threshold_returns_none_when_fit_never_succeeds() -> None:
    estimator = _FakeEstimator(min_frames=RR_IDX.numel() + 1)

    result = find_valid_rr_threshold(RR_IDX, x_m=None, params_estim=estimator, max_attempts=3)

    assert result == (None, None, None)
    assert len(estimator.masks) == 3
