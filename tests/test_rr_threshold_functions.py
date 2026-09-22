import torch

from utils.rr_threshold_functions import (
    compute_rr_mask,
    find_valid_rr_threshold,
    get_next_expansion,
    update_rr_expansion_and_compute_rr_mask,
)


class _DummyEstimator:
    def __init__(self) -> None:
        self.calls = 0

    def fit(self, x_m: torch.Tensor, mask: torch.Tensor):
        self.calls += 1
        if self.calls == 1:
            return torch.tensor([0.0]), torch.tensor([float("nan")])
        return torch.tensor([1.0]), torch.tensor([2.0])


def test_compute_rr_mask_wraparound() -> None:
    rr_idx = torch.tensor([8, 9, 0, 1, 2])
    mask = compute_rr_mask(rr_idx=rr_idx, base_idx=9, expansion_offsets=[0, 1])

    expected = torch.tensor([False, True, True, False, False])
    assert torch.equal(mask, expected)


def test_get_next_expansion_pattern() -> None:
    assert get_next_expansion([0]) == [0, 1]
    assert get_next_expansion([0, 1]) == [0, 1, -1]
    assert get_next_expansion([0, 1, -1]) == [0, 1, -1, 2]
    assert get_next_expansion([0, 1, -1, 2]) == [0, 1, -1, 2, -2]


def test_update_rr_expansion_and_compute_rr_mask_updates_offsets() -> None:
    rr_idx = torch.tensor([0, 1, 2, 3, 4])
    base_idx = 0
    expansion_offsets = [0, 1]

    mask, updated_offsets = update_rr_expansion_and_compute_rr_mask(
        rr_idx=rr_idx,
        base_idx=base_idx,
        expansion_offsets=expansion_offsets,
    )

    assert updated_offsets == [0, 1, -1]
    expected_mask = torch.tensor([True, True, False, False, True])
    assert torch.equal(mask, expected_mask)


def test_find_valid_rr_threshold_expands_until_non_nan_fit() -> None:
    rr_idx = torch.tensor([1, 2, 3, 4])
    x_m = torch.randn(4, 1)
    estimator = _DummyEstimator()

    p_0_ref, x_0_ref, expansion_offsets = find_valid_rr_threshold(
        rr_idx=rr_idx,
        x_m=x_m,
        params_estim=estimator,
        log_filepath=None,
        max_attempts=3,
    )

    assert estimator.calls == 2
    assert expansion_offsets == [0, 1, 2, -2]
    assert torch.equal(p_0_ref, torch.tensor([1.0]))
    assert torch.equal(x_0_ref, torch.tensor([2.0]))
