import pytest
import torch

pytest.importorskip("matplotlib")

from utils.MaskImgSize import MaskImgSize


def test_apply_and_get_masked_shape() -> None:
    mask = MaskImgSize(h_start=1, w_start=2, h_width=3, w_width=4)
    x = torch.arange(2 * 1 * 1 * 6 * 7).reshape(2, 1, 1, 6, 7)

    masked = mask.apply(x)

    assert mask.get_masked_shape() == (3, 4)
    assert masked.shape == (2, 1, 1, 3, 4)
    assert torch.equal(masked, x[..., 1:4, 2:6])


def test_inverse_mask_on_mf_places_values_in_window() -> None:
    mask = MaskImgSize(h_start=1, w_start=1, h_width=2, w_width=3)
    mf = torch.ones(5, 2, 2, 3)

    restored = mask.inverse_mask_on_mf(mf, output_shape=(6, 7))

    assert restored.shape == (5, 2, 6, 7)
    assert torch.equal(restored[..., 1:3, 1:4], mf)
    assert torch.count_nonzero(restored[..., :1, :]) == 0


def test_apply_raises_when_mask_not_set() -> None:
    mask = MaskImgSize()
    x = torch.zeros(1, 1, 1, 4, 4)

    with pytest.raises(ValueError, match="Mask values are not set"):
        _ = mask.apply(x)
