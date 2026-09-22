import pytest
import torch

from cardiac_motion.NetworkMotionFieldsWithInverse import NetworkMotionFieldsWithInverse
from utils.MaskImgSize import MaskImgSize


def _prepare_like_train_net(tensor: torch.Tensor) -> torch.Tensor:
    """Mirror the current train_net input reshaping exactly."""
    return tensor.squeeze().unsqueeze(0)


def _build_motion_net(times: int, masked_shape: tuple[int, int]) -> NetworkMotionFieldsWithInverse:
    return NetworkMotionFieldsWithInverse(
        dim=2,
        inshape=masked_shape,
        src_feats=times,
        out_feats=times * 2,
        ch_out_unet=8,
        n_enc_stages=2,
        int_steps=1,
        int_downsize=2,
        n_filters=8,
    ).eval()


def test_train_net_output_shape() -> None:
    times = 54
    height = 64
    width = 64
    n_fields = 2
    ch = 2

    imgs = torch.randn(times, 1, 1, height, width)
    prepared = _prepare_like_train_net(imgs)

    assert prepared.shape == (1, times, height, width)

    net = _build_motion_net(times=times, masked_shape=(height, width))
    preint_fields, motion_fields = net(prepared)

    assert preint_fields.shape == (times, ch, height, width)
    assert motion_fields.shape == (n_fields, times, ch, height, width)


def test_mask_inverse_motion_field_accepts_network_output_layout() -> None:
    times = 50
    full_height = 100
    full_width = 100
    masked_shape = (64, 64)

    mask = MaskImgSize(h_start=5, 
                       w_start=7, 
                       h_width=masked_shape[0], 
                       w_width=masked_shape[1])
    net = _build_motion_net(times=times, masked_shape=masked_shape)

    imgs = torch.randn(times, 1, 1, *masked_shape)
    prepared = _prepare_like_train_net(imgs)
    _, motion_fields = net(prepared)

    pos_motion_fields = motion_fields[0]
    restored = mask.inverse_mask_on_mf(pos_motion_fields, output_shape=(full_height, full_width))

    assert pos_motion_fields.shape == (times, 2, masked_shape[0], masked_shape[1])
    assert restored.shape == (times, 2, full_height, full_width)