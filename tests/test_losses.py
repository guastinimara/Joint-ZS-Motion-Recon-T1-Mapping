import torch

from losses.DeformationFieldRegularizer import DeformationFieldRegularizer
from losses.MutualInformation import MutualInformation

H, W = 32, 32


def _phantom(times: int = 3) -> torch.Tensor:
    """Disc on a smooth background, (times, 1, H, W) in [0, 1]."""
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    img = ((yy - 14) ** 2 + (xx - 17) ** 2 < 64).float() * 0.8 + 0.1 * torch.sin(xx / 4)
    img = (img - img.min()) / (img.max() - img.min())
    return img.expand(times, 1, H, W).clone()


def _build_mi() -> MutualInformation:
    # same settings as in MotionEstimationWithInverse
    return MutualInformation(sigma=0, num_bins=128, patch_size=H, overlap=0, normalize=True)


def test_mutual_information_is_highest_for_aligned_images() -> None:
    torch.manual_seed(0)
    mi = _build_mi()
    x = _phantom()

    mi_same = mi.getMutualInformation(x, x)
    mi_shifted = mi.getMutualInformation(x, torch.roll(x, shifts=6, dims=-1))
    mi_noise = mi.getMutualInformation(x, torch.rand_like(x))

    assert mi_same.shape == (x.shape[0],)
    assert (mi_same > mi_shifted).all()
    assert (mi_shifted > mi_noise).all()
    # the loss is the negative mutual information
    torch.testing.assert_close(mi(x, x), -mi_same)


def test_determinant_and_divergence_for_zero_and_rigid_translation() -> None:
    regularizer = DeformationFieldRegularizer((H, W))
    zero = torch.zeros(4, 2, H, W)
    translation = torch.ones(4, 2, H, W) * torch.tensor([1.5, -2.0]).view(1, 2, 1, 1)

    for disp in (zero, translation):
        determinant = regularizer.determinant_jacobian(disp)
        assert determinant.shape == (4, H, W)
        torch.testing.assert_close(determinant, torch.ones_like(determinant))
        assert regularizer.determinant_loss(disp) == 0
        assert regularizer.divergence_loss(disp) == 0
        assert regularizer.determinant_negative_penalty(disp) == 0
