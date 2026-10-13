"""Mutual-information losses for intensity-based image registration."""

import torch
import math
import torch.nn as nn
from einops import rearrange

from operators.GradOperators import GradOperators

class MutualInformation(nn.Module):
    """Differentiable mutual-information loss for image registration.

    This module estimates intensity distributions with Gaussian kernel density
    estimation, computes marginal/joint PDFs, and derives mutual information
    (MI) as a similarity metric between two images.

    Main public methods:
        forward: Returns negative MI, suitable as a loss term.
        getMutualInformation: Returns MI values before sign inversion.
        forward_with_grad: Returns negative MI with gradient-based augmentation.
        getMutualInformation_with_grad: Returns MI augmented with gradient terms.

    Adapted from: https://github.com/xzluo97/mutual-information-registration
    """

    def __init__(self, sigma=0.1, num_bins=256, patch_size=32, overlap=0, normalize=True, device=torch.device('cpu')):
        """Configure MI estimation and precompute fixed components.

        Args:
            sigma: Gaussian kernel bandwidth for density estimation (default: 0.1).
            num_bins: Number of histogram bins in the intensity range [0, 255] (default: 256).
            patch_size: Patch size for local MI as int or 2D tuple (default: 32).
            overlap: Overlap between neighboring patches (default: 0).
            normalize: Whether to return normalized MI (default: True).
            device: Torch device for module buffers/operators (default: torch.device('cpu')).
        """
        super(MutualInformation, self).__init__()

        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10
        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(), requires_grad=False).to(device)
        self.patch_size = patch_size
        self.overlap = overlap
        self.nabla = GradOperators(2).to(device)
       
        if self.sigma == 0: 
            self.sigma = torch.mean(torch.diff(self.bins)) * 0.5
        
        self.preterm = 1/(self.sigma * math.sqrt(2*math.pi))


    def marginalPdf(self, values):
        """Gaussian KDE histogram of values (n, voxels, 1). Returns the pdf (n, bins) and kernel values."""

        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
       
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_values


    def jointPdf(self, kernel_values1, kernel_values2):
        """Joint pdf (n, bins, bins) from the kernel values of two images."""

        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    def jointPdf_grad(self, kernel_values1, kernel_values2, kernel_values_grad1, kernel_values_grad2):
        """Joint pdf combining intensity and gradient-magnitude kernel values."""

        joint_kernel_values1 = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
        
        joint_kernel_values2 = torch.matmul(kernel_values_grad1.transpose(1, 2), kernel_values_grad2)
        
        joint_kernel_values = torch.matmul(joint_kernel_values1.transpose(1, 2), joint_kernel_values2)

        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
    
        pdf = joint_kernel_values / normalization

        return pdf
    
    def get_joint_pdfs(self, input1, input2):
        """Return marginal and joint pdfs per patch for inputs (B, C, H, W) in [0, 1]."""
        input1 = input1 * 255
        input2 = input2 * 255

        B, C, H, W = input1.shape
        assert((input1.shape == input2.shape))
        if isinstance(self.patch_size, (float, int)):
            stride = self.patch_size - self.overlap
            x1_patches = input1.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)
            x2_patches = input2.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)
        
        else:
            stride = self.patch_size[0] - self.overlap, self.patch_size[1] - self.overlap
            x1_patches = input1.unfold(2, int(self.patch_size[0]), stride[0]).unfold(3, int(self.patch_size[1]), stride[1])
            x2_patches = input2.unfold(2, int(self.patch_size[0]), stride[0]).unfold(3, int(self.patch_size[1]), stride[1])
        
        x1 = rearrange(x1_patches, 'b ch num_patches_h num_patches_w patch_size_h patch_size_w -> (b num_patches_h num_patches_w) (patch_size_h patch_size_w) ch')
        x2 = rearrange(x2_patches, 'b ch num_patches_h num_patches_w patch_size_h patch_size_w -> (b num_patches_h num_patches_w) (patch_size_h patch_size_w) ch')
                
        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)
        
        return pdf_x1, pdf_x2, pdf_x1x2
    
    def getMutualInformation(self, input1, input2):
        """Compute MI on tensors shaped B, C, H, W using local or global patches."""

        input1 = input1 * 255
        input2 = input2 * 255

        B, C, H, W = input1.shape
        assert((input1.shape == input2.shape))

        if isinstance(self.patch_size, (float, int)):
            stride = self.patch_size - self.overlap
            x1_patches = input1.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)
            x2_patches = input2.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)
        
        else:
            stride = self.patch_size[0] - self.overlap, self.patch_size[1] - self.overlap
            x1_patches = input1.unfold(2, int(self.patch_size[0]), stride[0]).unfold(3, int(self.patch_size[1]), stride[1])
            x2_patches = input2.unfold(2, int(self.patch_size[0]), stride[0]).unfold(3, int(self.patch_size[1]), stride[1])
        


        x1 = rearrange(x1_patches, 'b ch num_patches_h num_patches_w patch_size_h patch_size_w -> (b num_patches_h num_patches_w) (patch_size_h patch_size_w) ch')
        x2 = rearrange(x2_patches, 'b ch num_patches_h num_patches_w patch_size_h patch_size_w -> (b num_patches_h num_patches_w) (patch_size_h patch_size_w) ch')

        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2*mutual_information/(H_x1+H_x2)

        return mutual_information


    def forward(self, input1, input2):
        """Return negative mutual information."""
        return -self.getMutualInformation(input1, input2)

    def getMutualInformation_with_grad(self, input1, input2):
        """Compute mutual information augmented with gradient-based terms."""

        input1 = input1 * 255
        input2 = input2 * 255

        input1_grad = rearrange(self.nabla(input1), 'b ch dir h w -> b ch h w dir')
        input2_grad = rearrange(self.nabla(input2), 'b ch dir h w -> b ch h w dir')
        
        input1_grad = normalize(torch.sqrt(input1_grad[..., 0] ** 2 + input1_grad[..., 1] ** 2)) * 255
        input2_grad = normalize(torch.sqrt(input2_grad[..., 0] ** 2 + input2_grad[..., 1] ** 2)) * 255
        
        B, C, H, W = input1.shape
        assert((input1.shape == input2.shape))
        assert((input1_grad.shape == input2_grad.shape))

        stride = self.patch_size - self.overlap
        x1_patches = input1.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)
        x2_patches = input2.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)

        grad1_patches = input1_grad.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)
        grad2_patches = input2_grad.unfold(2, self.patch_size, stride).unfold(3, self.patch_size, stride)


        x1 = rearrange(x1_patches, 'b ch num_patches_h num_patches_w patch_size_h patch_size_w -> (b num_patches_h num_patches_w) (patch_size_h patch_size_w) ch')
        x2 = rearrange(x2_patches, 'b ch num_patches_h num_patches_w patch_size_h patch_size_w -> (b num_patches_h num_patches_w) (patch_size_h patch_size_w) ch')
        
        grad1 = rearrange(grad1_patches, 'b ch num_patches_h num_patches_w patch_size_h patch_size_w -> (b num_patches_h num_patches_w) (patch_size_h patch_size_w) ch')
        grad2 = rearrange(grad2_patches, 'b ch num_patches_h num_patches_w patch_size_h patch_size_w -> (b num_patches_h num_patches_w) (patch_size_h patch_size_w) ch')
        
        
        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
       
        pdf_grad1, kernel_values_grad1 = self.marginalPdf(grad1)
        pdf_grad2, kernel_values_grad2 = self.marginalPdf(grad2)
        
        pdf_x1x2_grad1grad2 = self.jointPdf_grad(kernel_values1, kernel_values2,
                                                 kernel_values_grad1, kernel_values_grad2)
        
        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
        
        H_grad1 = -torch.sum(pdf_grad1*torch.log2(pdf_grad1 + self.epsilon), dim=1)
        H_grad2 = -torch.sum(pdf_grad2*torch.log2(pdf_grad2 + self.epsilon), dim=1)
        
        
        H_x1x2_grad1grad2 = -torch.sum(pdf_x1x2_grad1grad2*torch.log2(pdf_x1x2_grad1grad2 + self.epsilon), dim=(1, 2))
       
        mutual_information = H_x1 + H_x2 + H_grad1 + H_grad2 - H_x1x2_grad1grad2

        if self.normalize:
            mutual_information = 2*mutual_information/(H_x1+H_x2 + H_grad1 + H_grad2)

        return mutual_information

    def forward_with_grad(self, input1, input2):
        """Return negative gradient-augmented mutual information."""
        return -self.getMutualInformation_with_grad(input1, input2)


def normalize(tensor):
    """Min-max normalize tensor to [0, 1]."""
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val + 1e-10)
    return normalized