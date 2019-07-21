import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PSNR(nn.Module):
    """The PSNR score.
    Args:
        size_average (bool): Whether to average the PSNR score all over the samples in the batch (default: True).
        max_value (int): The maximum pixel value of the image (default: 255).
    """
    def __init__(self, size_average=True, max_value=255):
        super().__init__()
        self.size_average = size_average
        self.max_value = max_value

    def forward(self, output, target):
        """
        Args:
            output (torch.tensor) (N, C, *): The model output.
            target (torch.tensor) (N, C, *): The data target.

        Returns:
            score (torch.tensor) (0) or (N): The PSNR score.
        """
        reduced_dims = list(range(1, output.dim()))
        mse = F.mse_loss(output, target, reduction='none').mean(reduced_dims)
        psnr = 10 * torch.log10(self.max_value ** 2 / mse)

        if self.size_average:
            return psnr.mean()
        else:
            return psnr


class SSIM(nn.Module):
    """The SSIM score.

    Ref: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
         https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8

    Args:
        dim (int): The dimention of the image (default: 2).
        channels (int): The number of the image channels (default: 1).
        size_average (bool): Whether to average the SSIM score all over the samples in the batch (default: True).
        value_range (int): The difference between the maximum and the minimum pixel value of the image. The common values are 1, 2 and 255 (default: 255).
    """
    def __init__(self, dim=2, channels=1, size_average=True, value_range=255):
        super().__init__()
        self.dim = dim
        self.channels = channels
        self.size_average = size_average
        self.value_range = value_range
        self.c1 = (0.01 * value_range) ** 2
        self.c2 = (0.03 * value_range) ** 2

        if dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise ValueError(f"Only dim=2, 3 are supported. Received dim={dim}.")

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel_size, sigma = 11, 1.5
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        kernel = 1
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = size // 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight.
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, output, target):
        """
        Args:
            output (torch.tensor) (N, C, *): The model output.
            target (torch.tensor) (N, C, *): The data target.

        Returns:
            score (torch.tensor) (0) or (N): The SSIM score.
        """
        # The average of the output and the target.
        mu1 = self.conv(output, weight=self.weight, groups=self.groups)
        mu2 = self.conv(target, weight=self.weight, groups=self.groups)

        # The variance of the output and the target.
        sigma1_sq = self.conv(output * output, weight=self.weight, groups=self.groups) - mu1.pow(2)
        sigma2_sq = self.conv(target * target, weight=self.weight, groups=self.groups) - mu2.pow(2)

        # The covariance of the output and the target.
        sigma12 = self.conv(output * target, weight=self.weight, groups=self.groups) - mu1 * mu2

        # The SSIM score map.
        ssim_map = ((2 * mu1 * mu2 + self.c1) * (2.0 * sigma12 + self.c2)) / ((mu1.pow(2) + mu2.pow(2) + self.c1) * (sigma1_sq + sigma2_sq + self.c2)) # (N, C, *)

        if self.size_average:
            return ssim_map.mean()
        else:
            reduced_dims = list(range(1, output.dim()))
            return ssim_map.mean(dim=reduced_dims)
