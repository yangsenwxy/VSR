import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PSNR(nn.Module):
    '''PSRN score
    Args:
        size_average (bool): Average the ssim all over the samples in the batch or get the value per sample. Default: `True`.
        value_range (int): The maximum of the pixel value. Default: `None`.
    '''

    def __init__(self, size_average=True, value_range=None):
        super().__init__()
        self.size_average = size_average
        self.value_range = value_range

    def forward(self, output, target):
        '''
        Args:
            output (torch.tensor) (N, C, *): The model output.
            target (torch.tensor) (N, C, *): The data target.

        Returns:
            score (torch.tensor) (0) or (N): The PSNR score. The return shape depends on the args.size_average.
        '''
        # Normalize the input and the target to be in the range of (0, 1)
        if self.value_range:
            _input = output / self.value_range
            _target = target / self.value_range
        else:
            # Min-max normalization
            _output = (output - output.min()) / (output.max() - output.min())
            _target = (target - target.min()) / (target.max() - target.min())

        # Calculate the PSNR score
        reduce_dim = list(range(1, _output.dim())) # except Batch dimention
        mse = F.mse_loss(_output, _target, reduction='none').mean(reduce_dim)
        psnr = 10 * torch.log10(1.0 / mse)

        if self.size_average:
            return psnr.mean(dim=0)
        else:
            return psnr


class SSIM(nn.Module):
    ''' SSIM score
    ref: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
         https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
    Args:
        dim (int): The dimention of the image. Default: `2`.
        channels (int): The channel number of the image. Default: `1`.
        kernel_size (int): The size of the gaussian kernel. Default: `11`.
        sigma (float): The standard deviation of the gaussian kernel. Default: `1.5`.
        size_average (bool): Average the ssim all over the samples in the batch or get the value per sample. Default: `True`.
        value_range (int): The maximum of the pixel value. Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh). Default: `1`.
    '''

    def __init__(self, dim=2, channels=1, kernel_size=11, sigma=1.5, size_average=True, value_range=1):
        super().__init__()
        self.size_average = size_average
        self.channels = channels
        self.c1 = (0.01 * value_range) ** 2
        self.c2 = (0.03 * value_range) ** 2

        if dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise ValueError(f"Only dim=2, 3 are supported. Received dim={dim}.")

        # The gaussian kernel is the product of the gaussian function of each dimension
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        kernel = 1
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = size // 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, output, target):
        '''
        Args:
            output (torch.tensor) (N, C, *): The model output.
            target (torch.tensor) (N, C, *): The data target.

        Returns:
            score (torch.tensor) (0) or (N): The SSIM score. The return shape depends on the args.size_average.
        '''
        self.weight = self.weight.to(output.device)

        # Calculate the SSIM score
        # - the average of the output and the target
        mu1 = self.conv(output, weight=self.weight, groups=self.groups)
        mu2 = self.conv(target, weight=self.weight, groups=self.groups)
        # - the variance of the output and the target
        sigma1_sq = self.conv(output * output, weight=self.weight, groups=self.groups) - mu1.pow(2)
        sigma2_sq = self.conv(target * target, weight=self.weight, groups=self.groups) - mu2.pow(2)
        # - the covariance of the output and the target
        sigma12 = self.conv(output * target, weight=self.weight, groups=self.groups) - mu1 * mu2
        # - the ssim score
        ssim_map = ((2 * mu1 * mu2 + self.c1) * (2.0 * sigma12 + self.c2)) / ((mu1.pow(2) + mu2.pow(2) + self.c1) * (sigma1_sq + sigma2_sq + self.c2))

        # Return the ssim score
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
