import torch
import numpy as np
from box import Box
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class AcdcMISRDUFDataset(BaseDataset):
    """The Automated Cardiac Diagnosis Challenge (ACDC) in MICCAI 2017 (ref: https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).
    Args:
        transforms (Box): The preprocessing and augmentation techiques applied to the training data.
        post_transforms (Box): The postprocessing techiques applied to the data after downscaling.
        degrade (Box): The downscaling function applied to the high resolution data.
        num_frames (int): The number of the lr images used for super resolution.
        temporal_order (str): The order to form the sequence (default: 'last').
            'last': The sequence would be {t-n+1, ..., t-1, t}.
            'middle': The sequence would be {t-(n-1)//2, ..., t-1, t, t+1, ..., t+[(n-1)-(n-1)//2]}.
        """
    def __init__(self, transforms, post_transforms, degrade, num_frames, temporal_order='last', **kwargs):
        super().__init__(**kwargs)
        self.transforms = compose(transforms)
        self.post_transforms = compose(post_transforms)
        self.degrade = compose(degrade)
        self.num_frames = num_frames
        self.temporal_order = temporal_order
        self.downscale_factor = degrade[0].kwargs.downscale_factor
        self.data_paths = sorted((self.data_dir / self.type).glob('**/*2d+1d*.nii.gz'))
        self.data = [[]]

        for path in self.data_paths:
            header = nib.load(str(path)).header
            dims = header.get_data_shape()
            self.data.extend([[path, i] for i in range(dims[-1])])
        self.data.remove([])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, t = self.data[index]
        imgs = nib.load(str(path)).get_data() # (H, W, C, T)

        # Make the image size divisible by the downscale_factor.
        h, w, r = imgs.shape[0], imgs.shape[1], self.downscale_factor
        h0, hn = (h % r) // 2, h - ((h % r) - (h % r) // 2)
        w0, wn = (w % r) // 2, w - ((w % r) - (w % r) // 2)
        imgs = imgs[h0:hn, w0:wn, ...]

        n = self.num_frames
        T = imgs.shape[-1]
        assert n <= T

        # Compute the start and the end index of the sequence according to the temporal order.
        if self.temporal_order == 'last':
            start, end = t - n + 1, t + 1
        elif self.temporal_order == 'middle':
            start, end = t - (n - 1) // 2, t + ((n - 1) - (n - 1) // 2) + 1

        if start < 0:
            imgs = np.concatenate((imgs[..., start:], imgs[..., :end]), axis=-1)
        elif end > T:
            end %= T
            imgs = np.concatenate((imgs[..., start:], imgs[..., :end]), axis=-1)
        else:
            imgs = imgs[..., start:end]

        hr_imgs = [imgs[..., t] for t in range(imgs.shape[-1])] # list of (H, W, C)

        hr_imgs = self.transforms(*hr_imgs)
        lr_imgs = self.degrade(*hr_imgs)
        lr_imgs = self.post_transforms(*lr_imgs)
        hr_imgs = self.post_transforms(*hr_imgs)
        lr_imgs = [img.permute(2, 0, 1).contiguous() for img in lr_imgs]
        hr_img = hr_imgs[self.num_frames // 2].permute(2, 0, 1).contiguous()
        return {'lr_imgs': lr_imgs, 'hr_img': hr_img}