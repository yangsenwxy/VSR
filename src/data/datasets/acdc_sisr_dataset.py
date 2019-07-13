import torch
import numpy as np
from box import Box
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class AcdcSISRDataset(BaseDataset):
    """The dataset of the Automated Cardiac Diagnosis Challenge (ACDC) in MICCAI 2017 (ref: https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html) for the Single-Image Super-Resolution.
    Args:
        transforms (Box): The preprocessing and augmentation techiques applied to the training data.
        post_transforms (Box): The postprocessing techiques applied to the data after downscaling.
        degrade (Box): The downscaling function applied to the high resolution data.
        """
    def __init__(self, transforms, post_transforms, degrade, **kwargs):
        super().__init__(**kwargs)
        self.transforms = compose(transforms)
        self.post_transforms = compose(post_transforms)
        self.degrade = compose(degrade)
        self.downscale_factor = degrade[0].kwargs.downscale_factor
        self.data_paths = sorted((self.data_dir / self.type).glob('**/*2d*.nii.gz'))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        img = nib.load(str(self.data_paths[index])).get_data() # (H, W, C)

        # Make the image size divisible by the downscale_factor.
        h, w, r = img.shape[0], img.shape[1], self.downscale_factor
        h0, hn = (h % r) // 2, h - ((h % r) - (h % r) // 2)
        w0, wn = (w % r) // 2, w - ((w % r) - (w % r) // 2)
        hr_img = img[h0:hn, w0:wn, ...]

        # Generate the low resolution image according to the target frame.
        # - Apply transforms
        if self.type == 'train':
            hr_img = self.transforms(hr_img)
        # - Degrade
        lr_img = self.degrade(hr_img)
        hr_img = self.post_transforms(hr_img).permute(2, 0, 1).contiguous()
        lr_img = self.post_transforms(lr_img).permute(2, 0, 1).contiguous()
        return {'lr_img': lr_img, 'hr_img': hr_img}
