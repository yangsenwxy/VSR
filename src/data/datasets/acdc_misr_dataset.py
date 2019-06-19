import torch
import numpy as np
from box import Box
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class AcdcMISRDataset(BaseDataset):
    """The Automated Cardiac Diagnosis Challenge (ACDC) in MICCAI 2017 (ref: https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).
    Args:
        transforms (Box): The preprocessing and augmentation techiques applied to the training data.
        post_transforms (Box): The postprocessing techiques applied to the data after downscaling.
        degrade (Box): The downscaling function applied to the high resolution data.
        num_frames (int): The number of the lr images used for super resolution.
        """
    def __init__(self, transforms, post_transforms, degrade, num_frames, **kwargs):
        super().__init__(**kwargs)
        self.transforms = compose(transforms)
        self.post_transforms = compose(post_transforms)
        self.degrade = compose(degrade)
        self.num_frames = num_frames
        self.downscale_factor = degrade[0].kwargs.downscale_factor
        self.data_paths = sorted((self.data_dir / self.type).glob('**/*2d+1d*.nii.gz'))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        img = nib.load(str(self.data_paths[index])).get_data() # (H, W, C, T)

        # Make the image size divisible by the downscale_factor.
        h, w, r = img.shape[0], img.shape[1], self.downscale_factor
        h0, hn = (h % r) // 2, h - ((h % r) - (h % r) // 2)
        w0, wn = (w % r) // 2, w - ((w % r) - (w % r) // 2)
        img = img[h0:hn, w0:wn, ...]

        # Generate the low resolution image according to the target frame
        # - Randomly select the target frame
        t = np.random.randint(0, img.shape[-1])
        hr_imgs = img[..., max(0, t - self.num_frames):t]
        if hr_imgs.shape[-1] != self.num_frames:
            hr_imgs = np.concatenate((img[..., t - self.num_frames:], hr_imgs), axis=-1)
        hr_imgs = [img[..., t] for t in range(hr_imgs.shape[-1])] # list of (H, W, C)
        # - Apply transforms
        if self.type == 'train':
            hr_imgs = self.transforms(*hr_imgs)
        lr_imgs = self.degrade(*hr_imgs)
        lr_imgs = self.post_transforms(*lr_imgs)
        hr_imgs = self.post_transforms(*hr_imgs)
        lr_imgs = [img.permute(2, 0, 1).contiguous() for img in lr_imgs]
        hr_imgs = [img.permute(2, 0, 1).contiguous() for img in hr_imgs]
        return {'lr_imgs': lr_imgs, 'hr_imgs': hr_imgs}
