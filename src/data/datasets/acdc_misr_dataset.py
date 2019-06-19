import torch
import numpy as np
from box import Box
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class AcdcMISRDataset(BaseDataset):
    """ MICCAI 2017 challeng (ref: https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).
    Args:
        transforms (Box): the preprocessing and augmentation techiques applied to the training data
        post_transforms (Box): the postprocessing techiques applied to the data after downscaling
        degrade (Box): the downscaling process applied to the high resolution data
        num_frames (int): the number of the lr images used for super resolution
    """
    def __init__(self, transforms, post_transforms, degrade, num_frames, **kwargs):
        super().__init__(**kwargs)
        self.transforms = compose(transforms)
        self.post_transform = compose(post_transforms)
        self.degrade = compose(degrade)
        self.num_frames = num_frames
        self.downscale_factor = degrade[0].kwargs.downscale_factor
        self.image_path = []
        self.data = np.empty([0, 2], dtype=int)

        _image_path = [path for path in self.data_dir.iterdir() if path.is_dir()]
        for i, path in enumerate(_image_path):
            self.image_path.append(path / f'{path.parts[-1]}_4d.nii.gz')
            # image dim: H, W, D, frames
            metadata = nib.load(str(self.image_path[i]))
            num_slices, num_frames = metadata.header.get_data_shape()[2], metadata.header.get_data_shape()[3]
            # Skip the data if its frames are not enough
            if num_frames < self.num_frames:
                continue
            # Build the look up table
            index_arr = np.stack((np.ones(num_slices, dtype=int)*i, np.arange(num_slices)), axis=1)
            self.data = np.concatenate((self.data, index_arr), axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patient_idx, slice_idx = self.data[index]
        image = nib.load(str(self.image_path[patient_idx])).get_data().astype(np.float32)

        # Keep image dimension divisible by self.downscale_factor
        h, w, r = image.shape[0], image.shape[1], self.downscale_factor
        h0, hn = (h % r) // 2, h - ((h % r) - (h % r) // 2)
        w0, wn = (w % r) // 2, w - ((w % r) - (w % r) // 2)
        _image = image[h0:hn, w0:wn, slice_idx]

        # Generate the low resolution image according to the target frame
        # - Randomly select target frame
        target_frame = np.random.randint(0, image.shape[-1])
        hr_imgs = _image[..., max(0, target_frame-self.num_frames):target_frame]
        if hr_imgs.shape[-1] != self.num_frames:
            hr_imgs = np.concatenate((_image[..., (target_frame-self.num_frames):], hr_imgs), axis=-1)
        # - Convert hr_img into a list of np.array(dim: H, W, 1)
        hr_imgs = hr_imgs.transpose((2, 0, 1))
        hr_imgs = [_img[..., None] for _img in hr_imgs]
        # - Apply transforms
        if self.type == 'train':
            hr_imgs = self.transforms(*hr_imgs)
        lr_imgs = self.degrade(*hr_imgs)
        lr_imgs = self.post_transform(*lr_imgs)
        hr_imgs = self.post_transform(*hr_imgs)
        # - Permute the tensor to (C, H, W)
        lr_imgs = tuple(_img.permute(2, 0, 1).contiguous() for _img in lr_imgs)
        hr_imgs = tuple(_img.permute(2, 0, 1).contiguous() for _img in hr_imgs)

        return {'hr_imgs': hr_imgs, 'lr_imgs': lr_imgs}
