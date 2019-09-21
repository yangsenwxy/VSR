import numpy as np
import nibabel as nib

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class AcdcVSRDataset(BaseDataset):
    """The dataset of the Automated Cardiac Diagnosis Challenge (ACDC) in MICCAI 2017 for the Video Super-Resolution.
    
    Ref: https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html
    
    Args:
        data_type (str): The type of the data ('2d' or '3d').
        downscale_factor (int): The downscale factor (2, 3, 4).
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
        num_frames (int): The number of the frames of a sequence (default: 5).
        temporal_order (str): The order to form the sequence (default: 'last').
            'last': The sequence would be {t-n+1, ..., t-1, t}.
            'middle': The sequence would be {t-(n-1)//2, ..., t-1, t, t+1, ..., t+[(n-1)-(n-1)//2]}.
    """
    def __init__(self, data_type, downscale_factor, transforms, augments=None, num_frames=5, temporal_order='last', **kwargs):
        super().__init__(**kwargs)        
        if data_type not in ['2d', '3d']:
            raise ValueError(f"The downscale factor should be '2d' or '3d'. Got {data_type}.")
        self.data_type = data_type
        
        if downscale_factor not in [2, 3, 4]:
            raise ValueError(f'The downscale factor should be 2, 3, 4. Got {downscale_factor}.')
        self.downscale_factor = downscale_factor

        self.transforms = compose(transforms)
        self.augments = compose(augments)        
        self.num_frames = num_frames
        
        if temporal_order not in ['last', 'middle']:
            raise ValueError(f"The temporal order should be 'last' or 'middle'. Got {temporal_order}.")
        self.temporal_order = temporal_order        

        # Save the data paths and the target frame index for training; only need to save the data paths
        # for validation to process dynamic length of the sequences.
        lr_paths = sorted((self.data_dir / self.type / 'LR' / f'X{downscale_factor}').glob(f'**/*{data_type}+1d*.nii.gz'))
        hr_paths = sorted((self.data_dir / self.type / 'HR').glob(f'**/*{data_type}+1d*.nii.gz'))
        if self.type == 'train':
            self.data = []
            for lr_path, hr_path in zip(lr_paths, hr_paths):
                T = nib.load(str(lr_path)).header.get_data_shape()[-1]
                self.data.extend([(lr_path, hr_path, t) for t in range(T)])
        else:
            self.data = [(lr_path, hr_path) for lr_path, hr_path in zip(lr_paths, hr_paths)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.type == 'train':
            lr_path, hr_path, t = self.data[index]
        else:
            lr_path, hr_path = self.data[index]
        lr_imgs = nib.load(str(lr_path)).get_data() # (H, W, C, T) or (H, W, D, C, T)
        hr_imgs = nib.load(str(hr_path)).get_data() # (H, W, C, T) or (H, W, D, C, T)
        
        if self.type == 'train':
            # Compute the start and the end index of the sequence according to the temporal order.
            n = self.num_frames
            T = lr_imgs.shape[-1]
            if self.temporal_order == 'last':
                start, end = t - n + 1, t + 1
            elif self.temporal_order == 'middle':
                start, end = t - (n - 1) // 2, t + ((n - 1) - (n - 1) // 2) + 1
            if start < 0:
                lr_imgs = np.concatenate((lr_imgs[..., start:], lr_imgs[..., :end]), axis=-1)
                hr_imgs = np.concatenate((hr_imgs[..., start:], hr_imgs[..., :end]), axis=-1)
            elif end > T:
                end %= T
                lr_imgs = np.concatenate((lr_imgs[..., start:], lr_imgs[..., :end]), axis=-1)
                hr_imgs = np.concatenate((hr_imgs[..., start:], hr_imgs[..., :end]), axis=-1)
            else:
                lr_imgs = lr_imgs[..., start:end]
                hr_imgs = hr_imgs[..., start:end]
            imgs = [lr_imgs[..., t] for t in range(lr_imgs.shape[-1])] + \
                   [hr_imgs[..., t] for t in range(hr_imgs.shape[-1])] # list of (H, W, C) or (H, W, D, C)
        else:
            imgs = [lr_imgs[..., t] for t in range(lr_imgs.shape[-1])] + \
                   [hr_imgs[..., t] for t in range(hr_imgs.shape[-1])] # list of (H, W, C) or (H, W, D, C)

        if self.type == 'train':
            imgs = self.augments(*imgs)
        imgs = self.transforms(*imgs)
        if self.data_type == '2d':
            imgs = [img.permute(2, 0, 1).contiguous() for img in imgs]
        elif self.data_type == '3d':
            imgs = [img.permute(3, 0, 1, 2).contiguous() for img in imgs]
        lr_imgs, hr_imgs = imgs[:len(imgs) // 2], imgs[len(imgs) // 2:]
        return {'lr_imgs': lr_imgs, 'hr_imgs': hr_imgs, 'index': index}
