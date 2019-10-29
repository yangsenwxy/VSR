import torch
import pickle
import numpy as np
import nibabel as nib

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class AcdcVSRRefineNetDataset(BaseDataset):
    """The dataset of the Automated Cardiac Diagnosis Challenge (ACDC) in MICCAI 2017 for the Video Super-Resolution.
    
    Ref: https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html
    
    Args:
        downscale_factor (int): The downscale factor (2, 3, 4).
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
        num_frames (int): The number of the frames of a sequence (default: 5).
        temporal_order (str): The order to form the sequence (default: 'last').
            'last': The sequence would be {t-n+1, ..., t-1, t}.
            'middle': The sequence would be {t-(n-1)//2, ..., t-1, t, t+1, ..., t+[(n-1)-(n-1)//2]}.
    """
    def __init__(self, downscale_factor, transforms, pos_code_path, augments=None, num_frames=5, temporal_order='last', **kwargs):
        super().__init__(**kwargs)
        if downscale_factor not in [2, 3, 4]:
            raise ValueError(f'The downscale factor should be 2, 3, 4. Got {downscale_factor}.')
        self.downscale_factor = downscale_factor

        self.transforms = compose(transforms)
        self.augments = compose(augments)        
        self.num_frames = num_frames
        self.pos_code_path = pos_code_path
        
        if temporal_order not in ['last', 'middle']:
            raise ValueError(f"The temporal order should be 'last' or 'middle'. Got {temporal_order}.")
        self.temporal_order = temporal_order        

        # Save the data paths and the target frame index for training; only need to save the data paths
        # for validation to process dynamic length of the sequences.
        lr_paths = sorted((self.data_dir / self.type / 'LR' / f'X{downscale_factor}').glob('**/*2d+1d*.nii.gz'))
        hr_paths = sorted((self.data_dir / self.type / 'HR').glob('**/*2d+1d*.nii.gz'))
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
        lr_imgs = nib.load(str(lr_path)).get_data() # (H, W, C, T)
        hr_imgs = nib.load(str(hr_path)).get_data() # (H, W, C, T)
        all_imgs = [lr_imgs[..., t] for t in range(lr_imgs.shape[-1])] + \
                   [hr_imgs[..., t] for t in range(hr_imgs.shape[-1])]
        
        if self.type == 'train':
            all_imgs = self.augments(*all_imgs)
        all_imgs = self.transforms(*all_imgs)
        all_imgs = [img.permute(2, 0, 1).contiguous() for img in all_imgs]
        lr_imgs, hr_imgs = all_imgs[:len(all_imgs) // 2], all_imgs[len(all_imgs) // 2:]    
        all_imgs = all_imgs[:len(all_imgs) // 2]
        start, num_all_frames = 0, len(all_imgs)
        
        # Position encoding
        with open(self.pos_code_path, 'rb') as f:
            pos_codes = pickle.load(f)
        filename = lr_path.parts[-1].split('.')[0]
        patient, _, _ = filename.split('_')
        pos_code = pos_codes[patient]
        pos_code = self.transforms(pos_code, normalize_tags=[False])
        
        if self.type == 'train':
            if self.num_frames > len(lr_imgs):
                lr_imgs, hr_imgs = lr_imgs+lr_imgs, hr_imgs+hr_imgs
            
            # Compute the start and the end index of the sequence according to the temporal order.
            n = self.num_frames
            T = len(lr_imgs)
            if self.temporal_order == 'last':
                start, end = t - n + 1, t + 1
            elif self.temporal_order == 'middle':
                start, end = t - (n - 1) // 2, t + ((n - 1) - (n - 1) // 2) + 1
            if start < 0:
                start %= T
                lr_imgs = lr_imgs[start:] + lr_imgs[:end]
                hr_imgs = hr_imgs[start:] + hr_imgs[:end]
            elif end > T:
                end %= T
                lr_imgs = lr_imgs[start:] + lr_imgs[:end]
                hr_imgs = hr_imgs[start:] + hr_imgs[:end]
            else:
                lr_imgs = lr_imgs[start:end]
                hr_imgs = hr_imgs[start:end]
        else:
            start = (-6) % len(lr_imgs)
            lr_imgs = lr_imgs[-6:] + lr_imgs + lr_imgs[:6]
            hr_imgs = hr_imgs[-6:] + hr_imgs + hr_imgs[:6]
        
        all_imgs = all_imgs + [all_imgs[-1]] * (35-num_all_frames)
        pos_code = pos_code.repeat(35 // num_all_frames + 2)[:35].unsqueeze(1)
        return {'lr_imgs': lr_imgs, 'hr_imgs': hr_imgs, 'all_imgs': all_imgs, 'index': index, \
                'frame_start': start, 'num_all_frames': num_all_frames, 'pos_code': pos_code}