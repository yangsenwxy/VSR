import torch
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from box import Box
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class AcdcMISRDataset(BaseDataset):
    """ MICCAI 2017 challeng (ref: https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).
    Args:
        transforms (Box): the preprocessing and augmentation techiques applied to the training data
        num_frames (int): the number of the lr images used for super resolution
        downscale_factor (int): the degrade factor of the lr image
    """
    def __init__(self, transforms, num_frames, downscale_factor, **kwargs):
        super().__init__(**kwargs)
        self.data_root = self.data_root / 'training' if self.type != 'testing' else self.data_root / 'testing'
        self.transforms = compose(transforms)
        self.num_frames = num_frames
        self.downscale_factor = downscale_factor

        self.degrade = Degrade(downscale_factor)
        self.post_transform = compose([Box({'name': 'Normalize'}),
                                       Box({'name': 'ToTensor'})])
        self.image_path = []
        self.data = np.empty([0, 2], dtype=int)

        _image_path = [path for path in self.data_root.iterdir() if path.is_dir()]
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
        image = nib.load(str(self.image_path[patient_idx])).get_data().astype(float)

        # Keep image dimension divisible by self.downscale_factor
        h, w, r = image.shape[0], image.shape[1], self.downscale_factor
        h0, hn = (h % r) // 2, h - ((h % r) - (h % r) // 2)
        w0, wn = (w % r) // 2, w - ((w % r) - (w % r) // 2)
        _image = image[h0:hn, w0:wn, slice_idx]

        # Generate the low resolution image according to the target frame
        target_frame = np.random.randint(0, image.shape[-1])
        hr_img = _image[..., max(0, target_frame-self.num_frames):target_frame]
        if hr_img.shape[-1] != self.num_frames:
            hr_img = np.concatenate((_image[..., (target_frame-self.num_frames):], hr_img), axis=-1)
        # Convert hr_img into a list of np.array(dim: H, W, 1)
        hr_img = hr_img.transpose((2, 0, 1))
        hr_img = [_img[..., None] for _img in hr_img]
        # Apply transforms
        if self.type == 'train':
            hr_img = self.transforms(*hr_img)
        lr_img = self.post_transform(*self.degrade(hr_img))
        hr_img = self.post_transform(hr_img[-1])

        return {'hr_img': hr_img, 'lr_img': lr_img}


class Degrade(object):
    """ The class is used for generating low resolution images
    Args:
        downscale_factor (int): the degrade factor
    """
    def __init__(self, downscale_factor):
        self.downscale_factor = downscale_factor

    def _transform_kspace_to_image(self, k, dim=None, img_shape=None):
        if not dim:
            dim = range(k.ndim)
        k = ifftshift(k, axes=dim)
        k = ifftn(k, s=img_shape, axes=dim, norm='ortho')
        img = fftshift(k, axes=dim)
        img = np.abs(img)
        img = img[..., None]
        return img

    def _transform_image_to_kspace(self, img, dim=None, k_shape=None):
        img = img.squeeze()
        if not dim:
            dim = range(img.ndim)
        img = ifftshift(img, axes=dim)
        img = fftn(img, s=k_shape, axes=dim, norm='ortho')
        k = fftshift(img, axes=dim)
        return k

    def __call__(self, imgs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be down-scaled.

        Returns:
            imgs (tuple of numpy.ndarray): The degraded images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        _imgs = []
        for img in imgs:
            if img.ndim == 3:
                # 2D image: H, W, C
                kspace = self._transform_image_to_kspace(img)
                kx_max = kspace.shape[0]//2
                ky_max = kspace.shape[1]//2
                lx = kspace.shape[0]//self.downscale_factor
                ly = kspace.shape[1]//self.downscale_factor
                img = self._transform_kspace_to_image(kspace[kx_max - lx // 2 : kx_max + (lx - (lx // 2)),
                                                             ky_max - ly // 2 : ky_max + (ly - (ly // 2))])
            elif img.ndim == 4:
                # Need to be tested
                # 3D image: H, W, D, C
                slices_list = []
                for i in range(img.shape[-2]):
                    kspace = self._transform_image_to_kspace(img[:, :, i])
                    kx_max = kspace.shape[0]//2
                    ky_max = kspace.shape[1]//2
                    # zero padding
                    rect_func = np.zeros_like(kspace)
                    rect_func[kx_max - kx_max//self.downscale_factor : kx_max + kx_max//self.downscale_factor,
                              ky_max - ky_max//self.downscale_factor : ky_max + ky_max//self.downscale_factor] = 1
                    slices_list.append(self._transform_kspace_to_image(kspace * rect_func))
                img = np.stack(slices_list, axis=-2)
            _imgs.append(img)
        imgs = tuple(_imgs)
        return imgs
