import random
import torch
from torchvision.utils import make_grid

from src.callbacks.loggers.base_logger import BaseLogger


class AcdcVSRLogger(BaseLogger):
    """The ACDC logger for the Video Super-Resolution.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_outputs, valid_batch, valid_outputs):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict): The training batch.
            train_outputs (list of torch.Tensor): The training outputs.
            valid_batch (dict): The validation batch.
            valid_outputs (list of torch.Tensor): The validation outputs.
        """

        if train_batch['hr_imgs'][-1].dim() == 5:
            # 3D network
            train_slice_id = random.randint(0, train_batch['hr_imgs'][-1].shape[2]-1)
            train_hr_img = make_grid(train_batch['hr_imgs'][-1][:, :, train_slice_id], nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_sr_img = make_grid(train_outputs[-1][:, :, train_slice_id], nrow=1, normalize=True, scale_each=True, pad_value=1)

            valid_slice_id = random.randint(0, valid_batch['hr_imgs'][-1].shape[2]-1)
            valid_hr_img = make_grid(valid_batch['hr_imgs'][-1][:, :, valid_slice_id], nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_sr_img = make_grid(valid_outputs[-1][:, :, valid_slice_id], nrow=1, normalize=True, scale_each=True, pad_value=1)
        else:
            # 2D network
            train_hr_img = make_grid(train_batch['hr_imgs'][-1], nrow=1, normalize=True, scale_each=True, pad_value=1)
            train_sr_img = make_grid(train_outputs[-1], nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_hr_img = make_grid(valid_batch['hr_imgs'][-1], nrow=1, normalize=True, scale_each=True, pad_value=1)
            valid_sr_img = make_grid(valid_outputs[-1], nrow=1, normalize=True, scale_each=True, pad_value=1)

        train_grid = torch.cat([train_hr_img, train_sr_img], dim=-1)
        valid_grid = torch.cat([valid_hr_img, valid_sr_img], dim=-1)
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)
