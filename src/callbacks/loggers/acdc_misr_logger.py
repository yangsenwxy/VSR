import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from .base_logger import BaseLogger


class AcdcMISRLogger(BaseLogger):
    """The ACDC logger for the MISR task.
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

        train_hr_img = make_grid(train_batch['hr_imgs'][-1], nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_sr_img = make_grid(train_outputs[-1], nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_hr_img = make_grid(valid_batch['hr_imgs'][-1], nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_sr_img = make_grid(valid_outputs[-1], nrow=1, normalize=True, scale_each=True, pad_value=1)

        train_grid = torch.cat([train_sr_img, train_hr_img], dim=-1)
        valid_grid = torch.cat([valid_sr_img, valid_hr_img], dim=-1)
        with SummaryWriter(self.log_dir) as writer:
            writer.add_image('train', train_grid)
            writer.add_image('valid', valid_grid)
