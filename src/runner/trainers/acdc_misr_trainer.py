import torch

from src.runner.trainers.base_trainer import BaseTrainer


class AcdcMISRTrainer(BaseTrainer):
    """The ACDC trainer for the Multi-Images Super Resolution.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.

        Returns:
            inputs (list of torch.Tensor): The data inputs.
            targets (list of torch.Tensor): The data targets.
        """
        return batch['lr_imgs'], batch['hr_imgs']

    def _compute_losses(self, outputs, targets):
        """Compute the losses.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            targets (list of torch.Tensor): The data targets.

        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        raise NotImplementedError

    def _compute_metrics(self, outputs, targets):
        """Compute the metrics.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            targets (list of torch.Tensor): The data targets.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        raise NotImplementedError

    @staticmethod
    def _denormalize(imgs):
        """Denormalize the images to [0-255].
        Args:
            imgs (torch.Tensor) (N, C, H, W): Te images to be denormalized.

        Returns:
            imgs (torch.Tensor) (N, C, H, W): The denormalized images.
        """
        imgs = imgs.clone()
        imgs = imgs.mul_(39.616).add_(40.951).clamp(0, 255).round()
        return imgs
