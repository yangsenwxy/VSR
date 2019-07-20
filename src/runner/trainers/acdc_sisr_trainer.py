import torch

from src.runner.trainers.base_trainer import BaseTrainer


class AcdcSISRTrainer(BaseTrainer):
    """The ACDC trainer for the Single-Image Super Resolution.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.

        Returns:
            iutput (torch.Tensor): The data input.
            target (torch.Tensor): The data target.
        """
        return batch['lr_img'], batch['hr_img']

    def _compute_losses(self, output, target):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The data target.

        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = [loss_fn(output, target) for loss_fn in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target):
        """Compute the metrics.
        Args:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The data target.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        # Do the min-max normalization before computing the metric.
        output, target = self._denormalize(output), self._denormalize(target)

        metrics = [metric_fn(output, target) for metric_fn in self.metric_fns]
        return metrics

    @staticmethod
    def _denormalize(imgs):
        """Denormalize the images to [0-255].
        Args:
            imgs (torch.Tensor) (N, C, H, W): Te images to be denormalized.

        Returns:
            imgs (torch.Tensor) (N, C, H, W): The denormalized images.
        """
        imgs = imgs.clone()
        imgs = imgs.mul_(39.616).add_(40.951).clamp(0, 255)
        return imgs
