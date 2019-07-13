import torch

from src.runner.trainers.base_trainer import BaseTrainer


class AcdcMISRDUFTrainer(BaseTrainer):
    """The ACDC trainer for DUFNet.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.

        Returns:
            input (torch.Tensor): The data inputs.
            target (torch.Tensor): The data targets.
        """
        return batch['lr_imgs'], batch['hr_img']

    def _compute_losses(self, output, target):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The data target.

        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = [loss(output, target) for loss in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target):
        """Compute the metrics.
        Args:
             output (torch.Tensor): The model output.
             target (torch.Tensor): The data target.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        output, target = self._min_max_normalize(output), self._min_max_normalize(target)
        metrics = [metric(output, target) for metric in self.metric_fns]
        return metrics

    @staticmethod
    def _min_max_normalize(imgs):
        """Normalize the image to [0, 1].
        Args:
            imgs (torch.Tensor) (N, C, H, W): Te images to be normalized.

        Returns:
            imgs (torch.Tensor) (N, C, H, W): The normalized images.
        """
        imgs = imgs.clone()
        for img in imgs:
            min, max = img.min(), img.max()
            img.sub_(min).div_(max - min + 1e-10)
        return imgs
