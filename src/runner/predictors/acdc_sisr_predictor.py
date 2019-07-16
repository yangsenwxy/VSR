import torch

from src.runner.predictors.base_predictor import BasePredictor


class AcdcSISRPredictor(BasePredictor):
    """The ACDC predictor for the Single-Image Super Resolution.
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
        output, target = self._min_max_normalize(output), self._min_max_normalize(target)

        metrics = [metric_fn(output, target) for metric_fn in self.metric_fns]
        return metrics

    @staticmethod
    def _min_max_normalize(imgs):
        """Normalize the images to [0, 1].
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
