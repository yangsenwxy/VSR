import torch

from src.runner.predictors.base_predictor import BasePredictor


class AcdcMISRPredictor(BasePredictor):
    """The ACDC predictor for the Multi-Images Super Resolution.
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
