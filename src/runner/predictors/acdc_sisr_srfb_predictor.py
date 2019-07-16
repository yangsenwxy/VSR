import torch

from src.runner.predictors import AcdcSISRPredictor


class AcdcSISRSRFBPredictor(AcdcSISRPredictor):
    """The ACDC predictor for the Single-Image Super Resolution using the SRFBNet.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_losses(self, outputs, target):
        """Compute the losses.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            target (torch.Tensor): The data target.
        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = []
        for loss_fn in self.loss_fns:
            # Average the losses computed at every time steps.
            loss = torch.stack([loss_fn(output, target) for output in outputs]).mean()
            losses.append(loss)
        return losses

    def _compute_metrics(self, outputs, target):
        """Compute the metrics.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            target (torch.Tensor): The data target.
        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        # Do the min-max normalization before computing the metric.
        output, target = self._min_max_normalize(outputs[-1]), self._min_max_normalize(target)

        metrics = [metric_fn(output, target) for metric_fn in self.metric_fns]
        return metrics
