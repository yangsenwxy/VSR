import torch

from src.runner.trainers import AcdcMISRTrainer


class AcdcMISRSRFBTrainer(AcdcMISRTrainer):
    """The ACDC trainer for the Multi-Images Super Resolution using the SRFBNet-based networks.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_losses(self, outputs, targets):
        """Compute the losses.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            targets (list of torch.Tensor): The data targets.

        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = []
        for loss_fn in self.loss_fns:
            # Average the losses computed at every time steps.
            loss = torch.stack([loss_fn(output, target) for output, target in zip(outputs, targets)]).mean()
            losses.append(loss)
        return losses

    def _compute_metrics(self, outputs, targets):
        """Compute the metrics.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            targets (list of torch.Tensor): The data targets.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        # Do the denormalization to [0-255] before computing the metric.
        outputs = list(map(self._denormalize, outputs))
        targets = list(map(self._denormalize, targets))

        # Average the metric of every frame in a video.
        metrics = []
        for metric_fn in self.metric_fns:
            metric = torch.stack([metric_fn(output, target) for output, target in zip(outputs, targets)]).mean()
            metrics.append(metric)
        return metrics
