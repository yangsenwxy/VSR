import torch

from src.runner.trainers.base_trainer import BaseTrainer


class AcdcMISRTrainer(BaseTrainer):
    """The ACDC trainer for Multi-Images Super Resolution.

    Specificly, there are multiple LR inputs and corresponding HR targets.
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
        losses = [loss(outputs, targets) for loss in self.losses]
        return losses

    def _compute_metrics(self, outputs, targets):
        """Compute the metrics.
        Args:
             outputs (list of torch.Tensor): The model outputs.
             targets (list of torch.Tensor): The data targets.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        metrics = [metric(outputs[-1], targets[-1]) for metric in self.metrics]
        return metrics
