import torch

from src.runner.trainers import AcdcMISRTrainer


class AcdcMISRDUFTrainer(AcdcMISRTrainer):
    """The ACDC trainer for DUFNet.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.

        Returns:
            inputs (list of torch.Tensor): The data inputs.
            target (torch.Tensor): The data target.
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
        # Do the denormalization to [0-255] before computing the metric.
        output, target = self._denormalize(output), self._denormalize(target)

        metrics = [metric(output, target) for metric in self.metric_fns]
        return metrics
