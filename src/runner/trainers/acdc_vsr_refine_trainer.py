import torch
from tqdm import tqdm
import functools

from src.runner.trainers import AcdcVSRTrainer


class AcdcVSRRefineTrainer(AcdcVSRTrainer):
    """The ACDC trainer for the Video Super-Resolution.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('training' or 'validation').
        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            inputs, targets, pos_code = self._get_inputs_targets(batch)
            T = len(inputs)
            if mode == 'training':
                outputs = self.net(inputs, pos_code)
                losses = self._compute_losses(outputs, targets)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.net(inputs, pos_code)
                    losses = self._compute_losses(outputs, targets)
                    loss = (torch.stack(losses) * self.loss_weights).sum()
            metrics =  self._compute_metrics(outputs, targets)

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, T, loss, losses, metrics)
            count += batch_size * T
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count
        return log, batch, outputs

    def _get_inputs_targets(self, batch):
        """Specify the data inputs and targets.
        Args:
            batch (dict): A batch of data.

        Returns:
            inputs (list of torch.Tensor): The data inputs.
            targets (list of torch.Tensor): The data targets.
        """
        return batch['lr_imgs'], batch['hr_imgs'], batch['pos_code']
    
    def _compute_losses(self, outputs, targets):
        """Compute the losses.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            targets (list of torch.Tensor): The data targets.

        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        if self.net.training:
            losses = []
            for loss_fn in self.loss_fns:
                loss = []
                for _outputs in outputs:
                    # Average the losses computed at every time steps.
                    loss.append(torch.stack([loss_fn(output, target) for output, target in zip(_outputs, targets)]).mean())
                # Average the losses computed at every refinement stages.
                loss = torch.stack(loss).mean()
                losses.append(loss)
        else:
            losses = []
            for loss_fn in self.loss_fns:
                # Average the losses computed at every time steps.
                loss = torch.stack([loss_fn(output, target) for output, target in zip(outputs[-1], targets)]).mean()
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
        outputs = list(map(self._denormalize, outputs[-1]))
        targets = list(map(self._denormalize, targets))

        # Average the metric of every frame in a video.
        metrics = []
        for metric_fn in self.metric_fns:
            metric = torch.stack([metric_fn(output, target) for output, target in zip(outputs, targets)]).mean()
            metrics.append(metric)
        return metrics
