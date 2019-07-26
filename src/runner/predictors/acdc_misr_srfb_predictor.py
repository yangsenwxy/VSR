import csv
import torch
import logging
import numpy as np
from scipy.misc import imsave
from tqdm import tqdm

from src.runner.predictors import AcdcMISRPredictor


class AcdcMISRSRFBPredictor(AcdcMISRPredictor):
    """The ACDC predictor for the Multi-Images Super Resolution using the SRFBNet-based networks.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self):
        """The testing process.
        """
        self.net.eval()
        trange = tqdm(self.test_dataloader,
                      total=len(self.test_dataloader),
                      desc='testing')

        if self.export_prediction:
            videos_dir = self.saved_dir / 'videos'
            imgs_dir = self.saved_dir / 'imgs'
            csv_path = self.saved_dir / 'results.csv'

            sr_imgs = []
            tmp_sid = None
            header = ['name'] + \
                     [metric_fn.__class__.__name__ for metric_fn in self.metric_fns] + \
                     [loss_fns.__class__.__name__ for loss_fns in self.loss_fns]
            results = [header]

        log = self._init_log()
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            inputs, targets, index = self._get_inputs_targets(batch)
            T = len(inputs)
            with torch.no_grad():
                outputs = self.net(inputs)
                losses = self._compute_losses(outputs, targets)
                loss = (losses.mean(dim=0) * self.loss_weights).sum()
                metrics = self._compute_metrics(outputs, targets)

                if self.export_prediction:
                    path = self.test_dataloader.dataset.data_paths[index]
                    filename = path.parts[-1].split('.')[0]
                    patient, _, sid = filename.split('_')

                    filename = filename.replace('2d+1d', '2d').replace('sequence', 'slice')
                    for t, _losses, _metrics in zip(range(T), losses, metrics):
                        _losses = [loss.item() for loss in _losses]
                        _metrics = [metric.item() for metric in _metrics]
                        results.append([filename + f'_frame{t+1:0>2d}', *_metrics, *_losses])

                    outputs = [(self._denormalize(output) * 255).round() for output in outputs]
                    sr_imgs = [output.squeeze().detach().cpu().numpy().astype(np.uint8)
                               for output in outputs]

                    # Save the video.
                    output_dir = videos_dir / patient
                    if not output_dir.is_dir():
                        output_dir.mkdir(parents=True)
                    self._dump_video(output_dir / f'{sid}.gif', sr_imgs)

                    # Save the image.
                    output_dir = imgs_dir / patient
                    if not output_dir.is_dir():
                        output_dir.mkdir(parents=True)
                    for t, sr_img in enumerate(sr_imgs):
                        img_name = sid.replace('sequence', 'slice') + f'_frame{t+1:0>2d}.png'
                        imsave(output_dir / img_name, sr_img)

            batch_size = self.test_dataloader.batch_size
            self._update_log(log, batch_size, T, loss, losses, metrics)
            count += batch_size * T
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        # Save the results.
        if self.export_prediction:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results)

        for key in log:
            log[key] /= count
        logging.info(f'Test log: {log}.')

    def _get_inputs_targets(self, batch):
        """Specify the data inputs and targets.
        Args:
            batch (dict): A batch of data.

        Returns:
            inputs (list of torch.Tensor): The data inputs.
            targets (list of torch.Tensor): The data targets.
            index (int): The index of the target path in the `dataloder.data`.
        """
        return batch['lr_imgs'], batch['hr_imgs'], batch['index']

    def _compute_losses(self, outputs, targets):
        """Compute the losses.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            targets (list of torch.Tensor): The data targets.

        Returns:
            losses (torch.Tensor): The computed losses.
        """
        losses = []
        for loss_fn in self.loss_fns:
            losses.append(torch.stack([loss_fn(output, target) for output, target in zip(outputs, targets)]))
        losses = torch.stack(losses, dim=1) # (T, #loss_fns)
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

        metrics = []
        for metric_fn in self.metric_fns:
            metrics.append(torch.stack([metric_fn(output, target) for output, target in zip(outputs, targets)]))
        metrics = torch.stack(metrics, dim=1) # (T, #metric_fns)
        return metrics

    def _update_log(self, log, batch_size, T, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            T (int): The total number of the frames.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (sequence of torch.Tensor): The computed losses.
            metrics (sequence of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size * T
        for loss_fn, loss in zip(self.loss_fns, losses.mean(dim=0)):
            log[loss_fn.__class__.__name__] += loss.item() * batch_size * T
        for metric_fn, metric in zip(self.metric_fns, metrics.mean(dim=0)):
            log[metric_fn.__class__.__name__] += metric.item() * batch_size * T
