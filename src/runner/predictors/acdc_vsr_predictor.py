import csv
import torch
import logging
import imageio
import numpy as np
import functools
from scipy.misc import imsave
from tqdm import tqdm
from pathlib import Path

from src.runner.predictors.base_predictor import BasePredictor
from src.utils import denormalize


class AcdcVSRPredictor(BasePredictor):
    """The ACDC predictor for the Video Super-Resolution.
    Args:
        data_type (str): The data type. Default: `2d`.
        saved_dir (str): The directory to save the predicted videos, images and metrics (default: None).
        exported (bool): Whether to export the predicted video, images and metrics (default: False).
    """
    def __init__(self, data_type='2d', saved_dir=None, exported=False, **kwargs):
        super().__init__(**kwargs)
        if self.test_dataloader.batch_size != 1:
            raise ValueError(f'The testing batch size should be 1. Got {self.test_dataloader.batch_size}.')

        if exported:
            self.saved_dir = Path(saved_dir)
        self.data_type = data_type
        self.exported = exported
        self._denormalize = functools.partial(denormalize, dataset='acdc')

    def predict(self):
        """The testing process.
        """
        self.net.eval()
        trange = tqdm(self.test_dataloader,
                      total=len(self.test_dataloader),
                      desc='testing')

        if self.exported:
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
            if self.data_type == '3d':
                D = inputs[0].shape[2]
            with torch.no_grad():
                outputs = self.net(inputs)
                losses = self._compute_losses(outputs, targets)
                loss = (losses.mean(dim=0) * self.loss_weights).sum()
                metrics = self._compute_metrics(outputs, targets)
                if self.exported:                
                    if self.data_type == '2d':
                        lr_path, hr_path = self.test_dataloader.dataset.data[index]
                        filename = lr_path.parts[-1].split('.')[0]
                        patient, _, sid = filename.split('_')

                        filename = filename.replace('2d+1d', '2d').replace('sequence', 'slice')
                        for t, _losses, _metrics in zip(range(T), losses, metrics):
                            _losses = [loss.item() for loss in _losses]
                            _metrics = [metric.item() for metric in _metrics]
                            results.append([filename + f'_frame{t+1:0>2d}', *_metrics, *_losses])

                        outputs = [self._denormalize(output) for output in outputs]
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
                    elif self.data_type == '3d':
                        lr_path, hr_path = self.test_dataloader.dataset.data[index]
                        filename = lr_path.parts[-1].split('.')[0]
                        patient, _ = filename.split('_')

                        for i, (_losses, _metrics) in enumerate(zip(losses, metrics)):
                            _losses = [loss.item() for loss in _losses]
                            _metrics = [metric.item() for metric in _metrics]
                            t, d = i % T, i // T
                            results.append([f'{patient}_2d_slice{d+1:0>2d}_frame{t+1:0>2d}', *_metrics, *_losses])

                        outputs = [self._denormalize(output) for output in outputs]
                        
                        # Save the video.
                        output_dir = videos_dir / patient
                        if not output_dir.is_dir():
                            output_dir.mkdir(parents=True)
                        for i in range(D):
                            sr_imgs = [output[:, :, i].squeeze().detach().cpu().numpy().astype(np.uint8)
                                   for output in outputs]
                            self._dump_video(output_dir / f'sequence{i+1:0>2d}.gif', sr_imgs)

                        # Save the image.
                        output_dir = imgs_dir / patient
                        if not output_dir.is_dir():
                            output_dir.mkdir(parents=True)
                        for i in range(D):
                            sr_imgs = [output[:, :, i].squeeze().detach().cpu().numpy().astype(np.uint8)
                                   for output in outputs]
                            for t, sr_img in enumerate(sr_imgs):
                                img_name = f'slice{i+1:0>2d}_frame{t+1:0>2d}.png'
                                imsave(output_dir / img_name, sr_img)

            batch_size = self.test_dataloader.batch_size
            if self.data_type == '2d':
                self._update_log(log, batch_size, T, loss, losses, metrics)
                count += batch_size * T
            elif self.data_type == '3d':
                self._update_log(log, batch_size, T, loss, losses, metrics, D)
                count += batch_size * T * D
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        # Save the results.
        if self.exported:
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
        if self.data_type == '2d':
            for loss_fn in self.loss_fns:
                losses.append(torch.stack([loss_fn(output, target) for output, target in zip(outputs, targets)]))
            losses = torch.stack(losses, dim=1) # (T, #loss_fns)
        elif self.data_type == '3d':
            for loss_fn in self.loss_fns:
                _losses = []
                for d in range(outputs[0].shape[2]):
                    _losses.extend([loss_fn(output[:, :, d], target[:, :, d]) for output, target in zip(outputs, targets)])    
                losses.append(torch.stack(_losses))
            losses = torch.stack(losses, dim=1) # (T*D, #loss_fns)
        return losses

    def _compute_metrics(self, outputs, targets):
        """Compute the metrics.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            targets (list of torch.Tensor): The data targets.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        outputs = list(map(self._denormalize, outputs))
        targets = list(map(self._denormalize, targets))

        metrics = []
        if self.data_type == '2d':
            for metric_fn in self.metric_fns:
                metrics.append(torch.stack([metric_fn(output, target) for output, target in zip(outputs, targets)]))
            metrics = torch.stack(metrics, dim=1) # (T, #metric_fns)
        elif self.data_type == '3d':
            for metric_fn in self.metric_fns:
                _metrics = []
                for d in range(outputs[0].shape[2]):
                    _metrics.extend([metric_fn(output[:, :, d], target[:, :, d]) for output, target in zip(outputs, targets)])
                metrics.append(torch.stack(_metrics))
            metrics = torch.stack(metrics, dim=1) # (T*D, #metric_fns)
        return metrics

    def _update_log(self, log, batch_size, T, loss, losses, metrics, D=1):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            T (int): The total number of the frames.
            D (int): The total number of the slices.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (sequence of torch.Tensor): The computed losses.
            metrics (sequence of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size * T * D
        for loss_fn, loss in zip(self.loss_fns, losses.mean(dim=0)):
            log[loss_fn.__class__.__name__] += loss.item() * batch_size * T * D
        for metric_fn, metric in zip(self.metric_fns, metrics.mean(dim=0)):
            log[metric_fn.__class__.__name__] += metric.item() * batch_size * T * D

    def _dump_video(self, path, imgs):
        """To dump the video by concatenate the images.
        Args:
            path (Path): The path to save the video.
            imgs (list): The images to form the video.
        """
        with imageio.get_writer(path) as writer:
            for img in imgs:
                writer.append_data(img)