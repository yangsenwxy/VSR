import csv
import torch
import logging
import imageio
import numpy as np
from scipy.misc import imsave
from tqdm import tqdm
from pathlib import Path

from src.runner.predictors.base_predictor import BasePredictor


class AcdcMISRDUFPredictor(BasePredictor):
    """The ACDC predictor for DUFNet.
    Args:
        export_prediction (bool): Export the predicted video and metrics or not. Default: False.
        saved_dir (Path): The export destination folder. Used when `export_prediction` is true. Default: None.
    """
    def __init__(self, export_prediction=False, saved_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.export_prediction = export_prediction
        self.saved_dir = Path(saved_dir) if export_prediction else None

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.

        Returns:
            inputs (list of torch.Tensor): The data inputs.
            target (torch.Tensor): The data target.
            index (int): The index of the target path in the `dataloder.data`.
        """
        return batch['lr_imgs'], batch['hr_img'], batch['index']

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

    def predict(self):
        """The testing process.
        """
        self.net.eval()
        trange = tqdm(self.test_dataloader,
                      total=len(self.test_dataloader),
                      desc='testing')

        if self.export_prediction:
            video_dir = self.saved_dir / 'video'
            img_dir = self.saved_dir / 'img'
            csv_path = self.saved_dir / 'result.csv'

            tmp_sid = None
            sr_video, metrics_table, header = [], [], ['name']
            header += [metric_fn.__class__.__name__ for metric_fn in self.metric_fns]
            header += [loss_fns.__class__.__name__ for loss_fns in self.loss_fns]
            metrics_table.append(header)

        log = self._init_log()
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            inputs, targets, index = self._get_inputs_targets(batch)
            with torch.no_grad():
                outputs = self.net(inputs)
                losses = self._compute_losses(outputs, targets)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                metrics = self._compute_metrics(outputs, targets)

                if self.export_prediction:
                    path, t = self.test_dataloader.dataset.data[index]
                    filename = path.parts[-1].split('.')[0]
                    patient, _, sid = filename.split('_')
                    fid = f'frame{t+1:0>2d}'
                    
                    _losses = [loss.cpu().numpy() for loss in losses]
                    _metrics = [metric.cpu().numpy() for metric in metrics]
                    filename = filename.replace("+1d", "").replace("sequence", "slice") + f'_{fid}'
                    metrics_table.append([filename, *_metrics, *_losses])

                    # Save as the gif file
                    if sid != tmp_sid and index != 0:
                        sr_video = np.stack(sr_video)
                        output_path = video_dir / patient
                        if not output_path.is_dir():
                            output_path.mkdir(parents=True)
                        
                        self._dump_video(sr_video, output_path / f'{tmp_sid}.gif')
                        sr_video = []

                    outputs = self._min_max_normalize(outputs) * 255
                    sr_img = outputs.squeeze().detach().cpu().numpy().astype(np.uint8)
                    sr_video.append(sr_img)
                    tmp_sid = sid

                    # Save as the png file
                    output_path = img_dir / patient
                    if not output_path.is_dir():
                        output_path.mkdir(parents=True)
                    img_name = sid.replace('sequence', 'slice') + f'_{fid}.png'
                    imsave(output_path / img_name, sr_img)
                    
            batch_size = self.test_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        # Save the metrics
        if self.export_prediction:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(metrics_table)

        for key in log:
            log[key] /= count
        logging.info(f'Test log: {log}.')

    def _dump_video(self, video, path):
        """
        Args:
            video (np.ndarray): The video.
            path (Path or str): The path of the output video.
        """
        with imageio.get_writer(path) as writer:
            for i in range(video.shape[0]):
                writer.append_data(video[i])
