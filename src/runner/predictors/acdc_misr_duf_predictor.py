import csv
import torch
import logging
import numpy as np
from scipy.misc import imsave
from tqdm import tqdm

from src.runner.predictors import AcdcMISRPredictor


class AcdcMISRDUFPredictor(AcdcMISRPredictor):
    """The ACDC predictor for the Multi-Images Super Resolution using the DUFNet.
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
            inputs, target, index = self._get_inputs_targets(batch)
            with torch.no_grad():
                output = self.net(inputs)
                losses = self._compute_losses(output, target)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                metrics = self._compute_metrics(output, target)

                if self.export_prediction:
                    path, t = self.test_dataloader.dataset.data[index]
                    filename = path.parts[-1].split('.')[0]
                    patient, _, sid = filename.split('_')
                    fid = f'frame{t+1:0>2d}'

                    _losses = [loss.item() for loss in losses]
                    _metrics = [metric.item() for metric in metrics]
                    filename = filename.replace('2d+1d', '2d').replace('sequence', 'slice') + f'_{fid}'
                    results.append([filename, *_metrics, *_losses])

                    # Save the video.
                    if sid != tmp_sid and index != 0:
                        output_dir = videos_dir / patient
                        if not output_dir.is_dir():
                            output_dir.mkdir(parents=True)
                        self._dump_video(output_dir / f'{tmp_sid}.gif', sr_imgs)
                        sr_imgs = []

                    output = self._denormalize(output) * 255
                    sr_img = output.squeeze().detach().cpu().numpy().astype(np.uint8)
                    sr_imgs.append(sr_img)
                    tmp_sid = sid

                    # Save the image.
                    output_dir = imgs_dir / patient
                    if not output_dir.is_dir():
                        output_dir.mkdir(parents=True)
                    img_name = sid.replace('sequence', 'slice') + f'_{fid}.png'
                    imsave(output_dir / img_name, sr_img)

            batch_size = self.test_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
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
        # Do the denormalization to [0-255] before computing the metric.
        output, target = self._denormalize(output), self._denormalize(target)

        metrics = [metric_fn(output, target) for metric_fn in self.metric_fns]
        return metrics
