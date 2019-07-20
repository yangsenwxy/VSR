import torch
import imageio
from pathlib import Path

from src.runner.predictors.base_predictor import BasePredictor


class AcdcMISRPredictor(BasePredictor):
    """The ACDC predictor for the Multi-Images Super Resolution.
    Args:
        saved_dir (str): The directory to save the predicted videos, images and metrics (default: None).
        export_prediction (bool): Whether to export the predicted video, images and metrics (default: False).
    """
    def __init__(self, saved_dir=None, export_prediction=False, **kwargs):
        super().__init__(**kwargs)
        if export_prediction:
            if self.test_dataloader.batch_size != 1:
                raise ValueError(f'The batch size should be 1 if export_prediction is True. Got {self.test_dataloader.batch_size}.')
            #if self.test_dataloader.shuffle is True:
            #    raise ValueError('The shuffle should be False if export_prediction is True.')
            self.saved_dir = Path(saved_dir)
        self.export_prediction = export_prediction

    def _dump_video(self, path, imgs):
        """To dump the video by concatenate the images.
        Args:
            path (Path): The path to save the video.
            imgs (list): The images to form the video.
        """
        with imageio.get_writer(path) as writer:
            for img in imgs:
                writer.append_data(img)

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
