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
    def _denormalize(imgs):
        """Denormalize the images to [0-255].
        Args:
            imgs (torch.Tensor) (N, C, H, W): Te images to be denormalized.

        Returns:
            imgs (torch.Tensor) (N, C, H, W): The denormalized images.
        """
        imgs = imgs.clone()
        imgs = imgs.mul_(39.616).add_(40.951).clamp(0, 255)
        return imgs
