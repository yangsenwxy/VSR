import torch
import logging
import argparse
import yaml
import imageio
import numpy as np
from box import Box
from tqdm import tqdm
import nibabel as nib
from pathlib import Path

import src
from src.data.transforms import compose


def main(args):
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path).test
    method = config.method
    saved_dir = Path(config.saved_dir)
    data_dir = Path(config.dataset.data_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)

    logging.info('Create the device.')
    if 'cuda' in config.device and not torch.cuda.is_available():
        raise ValueError("The cuda is not available. Please set the device in the trainer section to 'cpu'.")
    device = torch.device(config.device)

    logging.info('Create the testing data.')
    data_paths = sorted(data_dir.glob('**/*2d+1d*.nii.gz'))
    if len(data_paths) == 0:
        raise RuntimeError("The preprocessed files are not found. Please execute the misr_preprocess.py first.")
    degrade = compose(config.dataset.degrade)
    post_transforms = compose(config.dataset.post_transforms)

    if method in ['sisr', 'misr']:
        logging.info("Create the pre-trained model.")
        net = _get_instance(src.model.nets, config.net).to(device)
        checkpoint = torch.load(config.net.ckpt_path, map_location=device)
        net.load_state_dict(checkpoint['net'])

    trange = tqdm(data_paths, total=len(data_paths), desc="Test")
    if method in ['sisr', 'bicubic', 'gt']:
        for path in trange:
            file_name = path.parts[-1].split('.')[0]
            patient_name = path.parts[-2]

            # Dim: (H, W, 1, T)
            img = nib.load(str(path)).get_data().astype(np.float32)
            num_frames = img.shape[-1]

            sr_video = []
            for i in range(num_frames):
                if method == 'sisr':
                    hr_img = img[..., i]
                    lr_img = degrade(hr_img)
                    lr_img = post_transforms(lr_img).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
                    sr_img = net(lr_img)
                    sr_img = sr_img.squeeze().detach().cpu().numpy()
                elif method == 'bicubic':
                    import cv2
                    hr_img = img[..., i]
                    lr_img = degrade(hr_img).squeeze()
                    sr_img = cv2.resize(lr_img, (lr_img.shape[1]*config.upscale_factor, lr_img.shape[0]*config.upscale_factor), interpolation=cv2.INTER_CUBIC)
                elif method == 'gt':
                    sr_img = img[..., i].squeeze()
                sr_video.append(sr_img)

            # Dim: (T, H, W)
            sr_video = np.stack(sr_video)
            sr_video = _quantize(sr_video)

            # Save the super resolution video.
            output_path = saved_dir / patient_name
            if not output_path.is_dir():
                output_path.mkdir()
            _dump_video(sr_video, output_path / f'{file_name}.gif')

    elif method == 'misr':
        for path in trange:
            file_name = path.parts[-1].split('.')[0]
            patient_name = path.parts[-2]

            # Dim: (H, W, 1, T)
            img = nib.load(str(path)).get_data().astype(np.float32)
            hr_img = img.transpose(3, 0, 1, 2)
            lr_img = degrade(*list(hr_img))
            lr_img = post_transforms(*lr_img)
            lr_img = [img.permute(2,0,1).unsqueeze(dim=0).to(device) for img in lr_img]
            with torch.no_grad():
                sr_img = net(lr_img)
            sr_img = [img.squeeze().detach().cpu().numpy() for img in sr_img]

            # Dim: (T, H, W)
            sr_video = np.stack(sr_img)
            sr_video = _quantize(sr_video)

            # Save the super resolution video.
            output_path = saved_dir / patient_name
            if not output_path.is_dir():
                output_path.mkdir()
            _dump_video(sr_video, output_path / f'{file_name}.gif')


def _parse_args():
    parser = argparse.ArgumentParser(description="The data preprocessing.")
    parser.add_argument('config_path', type=Path, help='The path of the config file.')
    args = parser.parse_args()
    return args


def _get_instance(module, config, *args):
    """
    Args:
        module (module): The python module.
        config (Box): The config to create the class object.

    Returns:
        instance (object): The class object defined in the module.
    """
    cls = getattr(module, config.name)
    kwargs = config.get('kwargs')
    return cls(*args, **config.kwargs) if kwargs else cls(*args)


def _dump_video(video, path):
    """
    Args:
        video (np.ndarray): The video.
        path (Path or str): The path of the output video.
    """
    with imageio.get_writer(path) as writer:
        for i in range(video.shape[0]):
            writer.append_data(video[i])


def _quantize(video):
    """Apply min-max normalization and quantize the data to 0-255 (uint8).
    Args:
        video (np.ndarray): The video.

    Returns:
        video (np.ndarray): The video after the quantization.
    """
    video = video.copy()
    video = (video - video.min()) / (video.max() - video.min())
    video = (video * 255).astype(np.uint8)
    return video


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
