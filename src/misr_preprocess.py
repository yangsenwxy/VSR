import logging
import argparse
import random
import numpy as np
import nibabel as nib
from pathlib import Path


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir

    # Random split the patient directorys for training and validation (8:2).
    patient_dirs = sorted([dir_ for dir_ in (data_dir / 'training').iterdir() if dir_.is_dir()])
    random.seed('training') # Use a defined random seed.

    train_dirs = sorted(random.sample(population=patient_dirs, k=int(len(patient_dirs) * 0.8)))
    train_paths = []
    for dir_ in train_dirs:
        train_paths.extend(dir_.glob('*4d.nii.gz'))

    valid_dirs = sorted(list(set(patient_dirs) - set(train_dirs)))
    valid_paths = []
    for dir_ in valid_dirs:
        valid_paths.extend(dir_.glob('*4d.nii.gz'))

    test_paths = sorted((data_dir / 'testing').glob('**/*4d.nii.gz'))

    for type_, paths in zip(['train', 'valid', 'test'], [train_paths, valid_paths, test_paths]):
        pixel_sum = 0.0
        pixel_square = 0.0
        count = 0
        for path in paths:
            patient_name = path.parts[-2]
            logging.info(f'Process {patient_name}.')

            # Create output directory.
            if not (output_dir / type_ / patient_name).is_dir():
                (output_dir / type_/ patient_name).mkdir(parents=True)

            # Read in the 4D MRI scans.
            img = nib.load(str(path)).get_data() # (H, W, D, T)

            # If the image dtype is 'int16', remove the outlier and then apply the min-max normalization.
            if img.dtype == 'int16':
                hist, _ = np.histogram(img.ravel(), bins=range(int(img.max()) + 1), density=True)
                cdf = np.cumsum(hist)
                idx = (np.abs(cdf - 0.995)).argmin()
                img[img > idx] = idx
                img = ((img - img.min()) / (img.max() - img.min()) * 255.0).round()
            img = img.astype(np.float32)

            # Record the sum of the 4D MRI scans.
            pixel_sum += img.sum()
            pixel_square += (img ** 2).sum()
            count += img.shape[0] * img.shape[1] * img.shape[2] * img.shape[3]

            # Save each sequence of the slices of the scan into single file.
            for s in range(img.shape[2]):
                _img = img[..., s:s+1, :] # (H, W, C, T)
                nib.save(nib.Nifti1Image(_img, np.eye(4)),
                         str(output_dir / type_ / patient_name / f'{patient_name}_2d+1d_sequence{s+1:0>2d}.nii.gz'))

        # Calculate the mean and the std.
        mean = pixel_sum / count
        E_X_square = pixel_square / count
        std = np.sqrt(E_X_square - mean ** 2)
        logging.info(f'{type_} dataset: mean {mean:.4f}, std {std:.4f}.')

def _parse_args():
    parser = argparse.ArgumentParser(description="The data preprocessing.")
    parser.add_argument('data_dir', type=Path, help='The directory of the dataset.')
    parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
