import logging
from pathlib import Path
from typing import List

import numpy as np
from tifffile import imread
from tqdm import tqdm


def average_of_mips(
    image_files: List[Path],
    logger=logging,
):
    """
    Computes a maximum projection of each input image file (assumed to contain a 3D stack), and returns an average of
    all projections.

    Args:
        image_files: list of input image files, assumed to contain a 3D stack each
        logger:

    Returns: an average of maximum projections of all input files (stacks).

    """
    return average_of_thresholded_mips(
        image_files=image_files,
        logger=logger,
    )


def average_of_thresholded_mips(
    image_files: List[Path],
    lower_threshold: int = 0,
    upper_threshold: int = 65535,
    logger=logging,
):
    """
    Computes a maximum projection of each input image file (assumed to contain a 3D stack),
    applies a provided threshold, and returns an average of the thresholded signal of all projections.

    Args:
        image_files: list of input image files, assumed to contain a 3D stack each
        lower_threshold:
        upper_threshold:
        logger:

    Returns: an average of maximum projections of all input files (stacks).

    """

    def image_generator():
        for path in tqdm(sorted(image_files)):
            logger.info(f"Processing {path}.")
            yield imread(path)

    def apply_mask(img, mask):
        img[~mask] = 0
        return img

    images = image_generator()
    image = next(images)
    mask_image = (lower_threshold <= image) & (image <= upper_threshold)

    count_image = np.zeros(mask_image.shape[1:], dtype=np.uint16)
    sum_image = np.max(apply_mask(image, mask_image), axis=0).astype(np.float64)
    count_image += np.max(mask_image, axis=0)

    for image in images:
        mask_image = (lower_threshold <= image) & (image <= upper_threshold)
        sum_image += np.max(apply_mask(image, mask_image), axis=0)
        count_image += np.max(mask_image, axis=0)

    with np.errstate(invalid="ignore"):
        return sum_image / count_image
