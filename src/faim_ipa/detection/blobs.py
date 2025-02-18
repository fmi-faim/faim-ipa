from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_laplace
from skimage.feature import peak_local_max
from skimage.feature.blob import _prune_blobs
from skimage.morphology import ball, h_maxima
from skimage.util import img_as_float32

from faim_ipa.detection.utils import estimate_log_rescale_factor


def detect_blobs(
    img: np.ndarray,
    axial_sigma: float,
    lateral_sigma: float,
    h: int,
    scale_factors: list[int],
    overlap: float,
    background_img: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Detect blobs of different sizes.

    The blob detection finds blobs of different sizes by applying a
    Laplacian of Gaussian with increasing sigmas followed by h-maxima
    filtering. The blob detection starts with the provided sigmas and
    then doubles them for `n_scale_levels`.

    Parameters
    ----------
    img :
        Image containing spot signal.
    axial_sigma :
        Z extension of the spots.
    lateral_sigma :
        YX extension of the spots.
    h :
        h-maxima threshold.
    scale_factors :
        List of scaling factors to apply to the sigmas.
    overlap :
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    background_img :
        Estimated background image. This is subtracted before the
        blob detection.
    mask :
        Foreground mask to restrict the blob detection.

    Returns
    -------
    Detected spots.
    """
    image = (
        img_as_float32(img) - img_as_float32(background_img)
        if background_img is not None
        else img_as_float32(img)
    )

    rescale_factor = estimate_log_rescale_factor(
        (axial_sigma, lateral_sigma, lateral_sigma),
    )

    sigmas = [
        (axial_sigma * f, lateral_sigma * f, lateral_sigma * f) for f in scale_factors
    ]

    scale_cube = np.empty((*image.shape, len(sigmas)), dtype=np.uint8)

    h_ = img_as_float32(np.array(h, dtype=img.dtype))
    scale_norm = np.mean([axial_sigma, lateral_sigma, lateral_sigma])
    for i, sigma in enumerate(sigmas):
        log_img = (
            -gaussian_laplace(image, sigma=sigma)
            * rescale_factor
            * (np.mean(sigma) / scale_norm) ** 2
        )
        h_detections = h_maxima(log_img, h=h_, footprint=ball(1))
        if mask is not None:
            h_detections = h_detections * mask
        scale_cube[..., i] = h_detections

    maxima = peak_local_max(
        scale_cube,
        threshold_abs=0.1,
        exclude_border=False,
        footprint=np.ones((3,) * scale_cube.ndim),
    )

    # Convert local_maxima to float64
    lm = maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = np.array(sigmas)[maxima[:, -1]]

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]

    if len(lm) == 0:
        return np.empty((0, 6))
    else:
        return _prune_blobs(np.array(lm), overlap=overlap, sigma_dim=sigma_dim)
