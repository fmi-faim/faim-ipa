import numpy as np
from numpy._typing import ArrayLike
from scipy.ndimage import gaussian_laplace
from skimage.morphology import h_maxima, local_maxima
from skimage.util import img_as_float32


def compute_axial_sigma(wavelength: float, NA: float, axial_spacing: float):
    """
    Sigma which produces a Gaussian with the same full width
    half maximum as the theoretical Rayleigh Criterion.

    R = 0.61 * lambda / NA

    Parameters
    ----------
    wavelength :
        Emission wavelength
    NA :
        Numerical Aperture
    axial_spacing :
        Spacing of z planes

    Returns
    -------
    theoretical sigma
    """
    return 0.61 * wavelength / NA / (2 * np.sqrt(2 * np.log(2))) / axial_spacing


def compute_lateral_sigma(wavelength: float, NA: float, lateral_spacing: float):
    """
    Sigma which produces a Gaussian with the same full width
    half maximum as the theoretical resolution limit in Z described by E. Abbe.

    d = 2 * lambda / (NA^2)

    Parameters
    ----------
    wavelength :
        Emission wavelength
    NA :
        Numerical Aperture
    lateral_spacing :
        Spacing of z planes

    Returns
    -------
    theoretical sigma
    """
    return 2 * wavelength / (NA**2) / (2 * np.sqrt(2 * np.log(2))) / lateral_spacing


def log_detection(roi_img, lateral_sigma, axial_sigma):
    log_img = -gaussian_laplace(
        input=img_as_float32(roi_img), sigma=(axial_sigma, lateral_sigma, lateral_sigma)
    )
    log_detections = local_maxima(log_img, connectivity=0)
    return log_detections


def detection(
    image: ArrayLike,
    mask: ArrayLike,
    wavelength: int,
    numerical_aperture: float,
    spacing: tuple[float, float, float],
    intensity_threshold: int,
) -> ArrayLike:
    """
    Detect bright, diffraction limited spots.

    The Laplacian of Gaussian is used to detect diffraction limited spots,
    where the spot size is computed from the emission `wavelength`, `NA` and
    pixel `spacing`. This results in an over-detection of spots and only the
    ones with an intensity larger than `intensity_threshold` relative to
    their immediate neighborhood are kept.

    Parameters
    ----------
    image :
        Raw image data.
    mask :
        Foreground mask.
    wavelength :
        Emission wavelength of the spot signal.
    numerical_aperture :
        Numerical aperture.
    spacing :
        Z, Y, X spacing of the data.
    intensity_threshold :
        Minimum spot intensity relative to the immediate background.

    Returns
    -------
    Detected spots
    """
    axial_sigma = compute_axial_sigma(wavelength, numerical_aperture, spacing[0])
    lateral_sigma = compute_lateral_sigma(wavelength, numerical_aperture, spacing[1])

    log_detections = (
        log_detection(
            roi_img=image, lateral_sigma=lateral_sigma, axial_sigma=axial_sigma
        )
        * mask
    )
    h_detections = (
        h_maxima(image, h=intensity_threshold, footprint=np.ones((1, 1, 1))) * mask
    )

    return np.array(np.where(np.logical_and(log_detections, h_detections))).T
