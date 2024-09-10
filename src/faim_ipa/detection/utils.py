import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace


def compute_axial_sigma(
    wavelength: float, NA: float, axial_spacing: float  # noqa: N803
):
    """
    Sigma which produces a Gaussian with the same full width
    half maximum as described by Abbe's diffraction formula for axial resolution.

    R = 2 * lambda / (NA**2)

    Parameters
    ----------
    wavelength :
        Emission wavelength
    NA :
        Numerical Aperture
    axial_spacing :
        Spacing of z planes in nanometers

    Returns
    -------
    theoretical sigma in pixels
    """
    return 2 * wavelength / (NA**2) / (2 * np.sqrt(2 * np.log(2))) / axial_spacing


def compute_lateral_sigma(
    wavelength: float, NA: float, lateral_spacing: float  # noqa: N803
):
    """
    Sigma which produces a Gaussian with the same full width
    half maximum as the theoretical resolution limit in Y/X described by E. Abbe.

    d = lambda / (2*NA)

    Parameters
    ----------
    wavelength :
        Emission wavelength
    NA :
        Numerical Aperture
    lateral_spacing :
        Pixel size in YX.

    Returns
    -------
    theoretical sigma in pixels
    """
    return wavelength / (2 * NA) / (2 * np.sqrt(2 * np.log(2))) / lateral_spacing


def estimate_log_rescale_factor(sigmas: tuple[float, ...]) -> float:
    """
    Estimate the rescale factor for a LoG filter response, such that
    the LoG filter response intensities are equal to the input image
    intensities for spots of size equal to a Gaussian with sigmas
    (axial_sigma, lateral_sigma, lateral_sigma).

    Note: Arbitrary number of sigmas is possible.

    Parameters
    ----------
    sigmas :
        Tuple of sigmas used in LoG detection.

    Returns
    -------
    rescale_factor
    """
    extend = int(max(sigmas) * 7)
    img = np.zeros((extend,) * len(sigmas), dtype=np.float32)
    img[(extend // 2,) * len(sigmas)] = 1
    img = gaussian_filter(img, sigmas)
    img = img / img.max()
    img_log = -gaussian_laplace(input=img, sigma=sigmas)
    return 1 / img_log.max()
