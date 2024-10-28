import numpy as np
from scipy.ndimage import gaussian_filter


def create_blurred_illumination_matrix(img, sigma: float = 20):
    """
    Takes an image and blurs it with a Gaussian.

    :param img: The acquired illumination field.
    :param sigma: Gaussian sigma.
    :return: illumination matrix
    """
    matrix = gaussian_filter(input=img.astype(np.float32), sigma=sigma, mode="nearest")
    return matrix
