from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from faim_ipa.eicm.estimator.gaussian2D_fit import (
    compute_fitted_matrix,
    fit_gaussian_2d,
    get_coords,
)


def gaussian_2d(
    height: float = 1,
    mu_y: float = 0,
    mu_x: float = 0,
    sigma_y: float = 2,
    sigma_x: float = 2,
    offset: float = 0,
):
    """
    Return a parametrized 3D Gaussian function.

    Parameters:
        height: float
            Distance between the lowest and peak value of the Gaussian.
        mu_y: float
            Expected value of the Gaussian in Y dimension.
        mu_x: float
            Expected value of the Gaussian in X dimension.
        sigma_y: float
            Width of the Gaussian in Y dimension.
        sigma_x: float
            Width of the Gaussian in X dimension.
        offset: float
            Shifts the Gaussian `up` or `down` i.e. the background signal.
    """
    return lambda y, x: offset + height * np.exp(
        -(((y - mu_y) ** 2 / (2 * sigma_y**2)) + ((x - mu_x) ** 2 / (2 * sigma_x**2)))
    )


class Gaussian2DFit(TestCase):
    def setUp(self) -> None:
        Y, X = np.meshgrid(range(512), range(512), indexing="ij")
        self.blob = gaussian_2d(100, 245, 257, 40, 35, 128)(Y, X)

    def test_get_coords(self):
        coords = get_coords(self.blob)
        assert coords.shape == (self.blob.shape[0] * self.blob.shape[1], 2)

    def test_ellipsoid_fit(self):
        coords = get_coords(self.blob)

        popt, pcov = fit_gaussian_2d(self.blob, coords)

        matrix = compute_fitted_matrix(coords, popt, self.blob.shape)

        assert_array_almost_equal(self.blob, matrix, decimal=6)
