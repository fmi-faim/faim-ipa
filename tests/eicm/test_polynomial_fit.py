from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from faim_ipa.eicm.estimator.utils import normalize_matrix
from faim_ipa.eicm.estimator.polynomial_fit import polynomial_fit


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


class PolynomialFit(TestCase):
    def setUp(self) -> None:
        Y, X = np.meshgrid(range(512), range(512), indexing="ij")
        self.blob = gaussian_2d(100, 185, 257, 400, 350, 128)(Y, X)

    def test_polynomial_fit(self):
        fit, poly = polynomial_fit(self.blob, polynomial_degree=5, order=None)

        assert_array_almost_equal(self.blob / fit, np.ones((512, 512)), decimal=4)

        assert (
            poly == "1 + X^1 + X^2 + X^3 + X^4 + X^5 + Y^1 + X^1 * Y^1 + "
            "X^2 * Y^1 + X^3 * Y^1 + X^4 * Y^1 + X^5 * Y^1 + "
            "Y^2 + X^1 * Y^2 + X^2 * Y^2 + X^3 * Y^2 + "
            "X^4 * Y^2 + X^5 * Y^2 + Y^3 + X^1 * Y^3 + "
            "X^2 * Y^3 + X^3 * Y^3 + X^4 * Y^3 + X^5 * Y^3 + "
            "Y^4 + X^1 * Y^4 + X^2 * Y^4 + X^3 * Y^4 + "
            "X^4 * Y^4 + X^5 * Y^4 + Y^5 + X^1 * Y^5 + "
            "X^2 * Y^5 + X^3 * Y^5 + X^4 * Y^5 + X^5 * Y^5"
        )

    def test_poly_fit_str(self):
        fit, poly = polynomial_fit(self.blob, polynomial_degree=2, order=2)
        assert poly == "1 + X^1 + X^2 + Y^1 + X^1 * Y^1 + Y^2"

    def test_normalization(self):
        n_ = normalize_matrix(self.blob)
        assert n_.min() >= 0.0
        assert n_.max() <= 1.0
