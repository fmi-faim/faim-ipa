import numpy as np
from scipy.optimize import curve_fit
from skimage.measure import centroid


def ellipsoid_2D(coords, A, B, mu_x, mu_y, cxx, cxy, cyy):
    """
    Evaluates the given ellipsoid parameters at the provided coordinates.

    :param coords: at which the evaluation takes place
    :param A: amplitude
    :param B: offset
    :param mu_x: x location of the centroid
    :param mu_y: y location of the centroid
    :param cxx: covariance matrix term at 00
    :param cxy: covariance matrix term at 01
    :param cyy: covariance matrix term at 11
    :return:
    """
    inv = np.linalg.inv(np.array([[cxx, cxy], [cxy, cyy]]) + np.identity(2) * 1e-8)

    return (
        A
        * np.exp(
            -0.5
            * (
                inv[0, 0] * (coords[:, 1] - mu_x) ** 2
                + 2 * inv[0, 1] * (coords[:, 1] - mu_x) * (coords[:, 0] - mu_y)
                + inv[1, 1] * (coords[:, 0] - mu_y) ** 2
            )
        )
        + B
    )


def get_cov_matrix(img):
    """
    Compute covariance matrix of the provided image.

    :param img:
    :return: covariance matrix
    """

    def cov(x, y, i):
        return np.sum(x * y * i) / np.sum(i)

    y, x = np.meshgrid(
        np.arange(img.shape[0]),
        np.arange(img.shape[1]),
        indexing="ij",
    )
    cen = centroid(img)
    y = y.ravel() - cen[0]
    x = x.ravel() - cen[1]

    cxx = cov(x, x, img.ravel())
    cyy = cov(y, y, img.ravel())
    cxy = cov(x, y, img.ravel())

    return np.array([[cxx, cxy], [cxy, cyy]])


def get_estimates(data):
    """
    Estimate parameters of 2D Gaussian fit to the data.
    :param data:
    :return: (amplitude, offset, centroid-x, centroid-y, cxx, cxy, cyy)
    """
    max_ = data.max()
    mean_ = data.mean()
    cy, cx = centroid(data)
    cov = get_cov_matrix(data)
    return [
        max_ - mean_,
        mean_,
        cx,
        cy,
        cov[0, 0],
        cov[0, 1],
        cov[1, 1],
    ]


def get_coords(data):
    """
    Compute coordinate grid of provided 2D data.
    :param data:
    :return: coordinates
    """
    yy = np.arange(data.shape[0])
    xx = np.arange(data.shape[1])
    y, x = np.meshgrid(yy, xx, indexing="ij")
    return np.stack([y.ravel(), x.ravel()], -1)


def fit_gaussian_2d(data, coords):
    """
    Fit 2D ellipsoid to data.
    :param data: intensities
    :param coords: pixel coordinates
    :return: popt, pcov (see scipy.optimize.curve_fit)
    """
    return curve_fit(
        ellipsoid_2D,
        coords,
        data.ravel(),
        p0=get_estimates(data),
    )


def compute_fitted_matrix(coords, ellipsoid_parameters, shape):
    """
    Compute 2D ellipsoid for provided set of parameters
    :param coords: pixel coordinates
    :param ellipsoid_parameters:
    :param shape: of the final output
    :return: matrix
    """
    return ellipsoid_2D(coords, *ellipsoid_parameters).reshape(shape)
