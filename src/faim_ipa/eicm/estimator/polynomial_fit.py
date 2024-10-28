import numpy as np


def polynomial_fit(mip, polynomial_degree=3, order=None):
    """
    Fits a polynomial f(x, y) to the provided mip.

    :param mip: to fit
    :param polynomial_degree: highest degree of individual coefficients
    :param order: polynomial order
    :return: fitted matrix, string describing the polynomial
    """
    Y, X = np.meshgrid(
        np.linspace(0, 1, mip.shape[0]), np.linspace(0, 1, mip.shape[1]), indexing="ij"
    )

    coeffs = np.ones((polynomial_degree + 1, polynomial_degree + 1))

    A = np.zeros((coeffs.size, X.size))

    poly_str = ["1"]
    for row_idx, (i, j) in enumerate(np.ndindex(coeffs.shape)):
        if order is not None and i + j > order:
            tmp = np.zeros_like(X)
        else:
            tmp = coeffs[i, j] * X**j * Y**i

            substr = []
            if j > 0:
                substr.append(f"X^{j}")
            if i > 0:
                substr.append(f"Y^{i}")

            if len(substr) > 0:
                poly_str.append(" * ".join(substr))
        A[row_idx] = tmp.ravel()

    poly_str = " + ".join(poly_str)

    x, residuals, rank, s = np.linalg.lstsq(A.T, np.ravel(mip), rcond=None)

    x = x.reshape((polynomial_degree + 1, polynomial_degree + 1))
    fit = np.polynomial.polynomial.polyval2d(Y, X, x).reshape(mip.shape)
    return fit, poly_str
