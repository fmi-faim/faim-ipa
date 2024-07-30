from __future__ import annotations

import numpy as np


def mean_cast_to(target_dtype):
    """
    Wrap np.mean to cast the result to a given dtype.
    """

    def _mean(
        a,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        *,
        where=np._NoValue,
    ):
        return np.mean(
            a=a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
        ).astype(target_dtype)

    return _mean
