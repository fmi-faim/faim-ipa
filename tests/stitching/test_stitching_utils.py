import numpy as np
import pytest
from numpy.testing import assert_array_equal

from faim_hcs.stitching.stitching_utils import fuse_mean, fuse_sum


@pytest.fixture
def tiles():
    t1 = np.zeros((10, 20), dtype=np.uint16)
    t1[:, :15] = 1
    t2 = np.zeros((10, 20), dtype=np.uint16)
    t2[:, 5:] = 4
    return np.array([t1, t2])


@pytest.fixture
def masks():
    t1 = np.zeros((10, 20), dtype=bool)
    t1[:, :15] = True
    t2 = np.zeros((10, 20), dtype=bool)
    t2[:, 5:] = True
    return np.array([t1, t2])


def test_fuse_mean(tiles, masks):
    fused_result = fuse_mean(warped_tiles=tiles, warped_masks=masks)
    assert fused_result.shape == (10, 20)
    assert fused_result.dtype == np.uint16
    assert_array_equal(fused_result[:, :5], 1)
    assert_array_equal(fused_result[:, 5:15], 2)
    assert_array_equal(fused_result[:, 15:], 4)


def test_fuse_sum(tiles, masks):
    fused_result = fuse_sum(warped_tiles=tiles, warped_masks=masks)
    assert fused_result.shape == (10, 20)
    assert fused_result.dtype == np.uint16
    assert_array_equal(fused_result[:, :5], 1)
    assert_array_equal(fused_result[:, 5:15], 5)
    assert_array_equal(fused_result[:, 15:], 4)
