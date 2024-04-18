import numpy as np
import pytest

from faim_ipa.MetaSeriesUtils_dask import fuse_fw, fuse_random_gradient, fuse_rev


def tiles():
    fake_tiles = [
        np.ones((8, 8), dtype=np.uint8),
        np.ones((8, 8), dtype=np.uint8) + 1,
        np.ones((8, 8), dtype=np.uint8) + 2,
    ]
    return np.stack(fake_tiles, axis=0)


def positions():
    return np.array([[0, 0], [0, 4], [3, 1]])


@pytest.mark.parametrize(
    "tiles,positions",
    [
        (tiles(), positions()),
    ],
)
def test_fuse_rev(tiles, positions):
    fused_result = fuse_rev(tiles=tiles, positions=positions)
    # should be the same for all fuse-functions:
    assert fused_result.shape == (11, 12)
    assert fused_result[2, 3] == 1
    assert fused_result[2, 8] == 2
    assert fused_result[8, 3] == 3
    assert fused_result[8, 9] == 0
    # depends on fuse-functions:
    assert fused_result[3, 4] == 1
    assert fused_result[3, 7] == 1
    assert fused_result[7, 4] == 1
    assert fused_result[7, 7] == 1


@pytest.mark.parametrize(
    "tiles,positions",
    [
        (tiles(), positions()),
    ],
)
def test_fuse_fw(tiles, positions):
    fused_result = fuse_fw(tiles=tiles, positions=positions)
    # should be the same for all fuse-functions:
    assert fused_result.shape == (11, 12)
    assert fused_result[2, 3] == 1
    assert fused_result[2, 8] == 2
    assert fused_result[8, 3] == 3
    assert fused_result[8, 9] == 0
    # depends on fuse-functions:
    assert fused_result[3, 4] == 3
    assert fused_result[3, 7] == 3
    assert fused_result[7, 4] == 3
    assert fused_result[7, 7] == 3


@pytest.mark.parametrize(
    "tiles,positions",
    [
        (tiles(), positions()),
    ],
)
def test_fuse_random_gradient(tiles, positions):
    fused_result = fuse_random_gradient(tiles=tiles, positions=positions)
    # should be the same for all fuse-functions:
    assert fused_result.shape == (11, 12)
    assert fused_result[2, 3] == 1
    assert fused_result[2, 8] == 2
    assert fused_result[8, 3] == 3
    assert fused_result[8, 9] == 0
    # depends on fuse-functions:
    assert fused_result[3, 4] == 1
    assert fused_result[3, 7] == 1
    assert fused_result[7, 4] == 3
    assert fused_result[7, 7] == 3
