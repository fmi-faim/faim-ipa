import numpy as np
import pytest

from faim_hcs.MetaSeriesUtils_dask import fuse_rev


def tiles():
    fake_tiles = [
        np.ones((10, 10), dtype=np.uint8),
        np.ones((10, 10), dtype=np.uint8) + 1,
    ]
    return np.stack(fake_tiles, axis=0)


def four_tiles():
    fake_tiles = [
        np.ones((10, 10), dtype=np.uint8),
        np.ones((10, 10), dtype=np.uint8) + 1,
    ]
    return np.stack(fake_tiles, axis=0)


def positions():
    return np.array([[0, 0], [0, 5]])


def four_positions():
    return np.array([[0, 0], [0, 5]])


@pytest.mark.parametrize(
    "tiles,positions,expected",
    [
        (tiles(), positions(), ((10, 15), 1, 2)),
        (four_tiles(), four_positions(), ((10, 15), 1, 2)),
    ],
)
def test_fuse_rev(tiles, positions, expected):
    fused_result = fuse_rev(tiles=tiles, positions=positions)

    assert fused_result.shape == expected[0]
    assert fused_result[0, 10] == expected[1]
    assert fused_result[0, 11] == expected[2]
