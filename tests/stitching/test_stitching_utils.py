from dataclasses import dataclass

import numpy as np
import pytest
from faim_ipa.stitching import Tile
from faim_ipa.stitching.stitching_utils import (
    assemble_chunk,
    fuse_linear,
    fuse_linear_random,
    fuse_mean,
    fuse_overlay_bwd,
    fuse_overlay_fwd,
    fuse_sum,
    get_distance_mask,
    shift_to_origin,
    shift_yx,
    translate_tiles_2d,
)
from faim_ipa.stitching.Tile import TilePosition
from numpy.testing import assert_array_equal
from scipy.ndimage import distance_transform_cdt


@pytest.fixture
def tiles():
    t1 = np.zeros((1, 10, 20), dtype=np.uint16)
    t1[:, :, :15] = 1
    t2 = np.zeros((1, 10, 20), dtype=np.uint16)
    t2[:, :, 5:] = 4
    return np.array([t1, t2])


@pytest.fixture
def distance_masks():
    m1 = np.zeros((1, 10, 20), dtype=bool)
    m1[:, :, :15] = True
    t1 = distance_transform_cdt(m1, metric="taxicab")
    m2 = np.zeros((1, 10, 20), dtype=bool)
    m2[:, :, 5:] = True
    t2 = distance_transform_cdt(m2, metric="taxicab")
    return np.array([t1, t2])


def test_fuse_mean(tiles, distance_masks):
    fused_result = fuse_mean(warped_tiles=tiles, warped_distance_masks=distance_masks)
    assert fused_result.shape == (1, 10, 20)
    assert fused_result.dtype == np.uint16
    assert_array_equal(fused_result[:, :, :5], 1)
    assert_array_equal(fused_result[:, :, 5:15], 2)
    assert_array_equal(fused_result[:, :, 15:], 4)


def test_fuse_sum(tiles, distance_masks):
    fused_result = fuse_sum(warped_tiles=tiles, warped_distance_masks=distance_masks)
    assert fused_result.shape == (1, 10, 20)
    assert fused_result.dtype == np.uint16
    assert_array_equal(fused_result[:, :, :5], 1)
    assert_array_equal(fused_result[:, :, 5:15], 5)
    assert_array_equal(fused_result[:, :, 15:], 4)


def test_fuse_linear(tiles, distance_masks):
    fused_result = fuse_linear(warped_tiles=tiles, warped_distance_masks=distance_masks)
    assert fused_result.shape == (1, 10, 20)
    assert fused_result.dtype == np.uint16
    assert_array_equal(fused_result[:, :, :5], 1)
    assert_array_equal(fused_result[:, :, 5], int(1 * 10 / 11 + 4 * 1 / 11))
    assert_array_equal(fused_result[:, :, 6], int(1 * 9 / 11 + 4 * 2 / 11))
    assert_array_equal(fused_result[:, :, 7], int(1 * 8 / 11 + 4 * 3 / 11))
    assert_array_equal(fused_result[:, :, 8], int(1 * 7 / 11 + 4 * 4 / 11))
    assert_array_equal(fused_result[:, :, 9], int(1 * 6 / 11 + 4 * 5 / 11))
    assert_array_equal(fused_result[:, :, 10], int(1 * 5 / 11 + 4 * 6 / 11))
    assert_array_equal(fused_result[:, :, 11], int(1 * 4 / 11 + 4 * 7 / 11))
    assert_array_equal(fused_result[:, :, 12], int(1 * 3 / 11 + 4 * 8 / 11))
    assert_array_equal(fused_result[:, :, 13], int(1 * 2 / 11 + 4 * 9 / 11))
    assert_array_equal(fused_result[:, :, 14], int(1 * 1 / 11 + 4 * 10 / 11))
    assert_array_equal(fused_result[:, :, 15:], 4)

    fused_result = fuse_linear(
        warped_tiles=tiles[:1], warped_distance_masks=distance_masks[:1]
    )
    assert fused_result.shape == (1, 10, 20)
    assert fused_result.dtype == np.uint16
    assert_array_equal(fused_result, tiles[0])


def test_fuse_linear_random(tiles, distance_masks):
    fused_result = fuse_linear_random(
        warped_tiles=tiles, warped_distance_masks=distance_masks
    )
    assert fused_result.shape == (1, 10, 20)
    assert fused_result.dtype == np.uint16

    expected_result = np.array(
        [
            [
                [1, 1, 1, 1, 1, 1, 1, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4, 4, 4, 1, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 1, 1, 4, 4, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4],
            ]
        ],
        dtype=np.uint16,
    )
    assert_array_equal(fused_result, expected_result)

    fused_result = fuse_linear(
        warped_tiles=tiles[:1], warped_distance_masks=distance_masks[:1]
    )
    assert fused_result.shape == (1, 10, 20)
    assert fused_result.dtype == np.uint16
    assert_array_equal(fused_result, tiles[0])


def test_fuse_overlay_fwd(tiles, distance_masks):
    fused_result = fuse_overlay_fwd(
        warped_tiles=tiles, warped_distance_masks=distance_masks
    )
    assert fused_result.shape == (1, 10, 20)
    assert fused_result.dtype == np.uint16
    assert_array_equal(fused_result[:, :, :5], 1)
    assert_array_equal(fused_result[:, :, 5:], 4)


def test_fuse_overlay_bwd(tiles, distance_masks):
    fused_result = fuse_overlay_bwd(
        warped_tiles=tiles, warped_distance_masks=distance_masks
    )
    assert fused_result.shape == (1, 10, 20)
    assert fused_result.dtype == np.uint16
    assert_array_equal(fused_result[:, :, :15], 1)
    assert_array_equal(fused_result[:, :, 15:], 4)


@dataclass
class DummyTile:
    def __init__(self, yx_position, data):
        self._yx_position = yx_position
        self._data = data
        self.shape = data.shape

    def get_zyx_position(self):
        return (0,) + self._yx_position

    def load_data(self):
        return self._data

    def load_data_mask(self):
        return np.ones(self._data.shape, dtype=bool)


def test_assemble_chunk(tiles):
    tile_map = {
        (0, 0, 0, 0, 0): [
            DummyTile(yx_position=(0, 0), data=tiles[0][..., :15]),
            DummyTile(yx_position=(0, 5), data=tiles[1][..., 5:]),
        ],
        (1, 0, 0, 0, 0): [],
    }

    block_info = {
        None: {
            "array-location": [(0, 1), (0, 1), (0, 1), (0, 10), (0, 20)],
            "chunk-location": (0, 0, 0, 0, 0),
            "chunk-shape": (1, 1, 1, 10, 20),
            "dtype": "uint16",
            "num-chunks": (1, 1, 1, 1, 1),
            "shape": (1, 1, 1, 10, 20),
        }
    }

    stitched_img = assemble_chunk(
        block_info=block_info,
        tile_map=tile_map,
        warp_func=translate_tiles_2d,
        fuse_func=fuse_sum,
        dtype=np.uint16,
    )
    assert stitched_img.shape == (1, 1, 1, 10, 20)
    assert_array_equal(stitched_img[0, 0], tiles[0] + tiles[1])

    block_info = {
        None: {
            "array-location": [(0, 1), (0, 1), (0, 1), (0, 10), (0, 20)],
            "chunk-location": (1, 0, 0, 0, 0),
            "chunk-shape": (1, 1, 1, 10, 20),
            "dtype": "uint16",
            "num-chunks": (1, 1, 1, 1, 1),
            "shape": (1, 1, 1, 10, 20),
        }
    }
    stitched_img = assemble_chunk(
        block_info=block_info,
        tile_map=tile_map,
        warp_func=translate_tiles_2d,
        fuse_func=fuse_mean,
        dtype=np.uint16,
    )
    assert stitched_img.shape == (1, 1, 1, 10, 20)
    assert_array_equal(stitched_img[0, 0], np.zeros_like(tiles[0], dtype=np.uint16))

    tile_map = {
        (0, 0, 0, 0, 0): [
            DummyTile(yx_position=(0, 0), data=tiles[0][..., :15]),
        ],
        (1, 0, 0, 0, 0): [],
    }

    block_info = {
        None: {
            "array-location": [(0, 1), (0, 1), (0, 1), (0, 10), (0, 20)],
            "chunk-location": (0, 0, 0, 0, 0),
            "chunk-shape": (1, 1, 1, 10, 20),
            "dtype": "uint16",
            "num-chunks": (1, 1, 1, 1, 1),
            "shape": (1, 1, 1, 10, 20),
        }
    }

    stitched_img = assemble_chunk(
        block_info=block_info,
        tile_map=tile_map,
        warp_func=translate_tiles_2d,
        fuse_func=fuse_sum,
        dtype=np.uint16,
    )
    assert stitched_img.shape == (1, 1, 1, 10, 20)
    assert_array_equal(stitched_img[0, 0], tiles[0])


def test_shift_to_origin():
    result = shift_to_origin(
        [
            Tile(
                path="path",
                shape=(10, 10),
                position=TilePosition(time=20, channel=1, z=10, y=-1, x=2),
            )
        ]
    )

    assert result[0].get_position() == (0, 0, 0, 0, 0)


def test_get_distance_mask():
    expected_distance_mask_2D = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 3, 3, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint16,
    )

    result = get_distance_mask((5, 6))
    assert result.dtype == np.uint16
    assert_array_equal(result, expected_distance_mask_2D)

    result = get_distance_mask((1, 5, 6))
    assert result.dtype == np.uint16
    assert_array_equal(result, expected_distance_mask_2D.reshape((1, 5, 6)))

    result = get_distance_mask((4, 5, 6))
    assert result.dtype == np.uint16
    assert_array_equal(result, np.stack([expected_distance_mask_2D for n in range(4)]))


def test_translate_3d_tiles_2d(tiles):
    tile_map = {
        (0, 0, 0, 0, 0): [
            DummyTile(yx_position=(0, 0), data=tiles[0][..., :15]),
            DummyTile(yx_position=(0, 5), data=tiles[1][..., 5:]),
        ],
        (1, 0, 0, 0, 0): [],
    }

    block_info = {
        None: {
            "array-location": [(0, 1), (0, 1), (0, 1), (0, 10), (0, 20)],
            "chunk-location": (0, 0, 0, 0, 0),
            "chunk-shape": (1, 1, 1, 10, 20),
            "dtype": "uint16",
            "num-chunks": (1, 1, 1, 1, 1),
            "shape": (1, 1, 1, 10, 20),
        }
    }

    warped_tiles, warped_masks = translate_tiles_2d(
        block_info=block_info,
        chunk_shape=(1, 10, 20),
        tiles=tile_map[(0, 0, 0, 0, 0)],
    )

    assert warped_tiles.shape == (2, 1, 10, 20)
    assert warped_masks.shape == (2, 1, 10, 20)

    assert warped_tiles.dtype == np.uint16
    assert warped_masks.dtype == np.uint16

    assert_array_equal(warped_tiles[0], tiles[0])
    assert_array_equal(warped_tiles[1], tiles[1])


def test_translate_2d_tiles_2d(tiles):
    tile_map = {
        (0, 0, 0, 0, 0): [
            DummyTile(yx_position=(0, 0), data=tiles[0][0, ..., :15]),
            DummyTile(yx_position=(0, 5), data=tiles[1][0, ..., 5:]),
        ],
        (1, 0, 0, 0, 0): [],
    }

    block_info = {
        None: {
            "array-location": [(0, 1), (0, 1), (0, 1), (0, 10), (0, 20)],
            "chunk-location": (0, 0, 0, 0, 0),
            "chunk-shape": (1, 1, 1, 10, 20),
            "dtype": "uint16",
            "num-chunks": (1, 1, 1, 1, 1),
            "shape": (1, 1, 1, 10, 20),
        }
    }

    warped_tiles, warped_masks = translate_tiles_2d(
        block_info=block_info,
        chunk_shape=(1, 10, 20),
        tiles=tile_map[(0, 0, 0, 0, 0)],
    )

    assert warped_tiles.shape == (2, 1, 10, 20)
    assert warped_masks.shape == (2, 1, 10, 20)

    assert warped_tiles.dtype == np.uint16
    assert warped_masks.dtype == np.uint16

    assert_array_equal(warped_tiles[0], tiles[0])
    assert_array_equal(warped_tiles[1], tiles[1])


def test_translate_2d_tiles_2d_data_mask(tiles):
    tile_map = {
        (0, 0, 0, 0, 0): [
            DummyTile(yx_position=(0, 0), data=tiles[0][0, ..., :15]),
            DummyTile(yx_position=(0, 5), data=tiles[1][0, ..., 5:]),
        ],
        (1, 0, 0, 0, 0): [],
    }

    block_info = {
        None: {
            "array-location": [(0, 1), (0, 1), (0, 1), (0, 10), (0, 20)],
            "chunk-location": (0, 0, 0, 0, 0),
            "chunk-shape": (1, 1, 1, 10, 20),
            "dtype": "uint16",
            "num-chunks": (1, 1, 1, 1, 1),
            "shape": (1, 1, 1, 10, 20),
        }
    }

    warped_tiles, warped_masks = translate_tiles_2d(
        block_info=block_info,
        chunk_shape=(1, 10, 20),
        tiles=tile_map[(0, 0, 0, 0, 0)],
        build_acquisition_mask=True,
    )

    assert warped_tiles.shape == (2, 1, 10, 20)
    assert warped_masks.shape == (2, 1, 10, 20)

    assert warped_tiles.dtype == bool
    assert warped_masks.dtype == np.uint16

    assert_array_equal(warped_tiles[0], tiles[0] > 0)
    assert_array_equal(warped_tiles[1], tiles[1] > 0)


def test_translate_2d_shape_mismatch(tiles):
    tile_map = {
        (0, 0, 0, 0, 0): [
            DummyTile(yx_position=(0, 0), data=tiles[0][0, ..., :15]),
            DummyTile(yx_position=(0, 5), data=tiles[1][0, ..., 6:]),
        ],
        (1, 0, 0, 0, 0): [],
    }

    block_info = {
        None: {
            "array-location": [(0, 1), (0, 1), (0, 1), (0, 10), (0, 20)],
            "chunk-location": (0, 0, 0, 0, 0),
            "chunk-shape": (1, 1, 1, 10, 20),
            "dtype": "uint16",
            "num-chunks": (1, 1, 1, 1, 1),
            "shape": (1, 1, 1, 10, 20),
        }
    }
    with pytest.raises(ValueError):
        translate_tiles_2d(
            block_info=block_info,
            chunk_shape=(1, 10, 20),
            tiles=tile_map[(0, 0, 0, 0, 0)],
            build_acquisition_mask=True,
        )


def test_warp_yx():
    tile_data = np.ones((1, 3, 3))
    chunk_shape = (1, 3, 3)
    warped_tile = shift_yx(
        chunk_zyx_origin=np.array((0, 0, 0)),
        tile_data=tile_data,
        tile_origin=np.array((0, 0, 0)),
        chunk_shape=chunk_shape,
    )
    assert_array_equal(warped_tile, tile_data)

    warped_tile = shift_yx(
        chunk_zyx_origin=np.array((0, 0, 0)),
        tile_data=tile_data,
        tile_origin=np.array((0, -1, -1)),
        chunk_shape=chunk_shape,
    )
    expected_warped_tile = np.zeros_like(tile_data)
    expected_warped_tile[:, :-1, :-1] = 1
    assert_array_equal(warped_tile, expected_warped_tile)

    warped_tile = shift_yx(
        chunk_zyx_origin=np.array((0, 0, 0)),
        tile_data=tile_data,
        tile_origin=np.array((0, 1, 1)),
        chunk_shape=chunk_shape,
    )
    expected_warped_tile = np.zeros_like(tile_data)
    expected_warped_tile[:, 1:, 1:] = 1
    assert_array_equal(warped_tile, expected_warped_tile)

    warped_tile = shift_yx(
        chunk_zyx_origin=np.array((0, 0, 0)),
        tile_data=np.ones((1, 1, 1)),
        tile_origin=np.array((0, 1, 1)),
        chunk_shape=chunk_shape,
    )
    expected_warped_tile = np.zeros((1, 3, 3))
    expected_warped_tile[0, 1, 1] = 1
    assert_array_equal(warped_tile, expected_warped_tile)

    warped_tile = shift_yx(
        chunk_zyx_origin=np.array((0, 0, 0)),
        tile_data=np.ones((1, 5, 5)),
        tile_origin=np.array((0, -1, -1)),
        chunk_shape=chunk_shape,
    )
    expected_warped_tile = np.ones((1, 3, 3))
    assert_array_equal(warped_tile, expected_warped_tile)

    warped_tile = shift_yx(
        chunk_zyx_origin=np.array((0, 0, 0)),
        tile_data=np.ones((1, 5, 5)),
        tile_origin=np.array((0, 6, 0)),
        chunk_shape=chunk_shape,
    )
    expected_warped_tile = np.zeros((1, 3, 3))
    assert_array_equal(warped_tile, expected_warped_tile)
