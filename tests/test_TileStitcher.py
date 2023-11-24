import numpy as np
import pytest
from numpy.testing import assert_array_equal

from faim_hcs.stitching import BoundingBox, DaskTileStitcher, Tile, stitching_utils


@pytest.fixture
def tiles():
    return [
        Tile(path="path1", shape=(10, 10), position=(0, 0, 0)),
        Tile(path="path2", shape=(10, 10), position=(0, 0, 10)),
        Tile(path="path3", shape=(10, 10), position=(0, 10, 0)),
        Tile(path="path4", shape=(10, 10), position=(0, 10, 10)),
    ]


def test_get_block_coordinates():
    bbox = BoundingBox(z_start=0, z_end=1, y_start=10, y_end=21, x_start=20, x_end=41)
    corners = bbox.get_corner_points()
    expected = [
        (0, 10, 20),
        (0, 10, 40),
        (0, 20, 40),
        (0, 20, 20),
        (0, 10, 20),
        (0, 10, 40),
        (0, 20, 40),
        (0, 20, 20),
    ]

    assert corners == expected


def test_point_inside_block():
    bbox = BoundingBox(z_start=0, z_end=2, y_start=0, y_end=2, x_start=0, x_end=2)
    assert bbox.contains((1, 1, 1))
    assert bbox.contains((0, 1, 1))
    assert not bbox.contains((0, 2, 1))


def test_overlapping():
    bbox_a = BoundingBox(z_start=0, z_end=1, y_start=0, y_end=10, x_start=0, x_end=10)
    bbox_b = BoundingBox(z_start=0, z_end=1, y_start=0, y_end=10, x_start=0, x_end=10)
    bbox_c = BoundingBox(z_start=0, z_end=1, y_start=10, y_end=20, x_start=0, x_end=10)
    assert bbox_a.overlaps(bbox_b)
    assert bbox_b.overlaps(bbox_a)
    assert not bbox_a.overlaps(bbox_c)
    assert not bbox_c.overlaps(bbox_a)


def test_create_tile_map(tiles):
    ts = DaskTileStitcher(
        tiles=tiles,
        yx_chunk_shape=(10, 10),
    )

    assert ts._shape == (1, 20, 20)
    assert len(ts._block_to_tile_map) == 4
    assert ts._block_to_tile_map[(0, 0, 0)] == [tiles[0]]
    assert ts._block_to_tile_map[(0, 0, 1)] == [tiles[1]]
    assert ts._block_to_tile_map[(0, 1, 0)] == [tiles[2]]
    assert ts._block_to_tile_map[(0, 1, 1)] == [tiles[3]]


def test_create_tile_map_large_chunks(tiles):
    ts = DaskTileStitcher(
        tiles=tiles,
        yx_chunk_shape=(15, 15),
    )

    assert ts._shape == (1, 20, 20)
    assert len(ts._block_to_tile_map) == 4
    assert ts._block_to_tile_map[(0, 0, 0)] == [tiles[0], tiles[1], tiles[2], tiles[3]]
    assert ts._block_to_tile_map[(0, 0, 1)] == [tiles[1], tiles[3]]
    assert ts._block_to_tile_map[(0, 1, 0)] == [tiles[2], tiles[3]]
    assert ts._block_to_tile_map[(0, 1, 1)] == [tiles[3]]


@pytest.fixture
def overlapping_tiles():
    tiles = [
        Tile(path="path1", shape=(10, 10), position=(0, 0, 0)),
        Tile(path="path2", shape=(10, 10), position=(0, 0, 5)),
        Tile(path="path3", shape=(10, 10), position=(0, 5, 0)),
        Tile(path="path4", shape=(10, 10), position=(0, 5, 5)),
    ]
    for i, tile in enumerate(tiles):
        tile.i = i

        def loader(self=tile):
            return np.ones(self.shape) * self.i

        tile.load_data = loader

    return tiles


def test_stitch(overlapping_tiles):
    ts = DaskTileStitcher(tiles=overlapping_tiles, yx_chunk_shape=(10, 10))
    stitched = ts.get_stitched_image(
        transform_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_sum,
    )
    assert_array_equal(stitched[:, :5, :5], np.ones((1, 5, 5)) * 0)
    assert_array_equal(stitched[:, :5, 5:15], np.ones((1, 5, 10)) * 1)
    assert_array_equal(stitched[:, 5:15, :5], np.ones((1, 10, 5)) * 2)
    assert_array_equal(stitched[:, 5:10, 5:10], np.ones((1, 5, 5)) * 6)
    assert_array_equal(stitched[:, 10:15, 5:10], np.ones((1, 5, 5)) * 5)
    assert_array_equal(stitched[:, 5:10, 10:15], np.ones((1, 5, 5)) * 4)
    assert_array_equal(stitched[:, 10:15, 10:15], np.ones((1, 5, 5)) * 3)
    assert stitched.shape == (1, 15, 15)
