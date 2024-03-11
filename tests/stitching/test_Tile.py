import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tifffile import imwrite

from faim_hcs.stitching import Tile
from faim_hcs.stitching.Tile import TilePosition


def test_fields():
    tile = Tile(
        path="path",
        shape=(10, 10),
        position=TilePosition(time=0, channel=0, z=0, y=0, x=0),
    )
    assert tile.path == "path"
    assert tile.shape == (10, 10)
    assert tile.position.time == 0
    assert tile.position.channel == 0
    assert tile.position.z == 0
    assert tile.position.y == 0
    assert tile.position.x == 0

    assert (
        str(tile)
        == "Tile(path='path', shape=(10, 10), position=TilePosition(time=0, channel=0, z=0, y=0, x=0))"
    )


@pytest.fixture
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("tile")


@pytest.fixture
def test_img(tmp_dir):
    img = np.random.rand(10, 10) * 255
    img = img.astype(np.uint8)
    imwrite(tmp_dir / "test_img.tif", img)
    return tmp_dir / "test_img.tif", img


@pytest.fixture
def bgcm(tmp_dir):
    img = np.ones((10, 10)) * 50
    img = img.astype(np.uint8)
    imwrite(tmp_dir / "bgcm.tif", img)
    return tmp_dir / "bgcm.tif", img


@pytest.fixture
def icm(tmp_dir):
    img = np.ones((10, 10)) * 2
    img = img.astype(np.uint8)
    imwrite(tmp_dir / "icm.tif", img)
    return tmp_dir / "icm.tif", img


def test_load_data(test_img, bgcm, icm):
    tile = Tile(
        path=test_img[0],
        shape=(10, 10),
        position=TilePosition(time=0, channel=0, z=0, y=0, x=0),
    )

    assert_array_equal(tile.load_data(), test_img[1])

    tile = Tile(
        path=test_img[0],
        shape=(10, 10),
        position=TilePosition(time=0, channel=0, z=0, y=0, x=0),
        background_correction_matrix_path=bgcm[0],
    )

    assert_array_equal(tile.load_data(), test_img[1] - bgcm[1])

    tile = Tile(
        path=test_img[0],
        shape=(10, 10),
        position=TilePosition(time=0, channel=0, z=0, y=0, x=0),
        illumination_correction_matrix_path=icm[0],
    )

    assert_array_equal(tile.load_data(), (test_img[1] / icm[1]).astype(np.uint8))


def test_get_position():
    tile = Tile(
        path="path",
        shape=(10, 10),
        position=TilePosition(time=10, channel=20, z=-1, y=2, x=7),
    )
    assert tile.get_position() == (10, 20, -1, 2, 7)
    assert tile.get_zyx_position() == (-1, 2, 7)
    assert tile.get_yx_position() == (2, 7)


def test_Tile_data_mask():
    tile = Tile(
        path="path",
        shape=(10, 10),
        position=TilePosition(time=10, channel=20, z=-1, y=2, x=7),
    )
    mask = tile.load_data_mask()
    assert mask.dtype == bool
    assert mask.shape == (10, 10)
    assert mask.all()


def test_CVStackedTile_data_mask():
    from faim_hcs.hcs.cellvoyager.StackedTile import StackedTile

    tile = StackedTile(
        paths=["path1", None, "path3"],
        shape=(3, 10, 10),
        dtype=np.uint8,
        position=TilePosition(time=10, channel=20, z=-1, y=2, x=7),
    )
    mask = tile.load_data_mask()
    assert mask.dtype == bool
    assert mask.shape == (3, 10, 10)
    assert mask[0].all()
    assert not mask[1].any()
    assert mask[2].all()


def test_VisiViewStackedTile_data_mask():
    from faim_hcs.visiview.StackedTile import StackedTile

    tile = StackedTile(
        path="path1",
        shape=(3, 10, 10),
        position=TilePosition(time=10, channel=20, z=-1, y=2, x=7),
    )
    mask = tile.load_data_mask()
    assert mask.dtype == bool
    assert mask.shape == (3, 10, 10)
    assert mask.all()
