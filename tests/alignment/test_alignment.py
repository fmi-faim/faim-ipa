import pytest

from faim_ipa.hcs.source import FileSource
from faim_ipa.stitching.tile import Tile, TilePosition


@pytest.fixture
def tiles() -> list[Tile]:
    return [
        Tile(
            source=FileSource("dir"),
            path="path",
            shape=(512, 512),
            position=TilePosition(time=0, channel=0, z=231, y=4829, x=20128),
        ),
        Tile(
            source=FileSource("dir"),
            path="path",
            shape=(512, 512),
            position=TilePosition(time=0, channel=0, z=231, y=4829 + 512, x=20128),
        ),
        Tile(
            source=FileSource("dir"),
            path="path",
            shape=(512, 512),
            position=TilePosition(
                time=0, channel=0, z=231, y=4829 + 2 * 512 + 1, x=20128
            ),
        ),
        Tile(
            source=FileSource("dir"),
            path="path",
            shape=(512, 512),
            position=TilePosition(time=0, channel=0, z=231, y=4829, x=20128 + 512),
        ),
        Tile(
            source=FileSource("dir"),
            path="path",
            shape=(512, 512),
            position=TilePosition(
                time=0, channel=0, z=231, y=4829 + 512, x=20128 + 512 - 1
            ),
        ),
    ]


def test_stage_alignment(tiles):
    from faim_ipa.alignment import StageAlignment

    alignment = StageAlignment(tiles)
    aligned_tiles = alignment.get_tiles()
    assert len(aligned_tiles) == len(tiles)
    for tile in aligned_tiles:
        assert tile.shape == (512, 512)
        assert tile.position.time == 0
        assert tile.position.channel == 0
        assert tile.position.z == 0
        assert tile.position.y in [0, 512, 1025]
        assert tile.position.x in [0, 511, 512, 522]


def test_grid_alignment(tiles):
    from faim_ipa.alignment import GridAlignment

    alignment = GridAlignment(tiles)
    aligned_tiles = alignment.get_tiles()
    assert len(aligned_tiles) == len(tiles)
    for tile in aligned_tiles:
        assert tile.shape == (512, 512)
        assert tile.position.time == 0
        assert tile.position.channel == 0
        assert tile.position.z == 0
        assert tile.position.y in [0, 512, 1024]
        assert tile.position.x in [0, 512]


def test_abstract_alignment(tiles):
    from faim_ipa.alignment.alignment import AbstractAlignment

    with pytest.raises(TypeError):
        AbstractAlignment(tiles)
