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
