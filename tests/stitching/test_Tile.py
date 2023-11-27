from faim_hcs.stitching import Tile


def test_fields():
    tile = Tile(
        path="path",
        shape=(10, 10),
        position=(0, 0, 0),
    )
    assert tile.path == "path"
    assert tile.shape == (10, 10)
    assert tile.position == (0, 0, 0)
