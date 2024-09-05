import numpy as np
from numpy._typing import NDArray

from faim_ipa.stitching.tile import Tile, TilePosition


class StackedTile:
    def __init__(
        self,
        tiles: list[Tile],
        dtype: np.dtype,
        position: TilePosition,
    ):
        self.tiles = tiles
        self.shape = (len(tiles),) + tiles[0].shape
        self.position = position
        self._dtype = dtype

    def load_data(self):
        data = np.zeros(self.shape, dtype=self._dtype)
        for i, tile in enumerate(self.tiles):
            if tile:
                data[i] = tile.load_data()

        return data

    def load_data_mask(self) -> NDArray:
        data = np.zeros(self.shape, dtype=bool)
        for i, tile in enumerate(self.tiles):
            if tile:
                data[i] = True

        return data
