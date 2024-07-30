from pathlib import Path

import numpy as np
from numpy._typing import NDArray
from tifffile import imread

from faim_ipa.stitching.tile import Tile, TilePosition


class StackedTile(Tile):
    def __init__(
        self,
        paths: list[Path | str],
        shape: tuple[int, int, int],
        dtype: np.dtype,
        position: TilePosition,
        background_correction_matrix_path: Path | str | None = None,
        illumination_correction_matrix_path: Path | str | None = None,
    ):
        super().__init__(
            path=None,
            shape=(len(paths),) + shape[1:],
            position=position,
            background_correction_matrix_path=background_correction_matrix_path,
            illumination_correction_matrix_path=illumination_correction_matrix_path,
        )
        self._paths = paths
        self._dtype = dtype

    def load_data(self):
        data = np.zeros(self.shape, dtype=self._dtype)
        for i, path in enumerate(self._paths):
            if path:
                plane = imread(path)
                plane = self._apply_background_correction(plane)
                plane = self._apply_illumination_correction(plane)
                data[i] = plane

        return data

    def load_data_mask(self) -> NDArray:
        data = np.zeros(self.shape, dtype=bool)
        for i, path in enumerate(self._paths):
            if path:
                data[i] = True

        return data
