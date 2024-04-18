from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile
from numpy._typing import NDArray

from faim_ipa.stitching import Tile
from faim_ipa.stitching.Tile import TilePosition


class StackedTile(Tile):
    def __init__(
        self,
        path: Union[Path, str],
        shape: tuple[int, int],
        position: TilePosition,
        background_correction_matrix_path: Optional[Union[Path, str]] = None,
        illumination_correction_matrix_path: Optional[Union[Path, str]] = None,
        memmap: bool = True,
    ):
        self._memmap = memmap
        super().__init__(
            path=path,
            shape=shape,
            position=position,
            background_correction_matrix_path=background_correction_matrix_path,
            illumination_correction_matrix_path=illumination_correction_matrix_path,
        )

    def load_data(self) -> NDArray:
        if self._memmap:
            data = tifffile.memmap(self.path, mode="r")
        else:
            data = tifffile.imread(self.path)

        data = self._apply_background_correction(data)
        data = self._apply_illumination_correction(data)
        return data

    def load_data_mask(self) -> NDArray:
        return np.ones(self.shape, dtype=bool)
