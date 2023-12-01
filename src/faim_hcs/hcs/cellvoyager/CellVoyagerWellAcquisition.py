from pathlib import Path
from typing import Optional, Union

import pandas as pd

from faim_hcs.hcs.acquisition import TileAlignmentOptions, WellAcquisition
from faim_hcs.stitching import Tile


class CellVoyagerWellAcquisition(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        z_spacing: Optional[float],
        background_correction_matrices: dict[str, Union[Path, str]] = None,
        illumination_correction_matrices: dict[str, Union[Path, str]] = None,
    ):
        self._z_spacing = z_spacing
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _assemble_tiles(self) -> list[Tile]:
        pass

    def get_axes(self) -> list[str]:
        pass

    def get_yx_spacing(self) -> tuple[float, float]:
        pass

    def get_z_spacing(self) -> Optional[float]:
        pass
