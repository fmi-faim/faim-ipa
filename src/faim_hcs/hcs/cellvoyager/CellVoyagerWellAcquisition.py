from decimal import Decimal
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from faim_hcs.hcs.acquisition import TileAlignmentOptions, WellAcquisition
from faim_hcs.stitching import Tile
from faim_hcs.stitching.Tile import TilePosition


class CellVoyagerWellAcquisition(WellAcquisition):
    """
    Data structure for a CellVoyager well acquisition.
    """

    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        metadata: pd.DataFrame,
        background_correction_matrices: dict[str, Union[Path, str]] = None,
        illumination_correction_matrices: dict[str, Union[Path, str]] = None,
    ):
        self._metadata = metadata
        self._z_spacing = self._compute_z_spacing(files)
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _compute_z_spacing(self, files: pd.DataFrame) -> Optional[float]:
        if "Z" in files.columns:
            z_steps = np.array(
                sorted([float(i) for i in files.groupby("Z").mean("ZIndex").index])
            )

            precision = -Decimal(str(z_steps[0])).as_tuple().exponent
            z_step = np.round(np.mean(np.diff(z_steps)), decimals=precision)
            return z_step
        else:
            return None

    def _assemble_tiles(self) -> list[Tile]:
        tiles = []
        for i, row in self._files.iterrows():
            file = row["path"]
            time_point = row["TimePoint"]
            channel = row["Ch"]
            if "ZIndex" in row.keys():
                z = row["ZIndex"]
            else:
                z = 0

            ch_metadata = self._metadata[self._metadata["Ch"] == channel].iloc[0]
            shape = (
                int(ch_metadata["VerticalPixels"]),
                int(ch_metadata["HorizontalPixels"]),
            )

            yx_spacing = self.get_yx_spacing()

            bgcm = None
            if self._background_correction_matrices is not None:
                bgcm = self._background_correction_matrices[str(channel)]

            icm = None
            if self._illumincation_correction_matrices is not None:
                icm = self._illumincation_correction_matrices[str(channel)]

            tiles.append(
                Tile(
                    path=file,
                    shape=shape,
                    position=TilePosition(
                        time=time_point,
                        channel=int(channel),
                        z=z,
                        y=int(-float(row["Y"]) / yx_spacing[0]),
                        x=int(float(row["X"]) / yx_spacing[1]),
                    ),
                    background_correction_matrix_path=bgcm,
                    illumination_correction_matrix_path=icm,
                )
            )
        return tiles

    def get_axes(self) -> list[str]:
        if self._z_spacing is not None:
            return ["c", "z", "y", "x"]
        else:
            return ["c", "y", "x"]

    def get_yx_spacing(self) -> tuple[float, float]:
        ch_metadata = self._metadata.iloc[0]
        return (
            float(ch_metadata["VerticalPixelDimension"]),
            float(ch_metadata["HorizontalPixelDimension"]),
        )

    def get_z_spacing(self) -> Optional[float]:
        return self._z_spacing
