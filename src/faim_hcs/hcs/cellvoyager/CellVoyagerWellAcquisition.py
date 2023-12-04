from decimal import Decimal
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from faim_hcs.hcs.acquisition import TileAlignmentOptions, WellAcquisition
from faim_hcs.stitching import Tile
from faim_hcs.stitching.Tile import TilePosition


class CellVoyagerWellAcquisition(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        metadata: dict[str, Any],
        background_correction_matrices: dict[str, Union[Path, str]] = None,
        illumination_correction_matrices: dict[str, Union[Path, str]] = None,
    ):
        self._z_spacing = self._compute_z_spacing()
        self._metadata = metadata
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _assemble_tiles(self) -> list[Tile]:
        tiles = []
        for i, row in self._files.iterrows():
            file = row["path"]
            time_point = row["TimePoint"]
            channel = row["Ch"]
            z = row["ZIndex"]

            ch_metadata = self._metadata[self._metadata["Ch"] == channel].iloc[0]
            shape = (
                int(ch_metadata["VerticalPixels"]),
                int(ch_metadata["HorizontalPixels"]),
            )

            yx_spacing = self.get_yx_spacing()

            bgcm = None
            if self._background_correction_matrices is not None:
                bgcm = self._background_correction_matrices[channel]

            icm = None
            if self._illumincation_correction_matrices is not None:
                icm = self._illumincation_correction_matrices[channel]

            tiles.append(
                Tile(
                    path=file,
                    shape=shape,
                    position=TilePosition(
                        time=time_point,
                        channel=int(channel),
                        z=z,
                        y=int(row["Y"] / yx_spacing[0]),
                        x=int(row["X"] / yx_spacing[1]),
                    ),
                    background_correction_matrix_path=bgcm,
                    illumination_correction_matrix_path=icm,
                )
            )

    def get_axes(self) -> list[str]:
        if self._z_spacing is not None:
            return ["z", "y", "x"]
        else:
            return ["y", "x"]

    def get_yx_spacing(self) -> tuple[float, float]:
        ch_metadata = self._metadata.iloc[0]
        return (
            ch_metadata["VerticalPixelDimension"],
            ch_metadata["HorizontalPixelDimension"],
        )

    def get_z_spacing(self) -> Optional[float]:
        return self._z_spacing

    def _compute_z_spacing(self) -> Optional[float]:
        z_steps = np.array(
            [float(i) for i in self._files.groupby("Z").mean("ZIndex").index]
        )

        precision = -Decimal(str(z_steps[0])).as_tuple().exponent
        z_step = np.round(np.mean(np.diff(z_steps)), decimals=precision)
        self._z_spacing = z_step
