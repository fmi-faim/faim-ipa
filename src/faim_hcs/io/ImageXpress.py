from typing import Optional

import pandas as pd
from numpy._typing import NDArray

from faim_hcs.io.acquisition import TileAlignmentOptions, WellAcquisition
from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff_metadata
from faim_hcs.stitching import Tile
from faim_hcs.stitching.Tile import TilePosition


class ImageXpressWellAcquisition(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        z_spacing: Optional[float],
        background_correction_matrices: dict[str, Optional[NDArray]] = None,
        illumination_correction_matrices: dict[Optional[NDArray]] = None,
    ) -> None:
        self._z_spacing = z_spacing
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _parse_tiles(self) -> list[Tile]:
        tiles = []
        for i, row in self._files.iterrows():
            file = row["path"]
            time_point = 0
            channel = row["channel"]
            metadata = load_metaseries_tiff_metadata(file)
            if self._z_spacing is None:
                z = 0
            else:
                z = int(metadata["stage-position-z"] / self._z_spacing)

            bgcm = None
            if self._background_correction_matrices is not None:
                bgcm = self._background_correction_matrices[channel]

            icm = None
            if self._illumincation_correction_matrices is not None:
                icm = self._illumincation_correction_matrices[channel]

            tiles.append(
                Tile(
                    path=file,
                    shape=(metadata["pixel-size-y"], metadata["pixel-size-x"]),
                    position=TilePosition(
                        time=time_point,
                        channel=int(channel[1:]),
                        z=z,
                        y=int(
                            metadata["stage-position-y"]
                            / metadata["spatial-calibration-y"]
                        ),
                        x=int(
                            metadata["stage-position-x"]
                            / metadata["spatial-calibration-x"]
                        ),
                    ),
                    background_correction_matrix=bgcm,
                    illumination_correction_matrix=icm,
                )
            )
        return tiles

    def get_yx_spacing(self) -> tuple[float, float]:
        metadata = load_metaseries_tiff_metadata(self._files.iloc[0]["path"])
        return (metadata["spatial-calibration-y"], metadata["spatial-calibration-x"])

    def get_z_spacing(self) -> Optional[float]:
        return self._z_spacing
