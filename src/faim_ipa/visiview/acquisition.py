from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from numpy._typing import NDArray
from tifffile import TiffFile

from faim_ipa.hcs.acquisition import TileAlignmentOptions, WellAcquisition
from faim_ipa.io.metadata import ChannelMetadata
from faim_ipa.stitching.tile import Tile, TilePosition
from faim_ipa.visiview.ome_companion_utils import parse_basic_metadata


class StackedTile(Tile):
    def __init__(
        self,
        path: Path | str,
        shape: tuple[int, int],
        position: TilePosition,
        background_correction_matrix_path: Path | str | None = None,
        illumination_correction_matrix_path: Path | str | None = None,
        *,
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
        data = (
            tifffile.memmap(self.path, mode="r")
            if self._memmap
            else tifffile.imread(self.path)
        )

        data = self._apply_background_correction(data)
        return self._apply_illumination_correction(data)

    def load_data_mask(self) -> NDArray:
        return np.ones(self.shape, dtype=bool)


class RegionAcquisitionSTK(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        background_correction_matrices: dict[str, NDArray] | None,
        illumination_correction_matrices: dict[str, NDArray] | None,
        axes: list[str] | None = None,
        *,
        memmap: bool = True,
    ):
        if axes is None:
            axes = ["c", "z", "y", "x"]
        path = files.iloc[0]["path"]
        with TiffFile(path) as tif:
            metadata = tif.stk_metadata
            if metadata is None:
                msg = f"STK metadata is missing. Please check " f"{path}"
                raise ValueError(msg)
            x_spacing = metadata["XCalibration"]
            y_spacing = metadata["YCalibration"]
            self._yx_spacing = (y_spacing, x_spacing)
            self._z_spacing = metadata["ZDistance"].mean()
            self.tile_shape = tif.asarray().shape

        self._axes = axes
        self._memmap = memmap
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _assemble_tiles(self) -> list[Tile]:
        tiles = []
        for _i, row in self._files.iterrows():
            file = row["path"]
            time_point = row["time"]
            channel = row["channel"]

            tiles.append(
                StackedTile(
                    path=file,
                    shape=self.tile_shape,
                    position=TilePosition(
                        time=time_point,
                        channel=channel,
                        z=0,
                        y=int(row["Y"] / self._yx_spacing[0]),
                        x=int(row["X"] / self._yx_spacing[1]),
                    ),
                    memmap=self._memmap,
                )
            )

        return tiles

    def get_z_spacing(self) -> float | None:
        return self._z_spacing

    def get_yx_spacing(self) -> tuple[float, float]:
        return self._yx_spacing

    def get_axes(self) -> list[str]:
        return self._axes


class RegionAcquisitionOME(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        ome_xml: Path | str,
        alignment: TileAlignmentOptions,
        background_correction_matrices: dict[str, NDArray] | None,
        illumination_correction_matrices: dict[str, NDArray] | None,
        axes: list[str] | None = None,
        *,
        memmap: bool = True,
    ):
        if axes is None:
            axes = ["c", "z", "y", "x"]
        self.metadata = parse_basic_metadata(companion_file=ome_xml)
        self.stage_positions = self.metadata["stage_positions"]
        path = files.iloc[0]["path"]
        with TiffFile(path) as tif:
            self.tile_shape = tif.asarray().shape

        self._axes = axes
        self._memmap = memmap
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _assemble_tiles(self) -> list[Tile]:
        tiles = []
        for _i, row in self._files.iterrows():
            file = row["path"]
            time_point = row["time"]
            channel = row["channel"]

            tiles.append(
                StackedTile(
                    path=file,
                    shape=self.tile_shape,
                    position=TilePosition(
                        time=time_point,
                        channel=channel,
                        z=0,
                        y=int(
                            self.stage_positions[row["well"]][0]
                            / self.metadata["yx_spacing"][0]
                        ),
                        x=int(
                            self.stage_positions[row["well"]][1]
                            / self.metadata["yx_spacing"][1]
                        ),
                    ),
                    memmap=self._memmap,
                )
            )

        return tiles

    def get_z_spacing(self) -> float | None:
        return self.metadata["z_spacing"]

    def get_yx_spacing(self) -> tuple[float, float]:
        return self.metadata["yx_spacing"]

    def get_channels(self) -> list[ChannelMetadata]:
        return self.metadata["channels"]

    def get_axes(self) -> list[str]:
        return self._axes
