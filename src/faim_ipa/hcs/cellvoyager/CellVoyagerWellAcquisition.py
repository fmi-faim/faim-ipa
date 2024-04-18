from decimal import Decimal
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from tifffile import imread

from faim_ipa.hcs.acquisition import TileAlignmentOptions, WellAcquisition
from faim_ipa.hcs.cellvoyager.StackedTile import StackedTile
from faim_ipa.stitching import Tile
from faim_ipa.stitching.Tile import TilePosition


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
        n_planes_in_stacked_tile: int = 1,
    ):
        self._metadata = metadata
        self._z_spacing = self._compute_z_spacing(files)
        self._dtype = self._get_dtype(files)
        self._n_planes_in_stacked_tile = n_planes_in_stacked_tile
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _get_dtype(self, files: pd.DataFrame) -> np.dtype:
        return imread(files["path"].iloc[0]).dtype

    def _compute_z_spacing(self, files: pd.DataFrame) -> Optional[float]:
        if "ZIndex" in files.columns:
            z_steps = np.array(
                files.astype({"Z": float}).groupby("ZIndex", sort=True).mean("Z")["Z"]
            )

            precision = -Decimal(str(z_steps[0])).as_tuple().exponent
            z_step = np.round(np.mean(np.diff(z_steps)), decimals=precision)
            return z_step
        else:
            return None

    def _assemble_tiles(self) -> list[Tile]:
        min_z_index = 0
        max_z_index = min_z_index + 1
        if "ZIndex" in self._files.columns:
            min_z_index = self._files["ZIndex"].min()
            max_z_index = self._files["ZIndex"].max() + 1

        tiles = {}
        for i, row in self._files.iterrows():
            if "ZIndex" in row:
                tile_z_index = (
                    row["ZIndex"] - min_z_index
                ) // self._n_planes_in_stacked_tile
            else:
                tile_z_index = min_z_index
            tczyx_index = (
                row["TimePoint"],
                row["Ch"],
                tile_z_index,
                row["Y"],
                row["X"],
            )
            if tczyx_index not in tiles:
                tiles[tczyx_index] = [row]
            else:
                tiles[tczyx_index].append(row)

        stacked_tiles = []
        for tczyx_index, rows in tiles.items():
            row_dict = {}
            for r in rows:
                if "ZIndex" in r:
                    row_dict[r["ZIndex"]] = r["path"]
                else:
                    row_dict[min_z_index] = r["path"]

            files = []
            z_start = tczyx_index[2] * self._n_planes_in_stacked_tile + min_z_index
            for z in range(
                z_start, min(z_start + self._n_planes_in_stacked_tile, max_z_index)
            ):
                if z in row_dict.keys():
                    files.append(row_dict[z])
                else:
                    files.append(None)

            time_point = tczyx_index[0]
            channel = tczyx_index[1]
            y, x = tczyx_index[3], tczyx_index[4]

            ch_metadata = self._metadata[self._metadata["Ch"] == channel].iloc[0]
            shape = (
                len(files),
                int(ch_metadata["VerticalPixels"]),
                int(ch_metadata["HorizontalPixels"]),
            )

            yx_spacing = self.get_yx_spacing()

            bgcm = None
            if self._background_correction_matrices is not None:
                bgcm = self._background_correction_matrices[str(channel)]

            icm = None
            if self._illumination_correction_matrices is not None:
                icm = self._illumination_correction_matrices[str(channel)]

            stacked_tiles.append(
                StackedTile(
                    paths=files,
                    shape=shape,
                    position=TilePosition(
                        time=time_point,
                        channel=int(channel),
                        z=z_start,
                        y=int(-float(y) / yx_spacing[0]),
                        x=int(float(x) / yx_spacing[1]),
                    ),
                    background_correction_matrix_path=bgcm,
                    illumination_correction_matrix_path=icm,
                    dtype=self._dtype,
                )
            )

        return stacked_tiles

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
