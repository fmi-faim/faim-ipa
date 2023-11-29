import re
from decimal import Decimal
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from faim_hcs import MetaSeriesUtils
from faim_hcs.io.acquisition import (
    ChannelMetadata,
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff_metadata
from faim_hcs.MetaSeriesUtils import _build_ch_metadata
from faim_hcs.stitching import Tile
from faim_hcs.stitching.Tile import TilePosition


class ImageXpressPlateAcquisition(PlateAcquisition):
    _METASERIES_FILENAME_PATTERN = r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)_(?P<channel>w[1-9]{1})(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
    _METASERIES_FOLDER_PATTERN = r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?:[\/\\]ZStep_(?P<z>\d+))?.*"
    _METASERIES_MAIN_FOLDER_PATTERN = (
        r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?![\/\\]ZStep_.*)"
    )
    _METASERIES_ZSTEP_FOLDER_PATTERN = r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)[\/\\]ZStep_(?P<z>\d+).*"

    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        mode: str = "top-level",
    ) -> None:
        self.mode = mode
        super().__init__(acquisition_dir=acquisition_dir, alignment=alignment)

    def _get_root_re(self) -> re.Pattern:
        if self.mode == "top-level":
            root_pattern = self._METASERIES_MAIN_FOLDER_PATTERN
        elif self.mode == "z-steps":
            root_pattern = self._METASERIES_ZSTEP_FOLDER_PATTERN
        else:
            root_pattern = self._METASERIES_FOLDER_PATTERN
        return re.compile(root_pattern)

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(self._METASERIES_FILENAME_PATTERN)

    def get_well_acquisitions(self) -> list["WellAcquisition"]:
        return [
            ImageXpressWellAcquisition(
                files=self._files[self._files["well"] == well],
                alignment=self._alignment,
            )
            for well in self._files["well"].unique()
        ]

    def get_channel_metadata(self) -> dict[str, ChannelMetadata]:
        ch_metadata = {}
        for ch in self._files["channel"].unique():
            channel_files = self._files[self._files["channel"] == ch]
            path = channel_files["path"].iloc[0]
            metadata = load_metaseries_tiff_metadata(path=path)
            channel_metadata = _build_ch_metadata(metadata)
            ch_metadata[ch] = ChannelMetadata(
                channel_index=int(ch[1:]) - 1,
                channel_name=ch,
                display_color=channel_metadata["display-color"],
                spatial_calibration_x=metadata["spatial-calibration-x"],
                spatial_calibration_y=metadata["spatial-calibration-y"],
                spatial_calibration_units=metadata["spatial-calibration-units"],
                z_scaling=self._compute_z_scaling(channel_files),
                wavelength=channel_metadata["wavelength"],
                exposure_time=channel_metadata["exposure-time"],
                exposure_time_unit=channel_metadata["exposure-time-unit"],
                objective=metadata["_MagSetting_"],
            )

        return ch_metadata

    def _compute_z_scaling(self, channel_files: pd.DataFrame) -> float:
        # TODO: Fix z-scaling computation.
        plane_positions = {}
        for i, row in channel_files.iterrows():
            file = row["path"]
            z = MetaSeriesUtils.extract_z_position(row)
            metadata = load_metaseries_tiff_metadata(file)
            z_position = metadata["z-position"]
            if z_position is not None:
                if z in plane_positions.keys():
                    plane_positions[z].append(z_position)
                else:
                    plane_positions[z] = [z_position]

        if len(plane_positions) > 1:
            plane_positions = dict(sorted(plane_positions.items()))
            average_z_positions = []
            for z, positions in plane_positions.items():
                average_z_positions.append(np.mean(positions))

            precision = -Decimal(str(plane_positions[1][0])).as_tuple().exponent
            z_step = np.round(np.mean(np.diff(average_z_positions)), decimals=precision)
            return z_step
        else:
            return None


class ImageXpressWellAcquisition(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
    ) -> None:
        super().__init__(files=files, alignment=alignment)

    def _parse_tiles(self) -> list[Tile]:
        tiles = []
        for i, row in self._files.iterrows():
            file = row["path"]
            time_point = 0
            channel_index = int(row["channel"][1:])
            z = MetaSeriesUtils.extract_z_position(row)
            metadata = load_metaseries_tiff_metadata(file)
            tiles.append(
                Tile(
                    path=file,
                    shape=(metadata["pixel-size-y"], metadata["pixel-size-x"]),
                    position=TilePosition(
                        time=time_point,
                        channel=channel_index,
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
                )
            )
        return tiles
