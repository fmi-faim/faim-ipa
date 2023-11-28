import re
from pathlib import Path
from typing import Union

import pandas as pd

from faim_hcs.io.acquisition import ChannelMetadata, PlateAcquisition, WellAcquisition
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
        self, acquisition_dir: Union[Path, str], mode: str = "top-level"
    ) -> None:
        self.mode = mode
        super().__init__(acquisition_dir=acquisition_dir)

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

    def well_acquisitions(self):
        for well in self.wells:
            yield ImageXpressWellAcquisition(
                files=self._files[self._files["well"] == well],
                ch_metadata=self.channels(),
            )

    def channels(self):
        ch_metadata = []
        for ch in self._files["channel"].unique():
            ch_metadata.append(self._ch_metadata(ch))
        return ch_metadata

    def _ch_metadata(self, channel):
        # Read first image of channel
        path = self._files[self._files["channel"] == channel]["path"].iloc[0]
        metadata = load_metaseries_tiff_metadata(path=path)
        _channel_metadata = _build_ch_metadata(metadata)

        return ChannelMetadata(
            channel_index=None,
            channel_name=channel,
            display_color=_channel_metadata["display-color"],
            spatial_calibration_x=metadata["spatial-calibration-x"],
            spatial_calibration_y=metadata["spatial-calibration-y"],
            spatial_calibration_units=metadata["spatial-calibration-units"],
            z_scaling=None,
            unit=None,
            wavelength=_channel_metadata["wavelength"],
            exposure_time=_channel_metadata["exposure-time"],
            exposure_time_unit=_channel_metadata["exposure-time-unit"],
            objective=metadata["_MagSetting_"],
        )


class ImageXpressWellAcquisition(WellAcquisition):
    _tiles: list[Tile] = None

    def __init__(self, files: pd.DataFrame, ch_metadata: pd.DataFrame) -> None:
        super().__init__(files=files, ch_metadata=ch_metadata)
        self._tiles = self._parse_tiles()

    def _parse_tiles(self) -> list[Tile]:
        tiles = []
        for i, row in self._files.iterrows():
            file = row["path"]
            channel_index = row["channel"]
            time_point = 0
            metadata = load_metaseries_tiff_metadata(file)
            tiles.append(
                Tile(
                    path=file,
                    shape=(metadata["pixel-size-y"], metadata["pixel-size-x"]),
                    position=TilePosition(
                        time=time_point,
                        channel=channel_index,
                        z=metadata["z-position"],
                        y=metadata["stage-position-y"],
                        x=metadata["stage-position-x"],
                    ),
                )
            )
        return tiles
