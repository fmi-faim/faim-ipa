import os
import re
from pathlib import Path
from typing import Union

import pandas as pd

from faim_hcs.io.acquisition import Plate_Acquisition, Well_Acquisition
from faim_hcs.io.MetaSeriesTiff import (
    load_metaseries_tiff,
    load_metaseries_tiff_metadata,
)
from faim_hcs.MetaSeriesUtils import _build_ch_metadata


class ImageXpress_Plate_Acquisition(Plate_Acquisition):
    _METASERIES_FILENAME_PATTERN = r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)_(?P<channel>w[1-9]{1})(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
    _METASERIES_FOLDER_PATTERN = r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?:[\/\\]ZStep_(?P<z>\d+))?.*"
    _METASERIES_MAIN_FOLDER_PATTERN = (
        r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?![\/\\]ZStep_.*)"
    )
    _METASERIES_ZSTEP_FOLDER_PATTERN = r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)[\/\\]ZStep_(?P<z>\d+).*"

    def wells(self):
        if self._wells is None:
            self._populate_wells()
        return self._wells

    def well_acquisitions(self):  # TODO consider moving this logic in parent class
        for well in self.wells():
            yield ImageXpress_Well_Acquisition(
                files=self._files[self._files["well"] == well],
                ch_metadata=self.channels(),
            )

    def channels(self):
        ch_metadata = []
        for ch in self._files["channel"].unique():
            ch_metadata.append(self._ch_metadata(ch))
        # also include tile size (ny, nx)
        return ch_metadata

    def _ch_metadata(self, channel):
        # Read first image of channel
        path = self._files[self._files["channel"] == channel]["path"].iloc[0]
        data, metadata = load_metaseries_tiff(path=path)
        _channel_metadata = _build_ch_metadata(metadata)

        return {
            "channel-index": None,
            "channel-name": channel,
            "display-color": _channel_metadata["display-color"],
            "pixel-type": metadata["PixelType"],
            "spatial-calibration-x": metadata["spatial-calibration-x"],
            "spatial-calibration-y": metadata["spatial-calibration-y"],
            "spatial-calibration-units": metadata["spatial-calibration-units"],
            "z-scaling": None,
            "unit": None,
            "wavelength": _channel_metadata["wavelength"],
            "exposure-time": _channel_metadata["exposure-time"],
            "exposure-time-unit": _channel_metadata["exposure-time-unit"],
            "objective": metadata["_MagSetting_"],
            "tile-size-x": data.shape[-1],
            "tile-size-y": data.shape[-2],
        }

    def _populate_wells(self):
        if self._files is None:
            self._parse_files()
        self._wells = sorted(self._files["well"].unique())

    def _parse_files(self):
        if self.mode == "top-level":
            root_pattern = self._METASERIES_MAIN_FOLDER_PATTERN
        elif self.mode == "z-steps":
            root_pattern = self._METASERIES_ZSTEP_FOLDER_PATTERN
        else:
            root_pattern = self._METASERIES_FOLDER_PATTERN
        self._files = pd.DataFrame(
            self._list_dataset_files(
                root_dir=self.acquisition_dir,
                root_re=re.compile(root_pattern),
                filename_re=re.compile(self._METASERIES_FILENAME_PATTERN),
            )
        )

    def _list_dataset_files(
        self,
        root_dir: Union[Path, str],
        root_re: re.Pattern,
        filename_re: re.Pattern,
    ) -> list[str]:
        files = []
        for root, _, filenames in os.walk(root_dir):
            m_root = root_re.fullmatch(root)
            if m_root:
                for f in filenames:
                    m_filename = filename_re.fullmatch(f)
                    if m_filename:
                        row = m_root.groupdict()
                        row |= m_filename.groupdict()
                        row["path"] = str(Path(root).joinpath(f))
                        files.append(row)
        return files


class ImageXpress_Well_Acquisition(Well_Acquisition):
    def files(self) -> pd.DataFrame:
        return self._files

    def positions(self) -> pd.DataFrame:
        if self._positions is None:
            self._parse_positions()
        return self._positions

    def roi_tables(self) -> list[dict]:
        pass  # TODO

    def _parse_positions(self):
        path = []
        pos_x = []
        pos_y = []
        pos_z = []
        for file in self.files()["path"]:
            path.append(file)
            x, y, z = self._get_position(file)
            pos_x.append(x)
            pos_y.append(y)
            pos_z.append(z)
        self._positions = pd.DataFrame(
            {
                "path": path,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "pos_z": pos_z,
            }
        )

    def _get_position(self, file):
        metadata = load_metaseries_tiff_metadata(file)
        return (
            metadata["stage-position-x"],
            metadata["stage-position-y"],
            metadata["z-position"],
        )
