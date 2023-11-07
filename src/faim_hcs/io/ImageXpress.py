import os
import re
from pathlib import Path
from typing import Union

import pandas as pd

from faim_hcs.io.acquisition import Plate_Acquisition, Well_Acquisition
from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff_metadata


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

    def well_acquisitions(self):
        for well in self.wells():
            yield ImageXpress_Well_Acquisition(self._files[self._files["well"] == well])

    def channels(self):
        pass  # TODO

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
