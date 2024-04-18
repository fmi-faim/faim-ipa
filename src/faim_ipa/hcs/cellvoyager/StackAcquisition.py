from os.path import exists, join
from pathlib import Path
from typing import Optional, Union
from xml.etree import ElementTree as ET

import pandas as pd
from tqdm import tqdm

from faim_ipa.hcs.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_ipa.hcs.cellvoyager.CellVoyagerWellAcquisition import (
    CellVoyagerWellAcquisition,
)
from faim_ipa.io.ChannelMetadata import ChannelMetadata

BTS_NS = "{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}"


class StackAcquisition(PlateAcquisition):
    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, Union[Path, str]]] = None,
        illumination_correction_matrices: Optional[dict[str, Union[Path, str]]] = None,
        n_planes_in_stacked_tile: int = 1,
    ):
        self._n_planes_in_stacked_tile = n_planes_in_stacked_tile
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def get_channel_metadata(self) -> dict[int, ChannelMetadata]:
        metadata = self._parse_metadata()
        ch_metadata = {}

        for i, row in metadata.iterrows():
            index = int(row["Ch"]) - 1
            ch_metadata[index] = ChannelMetadata(
                channel_index=index,
                channel_name=row["Ch"],
                display_color=row["Color"],
                spatial_calibration_x=row["HorizontalPixelDimension"],
                spatial_calibration_y=row["VerticalPixelDimension"],
                spatial_calibration_units="um",
                z_spacing=self.get_z_spacing(),
                wavelength=row["Target"],
                exposure_time=row["ExposureTime"],
                exposure_time_unit="ms",
                objective=row["Objective"],
            )

        assert min(ch_metadata.keys()) == 0, "Channel indices must start at 0."

        return ch_metadata

    def get_z_spacing(self) -> float:
        return self._wells[0].get_z_spacing()

    def _build_well_acquisitions(self, files: pd.DataFrame) -> list[WellAcquisition]:
        wells = []
        for well in tqdm(files["well"].unique()):
            wells.append(
                CellVoyagerWellAcquisition(
                    files=files[files["well"] == well],
                    alignment=self._alignment,
                    metadata=self._parse_metadata(),
                    background_correction_matrices=self._background_correction_matrices,
                    illumination_correction_matrices=self._illumination_correction_matrices,
                    n_planes_in_stacked_tile=self._n_planes_in_stacked_tile,
                )
            )
        return wells

    def _parse_metadata(self) -> pd.DataFrame:
        mrf_file = join(self._acquisition_dir, "MeasurementDetail.mrf")
        if not exists(mrf_file):
            raise ValueError(
                f"MeasurementDetail.mrf not found in: {self._acquisition_dir}"
            )
        mrf_tree = ET.parse(mrf_file)
        mrf_root = mrf_tree.getroot()

        channels = []
        for channel in mrf_root.findall(BTS_NS + "MeasurementChannel"):
            row = {
                key.replace(BTS_NS, ""): value for key, value in channel.attrib.items()
            }
            channels.append(row)

        mes_file = join(
            self._acquisition_dir,
            mrf_root.attrib[BTS_NS + "MeasurementSettingFileName"],
        )
        if not exists(mes_file):
            raise ValueError(f"Settings file not found: {mes_file}")
        mes_tree = ET.parse(mes_file)
        mes_root = mes_tree.getroot()

        channel_settings = []
        for channel in mes_root.find(BTS_NS + "ChannelList").findall(
            BTS_NS + "Channel"
        ):
            row = {
                key.replace(BTS_NS, ""): value for key, value in channel.attrib.items()
            }
            channel_settings.append(row)

        return pd.merge(
            pd.DataFrame(channels),
            pd.DataFrame(channel_settings),
            left_on="Ch",
            right_on="Ch",
        )

    def _parse_files(self) -> pd.DataFrame:
        mlf_file = join(self._acquisition_dir, "MeasurementData.mlf")
        if not exists(mlf_file):
            raise ValueError(
                f"MeasurementData.mlf not found in: {self._acquisition_dir}"
            )
        mlf_tree = ET.parse(mlf_file)
        mlf_root = mlf_tree.getroot()

        files = []
        for record in mlf_root.findall(BTS_NS + "MeasurementRecord"):
            row = {
                key.replace(BTS_NS, ""): value for key, value in record.attrib.items()
            }
            if row.pop("Type") == "IMG":
                row |= {
                    "path": join(self._acquisition_dir, record.text),
                    "well": chr(ord("@") + int(row.pop("Row")))
                    + row.pop("Column").zfill(2),
                }
                files.append(row)

        files = pd.DataFrame(files)
        files["TimePoint"] = files["TimePoint"].astype(int)
        files["ZIndex"] = files["ZIndex"].astype(int)
        return files
