import re
from pathlib import Path
from typing import Optional, Union

from faim_hcs.io.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_hcs.io.ImageXpress import ImageXpressWellAcquisition
from faim_hcs.io.metadata import ChannelMetadata
from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff_metadata


class SinglePlaneAcquisition(PlateAcquisition):
    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, Union[Path, str]]] = None,
        illumination_correction_matrices: Optional[dict[str, Union[Path, str]]] = None,
    ):
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _get_root_re(self) -> re.Pattern:
        return re.compile(r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)")

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)_(?P<channel>w[1-9]{1})(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
        )

    def get_well_acquisitions(self) -> list[WellAcquisition]:
        return [
            ImageXpressWellAcquisition(
                files=self._files[self._files["well"] == well],
                alignment=self._alignment,
                z_spacing=None,
            )
            for well in self._files["well"].unique()
        ]

    def get_channel_metadata(self) -> dict[str, ChannelMetadata]:
        ch_metadata = {}
        for ch in self._files["channel"].unique():
            channel_files = self._files[self._files["channel"] == ch]
            path = channel_files["path"].iloc[0]
            metadata = load_metaseries_tiff_metadata(path=path)
            from faim_hcs.MetaSeriesUtils import _build_ch_metadata

            channel_metadata = _build_ch_metadata(metadata)
            ch_metadata[ch] = ChannelMetadata(
                channel_index=int(ch[1:]) - 1,
                channel_name=ch,
                display_color=channel_metadata["display-color"],
                spatial_calibration_x=metadata["spatial-calibration-x"],
                spatial_calibration_y=metadata["spatial-calibration-y"],
                spatial_calibration_units=metadata["spatial-calibration-units"],
                z_spacing=None,
                wavelength=channel_metadata["wavelength"],
                exposure_time=channel_metadata["exposure-time"],
                exposure_time_unit=channel_metadata["exposure-time-unit"],
                objective=metadata["_MagSetting_"],
            )

        return ch_metadata
