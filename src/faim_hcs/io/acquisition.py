import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import dask.array as da
import numpy as np
import pandas as pd
from pydantic import BaseModel, NonNegativeInt, PositiveFloat, PositiveInt

from faim_hcs.stitching import Tile


class PlateAcquisition(ABC):
    _acquisition_dir = None
    _files = None
    wells = None

    def __init__(self, acquisition_dir: Union[Path, str]) -> None:
        self._acquisition_dir = acquisition_dir
        self._files = self._parse_files()
        self.wells = self._get_wells()
        super().__init__()

    def _parse_files(self) -> pd.DataFrame:
        """Parse all files in the acquisition directory.

        Returns
        -------
        DataFrame
            Table of all files in the acquisition.
        """
        return pd.DataFrame(
            self._list_and_match_files(
                root_dir=self._acquisition_dir,
                root_re=self._get_root_re(),
                filename_re=self._get_filename_re(),
            )
        )

    def _list_and_match_files(
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

    @abstractmethod
    def _get_root_re(self) -> re.Pattern:
        """Regular expression for matching the root directory of the acquisition."""
        raise NotImplementedError()

    @abstractmethod
    def _get_filename_re(self) -> re.Pattern:
        """Regular expression for matching the filename of the acquisition."""
        raise NotImplementedError()

    def _get_wells(self) -> list["WellAcquisition"]:
        """List of wells."""
        return sorted(self._files["well"].unique())

    @abstractmethod
    def well_acquisitions(self):
        """Iterator over Well_Acquisition objects."""
        raise NotImplementedError()


class WellAcquisition(ABC):
    _files = None
    _positions: pd.DataFrame = None
    _channel_metadata = None

    def __init__(self, files: pd.DataFrame, ch_metadata: pd.DataFrame) -> None:
        self._files = files
        self._channel_metadata = ch_metadata
        super().__init__()

    @abstractmethod
    def _parse_tiles(self) -> list[Tile]:
        """Parse all tiles in the well."""
        raise NotImplementedError()

    @abstractmethod
    def positions(self) -> pd.DataFrame:
        """Table of stage positions corresponding to files.

        Dataframe columns:
          * index
          * x
          * y
          * [z]
          * [units]
        """

    def pixel_positions(self) -> np.ndarray:
        """Positions of all fields in pixel coordinates.

        The order corresponds to the order of fields,
        i.e. the first dimension of read_array()
        """
        x_spacing = self._channel_metadata[0]["spatial-calibration-x"]
        y_spacing = self._channel_metadata[0]["spatial-calibration-y"]
        x_pos = self._positions["pos_x"] / x_spacing
        y_pos = self._positions["pos_y"] / y_spacing
        return np.array((y_pos, x_pos)).T

    def read_array(self) -> da.Array:
        """Dask array in FC(Z)YX dimension order.

        The order of the field (dimension 0) corresponds to
        the order of pixel_positions().
        """

    @abstractmethod
    def roi_tables(self) -> list[dict]:
        """ROI tables corresponding to the fields in this well.

        Contains:
          * well_ROI_table
          * FOV_ROI_table

        each with columns:
          * x_micrometer
          * y_micrometer
          * z_micrometer
          * len_x_micrometer
          * len_y_micrometer
          * len_z_micrometer
        """


class TileMetadata(BaseModel):
    tile_size_x: PositiveInt
    tile_size_y: PositiveInt


class ChannelMetadata(BaseModel):
    channel_index: Optional[NonNegativeInt]
    channel_name: str
    display_color: str
    spatial_calibration_x: float
    spatial_calibration_y: float
    spatial_calibration_units: str
    z_scaling: Optional[PositiveFloat]
    unit: Optional[str]
    wavelength: PositiveInt
    exposure_time: PositiveFloat
    exposure_time_unit: str
    objective: str

    def __init__(
        self,
        channel_index: Optional[NonNegativeInt],
        channel_name: str,
        display_color: str,
        spatial_calibration_x: float,
        spatial_calibration_y: float,
        spatial_calibration_units: str,
        z_scaling: Optional[PositiveFloat],
        unit: Optional[str],
        wavelength: PositiveInt,
        exposure_time: PositiveFloat,
        exposure_time_unit: str,
        objective: str,
    ):
        super().__init__()
        self.channel_index = channel_index
        self.channel_name = channel_name
        self.display_color = display_color
        self.spatial_calibration_x = spatial_calibration_x
        self.spatial_calibration_y = spatial_calibration_y
        self.spatial_calibration_units = spatial_calibration_units
        self.z_scaling = z_scaling
        self.unit = unit
        self.wavelength = wavelength
        self.exposure_time = exposure_time
        self.exposure_time_unit = exposure_time_unit
        self.objective = objective
