from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
import pandas as pd


class Plate_Acquisition(ABC):
    _acquisition_dir = None
    _files = None
    _wells = None
    _channels = None

    def __init__(self, acquisition_dir: Union[Path, str], mode: str = None) -> None:
        self.acquisition_dir = acquisition_dir
        self.mode = mode
        super().__init__()

    @abstractmethod
    def wells(self):
        """List of wells."""

    @abstractmethod
    def well_acquisitions(self):
        """Iterator over Well_Acquisition objects."""

    @abstractmethod
    def channels(self) -> pd.DataFrame:
        """Table of channels with their metadata.

        Dataframe columns:
          * channel-index
          * channel-name
          * display-color
          * pixel-type
          * spatial-calibration-x
          * spatial-calibration-y
          * [z-scaling]
          * [unit]
          * [wavelength]
          * [exposure-time]
          * [exposure-time-unit]
          * [objective]
        """


class Well_Acquisition(ABC):
    _files = None
    _positions: pd.DataFrame = None
    _channel_metadata = None

    def __init__(self, files: pd.DataFrame, ch_metadata: pd.DataFrame) -> None:
        self._files = files
        self._channel_metadata = ch_metadata
        super().__init__()

    @abstractmethod
    def files(self) -> pd.DataFrame:
        """Table of all files contained in the acquisition.

        Subsets of the acquisition files depending on 'mode'.

        Dataframe columns:
          * index
          * path
          * channel
          * well
          * field
          * *(more optional)
        """

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
