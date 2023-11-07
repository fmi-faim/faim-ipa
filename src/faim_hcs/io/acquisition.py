from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd


class Plate_Acquisition(ABC):
    _acquisition_dir = None
    _files = None
    _wells = None

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
    _positions = None

    def __init__(self, files: pd.DataFrame) -> None:
        self._files = files
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
