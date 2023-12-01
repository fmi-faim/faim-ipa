import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from faim_hcs.io.metadata import ChannelMetadata
from faim_hcs.stitching import Tile


class TileAlignmentOptions(Enum):
    """Tile alignment options."""

    STAGE_POSITION = "StageAlignment"
    GRID = "GridAlignment"


class PlateAcquisition(ABC):
    _acquisition_dir = None
    _files = None
    _alignment: TileAlignmentOptions = None
    _background_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _illumination_correction_matrices: Optional[dict[str, Union[Path, str]]]

    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, Union[Path, str]]],
        illumination_correction_matrices: Optional[dict[str, Union[Path, str]]],
    ) -> None:
        self._acquisition_dir = acquisition_dir
        self._files = self._parse_files()
        self._alignment = alignment
        self._background_correction_matrices = background_correction_matrices
        self._illumination_correction_matrices = illumination_correction_matrices
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

    @abstractmethod
    def get_well_acquisitions(self) -> list["WellAcquisition"]:
        """List of wells."""
        raise NotImplementedError()

    @abstractmethod
    def get_channel_metadata(self) -> dict[str, ChannelMetadata]:
        """Channel metadata."""
        raise NotImplementedError()

    def get_well_names(self) -> Iterable[str]:
        for well in self.get_well_acquisitions():
            yield well.name


class WellAcquisition(ABC):
    name: str = None
    _files = None
    _alignment: TileAlignmentOptions = None
    _background_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _illumincation_correction_matrices: Optional[dict[str, Union[Path, str]]]
    _tiles = None

    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, Union[Path, str]]],
        illumination_correction_matrices: Optional[dict[str, Union[Path, str]]],
    ) -> None:
        assert (
            files["well"].nunique() == 1
        ), "WellAcquisition must contain files from a single well."
        self.name = files["well"].iloc[0]
        self._files = files
        self._alignment = alignment
        self._background_correction_matrices = background_correction_matrices
        self._illumincation_correction_matrices = illumination_correction_matrices
        self._tiles = self._align_tiles(tiles=self._parse_tiles())
        super().__init__()

    @abstractmethod
    def _parse_tiles(self) -> list[Tile]:
        """Parse all tiles in the well."""
        raise NotImplementedError()

    def _align_tiles(self, tiles: list[Tile]) -> list[Tile]:
        if self._alignment == TileAlignmentOptions.STAGE_POSITION:
            from faim_hcs.alignment import StageAlignment

            return StageAlignment(tiles=tiles).get_tiles()

        if self._alignment == TileAlignmentOptions.GRID:
            from faim_hcs.alignment import GridAlignment

            return GridAlignment(tiles=tiles).get_tiles()

        raise ValueError(f"Unknown alignment option: {self._alignment}")

    def get_tiles(self) -> list[Tile]:
        """List of tiles."""
        return self._tiles

    def get_row_col(self) -> tuple[str, str]:
        return self.name[0], self.name[1:]

    def get_axes(self) -> list[str]:
        if "z" in self._files.columns:
            return ["c", "z", "y", "x"]
        else:
            return ["c", "y", "x"]

    @abstractmethod
    def get_yx_spacing(self) -> tuple[float, float]:
        raise NotImplementedError()

    @abstractmethod
    def get_z_spacing(self) -> Optional[float]:
        raise NotImplementedError()

    def get_coordinate_transformations(
        self, max_layer: int, yx_binning: int
    ) -> list[dict[str, Any]]:
        transformations = []
        for s in range(max_layer + 1):
            if self.get_z_spacing() is not None:
                transformations.append(
                    [
                        {
                            "scale": [
                                1.0,
                                self.get_z_spacing(),
                                self.get_yx_spacing()[0] * yx_binning * 2**s,
                                self.get_yx_spacing()[1] * yx_binning * 2**s,
                            ],
                            "type": "scale",
                        }
                    ]
                )
            else:
                transformations.append(
                    [
                        {
                            "scale": [
                                1.0,
                                self.get_yx_spacing()[0] * yx_binning * 2**s,
                                self.get_yx_spacing()[1] * yx_binning * 2**s,
                            ],
                            "type": "scale",
                        }
                    ]
                )

        return transformations

    # TODO: Move in dedicated class.
    # @abstractmethod
    # def roi_tables(self) -> list[dict]:
    #     """ROI tables corresponding to the fields in this well.
    #
    #     Contains:
    #       * well_ROI_table
    #       * FOV_ROI_table
    #
    #     each with columns:
    #       * x_micrometer
    #       * y_micrometer
    #       * z_micrometer
    #       * len_x_micrometer
    #       * len_y_micrometer
    #       * len_z_micrometer
    #     """
    #     raise NotImplementedError()
