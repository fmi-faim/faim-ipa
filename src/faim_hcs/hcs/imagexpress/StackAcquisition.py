import re
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union

import numpy as np

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.imagexpress import ImageXpressPlateAcquisition
from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff_metadata


class StackAcquisition(ImageXpressPlateAcquisition):
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
        self._z_spacing = self._compute_z_spacing()

    def _get_root_re(self) -> re.Pattern:
        return re.compile(
            r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?:[\/\\]ZStep_(?P<z>\d+))"
        )

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)_(?P<channel>w[1-9]{1})(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
        )

    def _get_z_spacing(self) -> Optional[float]:
        return self._z_spacing

    def _compute_z_spacing(
        self,
    ) -> Optional[float]:
        if "z" in self._files.columns:
            channels_with_stack = self._files[self._files["z"] == "2"][
                "channel"
            ].unique()
        else:
            return None

        plane_positions = {}

        for i, row in self._files[
            self._files["channel"].isin(channels_with_stack)
        ].iterrows():
            file = row["path"]
            if "z" in row.keys() and row["z"] is not None:
                z = int(row["z"])
                metadata = load_metaseries_tiff_metadata(file)
                z_position = metadata["stage-position-z"]
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
