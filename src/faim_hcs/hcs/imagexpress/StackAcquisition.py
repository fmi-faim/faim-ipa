import re
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union

import numpy as np

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.imagexpress import ImageXpressPlateAcquisition
from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff_metadata


class StackAcquisition(ImageXpressPlateAcquisition):
    """Image stack acquisition with a Molecular Devices ImageXpress Micro
    Confocal system.

    MIP-2P-2sub-Stack --> {name} [Optional]
    └── 2023-02-21 --> {date}
        └── 1334 --> {acquisition id}
            ├── ZStep_1
            │   ├── Projection-Mix_E07_s1_w1E78EB128-BD0D-4D94-A6AD-3FF28BB1B105.tif
            │   ├── Projection-Mix_E07_s1_w1_thumb187DE64B-038A-4671-BF6B-683721723769.tif
            │   ├── Projection-Mix_E07_s1_w2C0A49256-E289-4C0F-ADC9-F7728ABDB141.tif
            │   ├── Projection-Mix_E07_s1_w2_thumb57D4B151-71BF-480E-8CC4-C23A2690B763.tif
            │   ├── Projection-Mix_E07_s1_w427CCB2E4-1BF4-45E7-8BC7-264B48EF9C4A.tif
            │   ├── Projection-Mix_E07_s1_w4_thumb555647D0-77F1-4A43-9472-AE509F95E236.tif
            │   ├── ...
            │   └── Projection-Mix_E08_s2_w4_thumbD2785594-4F49-464F-9F80-1B82E30A560A.tif
            ├── ...
            └── ZStep_9
                ├── Projection-Mix_E07_s1_w1091EB8A5-272A-466D-B8A0-7547C6BA392B.tif
                ├── ...
                └── Projection-Mix_E08_s2_w2_thumb210C0D5D-C20E-484D-AFB2-EFE669A56B84.tif

    Image data is stored in {name}_{well}_{field}_w{channel}{md_id}.tif.
    The *_thumb*.tif files, used by Molecular Devices as preview, are ignored.
    """

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
            channel_with_stack = self._files[self._files["z"] == "2"][
                "channel"
            ].unique()[0]
            subset = self._files[self._files["channel"] == channel_with_stack]
            subset = subset[subset["well"] == subset["well"].unique()[0]]
            subset = subset[subset["field"] == subset["field"].unique()[0]]
        else:
            return None

        plane_positions = {}

        for i, row in subset.iterrows():
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
