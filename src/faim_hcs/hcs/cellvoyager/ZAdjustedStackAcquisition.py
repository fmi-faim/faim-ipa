from os.path import join
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pandas.core.api import DataFrame as DataFrame

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.cellvoyager.StackAcquisition import StackAcquisition


class ZAdjustedStackAcquisition(StackAcquisition):
    _trace_log_files = []

    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        trace_log_files: list[Union[Path, str]],
        alignment: TileAlignmentOptions,
        background_correction_matrices: Optional[dict[str, Union[Path, str]]] = None,
        illumination_correction_matrices: Optional[dict[str, Union[Path, str]]] = None,
        n_planes_in_stacked_tile: int = 1,
    ):
        self._trace_log_files = trace_log_files
        super().__init__(
            acquisition_dir,
            alignment,
            background_correction_matrices,
            illumination_correction_matrices,
            n_planes_in_stacked_tile=n_planes_in_stacked_tile,
        )

    def _parse_files(self) -> DataFrame:
        files = super()._parse_files()
        z_mapping = self._create_z_mapping()
        # merge files left with mapping on path
        merged = files.merge(z_mapping, how="left", left_on=["path"], right_on=["path"])
        min_z = np.min(merged["z_pos"].astype(float))
        z_spacing = np.mean(
            merged[merged["ZIndex"].astype(int) == 2]["Z"].astype(float)
        ) - np.mean(merged[merged["ZIndex"].astype(int) == 1]["Z"].astype(float))
        merged["ZIndex"] = np.round(
            (merged["z_pos"].astype(float) - min_z) / z_spacing
        ).astype(int)
        # update Z
        merged["Z"] = merged["z_pos"]
        return merged

    def _create_z_mapping(self) -> DataFrame:
        z_pos = []
        filenames = []
        value = None
        for trace_file in self._trace_log_files:
            with open(trace_file) as log:
                for line in log:
                    tokens = line.split(",")
                    if (
                        (len(tokens) > 14)
                        and (tokens[7] == "--->")
                        and (tokens[8] == "MS_MANU")
                    ):
                        value = float(tokens[14])
                    if (
                        (len(tokens) > 12)
                        and (tokens[7] == "--->")
                        and (tokens[8] == "AF_MANU")
                        and (tokens[9] == "34")
                    ):
                        value = float(tokens[12])
                    if (
                        (len(tokens) > 8)
                        and (tokens[4] == "Measurement")
                        and (tokens[7] == "_init_frame_save")
                    ):
                        filename = tokens[8]
                        if value is None:
                            raise ValueError(f"No z position found for {filename}")
                        filenames.append(join(self._acquisition_dir, filename))
                        z_pos.append(value)
                        value = None

        return DataFrame(
            {
                "path": filenames,
                "z_pos": z_pos,
            }
        )
