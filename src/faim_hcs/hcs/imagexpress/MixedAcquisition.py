import re
from pathlib import Path
from typing import Optional, Union

from numpy._typing import NDArray

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.imagexpress import StackAcquisition


class MixedAcquisition(StackAcquisition):
    def __init__(
        self,
        acquisition_dir: Union[Path, str],
        alignment: TileAlignmentOptions,
        background_correction_matrix: Optional[dict[str, NDArray]] = None,
        illumination_correction_matrix: Optional[NDArray] = None,
    ):
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrix,
            illumination_correction_matrices=illumination_correction_matrix,
        )
        self._filter_mips()

    def _get_root_re(self) -> re.Pattern:
        return re.compile(
            r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?:[\/\\]ZStep_(?P<z>\d+))?.*"
        )

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)_(?P<channel>w[1-9]{1})(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
        )

    def _filter_mips(self):
        """Remove MIP files if the whole stack was acquired."""
        for ch in self._files["channel"].unique():
            channel_files = self._files[self._files["channel"] == ch]
            z_positions = channel_files["z"].unique()
            has_mip = None in z_positions
            has_stack = len(z_positions) > 1
            if has_mip and has_stack:
                self._files.drop(
                    self._files[
                        (self._files["channel"] == ch) & (self._files["z"].isna())
                    ].index,
                    inplace=True,
                )
