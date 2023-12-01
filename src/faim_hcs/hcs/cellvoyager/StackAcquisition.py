from pathlib import Path
from typing import Optional, Union

import pandas as pd

from faim_hcs.hcs.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_hcs.io.metadata import ChannelMetadata


class StackAcquistion(PlateAcquisition):
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

    def get_channel_metadata(self) -> dict[str, ChannelMetadata]:
        pass

    def get_well_acquisitions(self) -> list[WellAcquisition]:
        pass

    def _parse_files(self) -> pd.DataFrame:
        pass
