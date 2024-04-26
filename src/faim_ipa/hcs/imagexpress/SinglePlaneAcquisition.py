import re
from pathlib import Path
from typing import Optional, Union

from faim_ipa.hcs.acquisition import TileAlignmentOptions
from faim_ipa.hcs.imagexpress import ImageXpressPlateAcquisition


class SinglePlaneAcquisition(ImageXpressPlateAcquisition):
    """Parse top folder (single planes) of an acquisition of a MolecularDevices ImageXpress Micro Confocal system.

    Storage layout on disk for 2 wells with 2 fields and 2 channels::

        MIP-2P-2sub --> {name} [Optional]
        └── 2022-07-05 --> {date}
            └── 1075 --> {acquisition id}
                ├── MIP-2P-2sub_C05_s1_w146C9B2CD-0BB3-4B8A-9187-2805F4C90506.tif
                ├── MIP-2P-2sub_C05_s1_w1_thumb6EFE77C6-B96D-412A-9FD1-710DBDA32821.tif
                ├── MIP-2P-2sub_C05_s1_w2B90625C8-6EA7-4E54-8289-C539EB75263E.tif
                ├── MIP-2P-2sub_C05_s1_w2_thumbEDDF803A-AE5E-4190-8C06-F54341AEC4A6.tif
                ├── MIP-2P-2sub_C05_s2_w1E2913F7F-E229-4B6A-BFED-02BCF54561FA.tif
                ├── MIP-2P-2sub_C05_s2_w1_thumb72E3641A-C91B-4501-900A-245BAC58FF46.tif
                ├── MIP-2P-2sub_C05_s2_w241C38630-BCFD-4393-8706-58755CECE059.tif
                ├── MIP-2P-2sub_C05_s2_w2_thumb5377A5AC-9BBF-4BAF-99A2-24896E3373A2.tif
                ├── MIP-2P-2sub_C06_s1_w152C23B9A-EB4C-4DF6-8A7F-F4147A9E7DDE.tif
                ├── MIP-2P-2sub_C06_s1_w1_thumb541AA634-387C-4B84-B0D8-EE4CB1C88E81.tif
                ├── MIP-2P-2sub_C06_s1_w2FB0D7D9B-3EA0-445E-9A05-7D01154A9A5C.tif
                ├── MIP-2P-2sub_C06_s1_w2_thumb8FA1E466-57CD-4237-B09B-CAB48154647D.tif
                ├── MIP-2P-2sub_C06_s2_w1F365E60C-BCC2-4B74-9856-BCE07C8B0FD3.tif
                ├── MIP-2P-2sub_C06_s2_w1_thumb9652366E-36A0-4B7F-8B18-DA89D7DB41BD.tif
                ├── MIP-2P-2sub_C06_s2_w20EEC6AEA-1727-41E6-806C-40FF6AF68B6C.tif
                └── MIP-2P-2sub_C06_s2_w2_thumb710CD846-0185-4362-BBAF-C700AE0013B3.tif

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

    def _get_root_re(self) -> re.Pattern:
        return re.compile(
            r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?:[\/\\]TimePoint_(?P<t>\d+))?"
        )

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_?(?P<field>s\d+)?_?(?P<channel>w[1-9]{1})?(?!_thumb)(?P<md_id>[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12})(?P<ext>.tif)"
        )

    def _get_z_spacing(self) -> Optional[float]:
        return None
