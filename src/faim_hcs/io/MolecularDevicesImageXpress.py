import os
import re
from pathlib import Path
from typing import Union

import pandas as pd


def parse_single_plane_multi_fields(acquisition_dir: Union[Path, str]) -> pd.DataFrame:
    """Parse a single plane multi-field acquisition of a Molecular Devices
    ImageXpress Micro Confocal system.

    Storage hierarchy on disk for 2 wells with 2 fields and 2 channels:
    MIP-2P-2sub --> {name}
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
    The *_thumb*.tif files are used by Molecular Devices as preview.

    :param acquisition_dir: Path to acquisition directory.
    :return: table of all acquired image data.
    """

    data_pattern = r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)_(?P<channel>w[1-9]{1})(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
    data_re = re.compile(data_pattern)

    return pd.DataFrame(_list_image_files(root_dir=acquisition_dir, data_re=data_re))


def parse_multi_field_stacks(acquisition_dir: Union[Path, str]) -> pd.DataFrame:
    folder_pattern = r".*[/\\]ZStep_(?P<z>\d+).*"
    filename_pattern = r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)_(?P<channel>w[1-9]{1})(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
    return pd.DataFrame(
        _list_dataset_files(
            root_dir=acquisition_dir,
            root_re=re.compile(folder_pattern),
            filename_re=re.compile(filename_pattern),
        )
    )


def _list_image_files(root_dir: Path, data_re: re.Pattern) -> list[str]:
    files = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            files.extend(_list_image_files(entry, data_re))

        if entry.is_file():
            match = data_re.fullmatch(entry.name)
            if match:
                row = match.groupdict()
                row["path"] = entry.path
                files.append(row)

    return files


def _list_dataset_files(
    root_dir: Path, root_re: re.Pattern, filename_re: re.Pattern
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
                    row["path"] = str(Path(root) / f)
                    files.append(row)
    return files
