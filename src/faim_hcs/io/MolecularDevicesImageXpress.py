# SPDX-FileCopyrightText: 2023 Friedrich Miescher Institute for Biomedical Research (FMI), Basel (Switzerland)
#
# SPDX-License-Identifier: BSD-3-Clause

"""Methods to parse image files acquired with a MolecularDevices ImageXpress system."""

import os
import re
from pathlib import Path
from typing import Union

import pandas as pd

_METASERIES_FILENAME_PATTERN = r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)_(?P<channel>w[1-9]{1})(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
_METASERIES_FOLDER_PATTERN = r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?:[\/\\]ZStep_(?P<z>\d+))?.*"
_METASERIES_MAIN_FOLDER_PATTERN = (
    r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)(?![\/\\]ZStep_.*)"
)
_METASERIES_ZSTEP_FOLDER_PATTERN = (
    r".*[\/\\](?P<date>\d{4}-\d{2}-\d{2})[\/\\](?P<acq_id>\d+)[\/\\]ZStep_(?P<z>\d+).*"
)


def parse_single_plane_multi_fields(acquisition_dir: Union[Path, str]) -> pd.DataFrame:
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

    :param acquisition_dir: Path to acquisition directory.
    :return: table of all acquired image data.
    """
    return parse_files(acquisition_dir=acquisition_dir, mode="top-level")


def parse_multi_field_stacks(acquisition_dir: Union[Path, str]) -> pd.DataFrame:
    """Parse ZStep folders of an acquisition of a Molecular Devices ImageXpress Micro Confocal system.

    Storage hierarchy on disk for 2 wells with 2 fields and 2 channels::

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

    :param acquisition_dir: Path to acquisition directory.
    :return: table of all acquired image data.
    """
    return parse_files(acquisition_dir=acquisition_dir, mode="z-steps")


def parse_files(acquisition_dir: Union[Path, str], mode: str = "all") -> pd.DataFrame:
    """Parse any multi-field acquisition of a Molecular Devices ImageXpress Micro Confocal system.

    Storage layout on disk::

        Experiment --> {name} [Optional]
        └── 2023-02-22 --> {date}
            └── 1099 --> {acquisition id}
                ├── ZStep_1
                TODO fix file tree

    Image data is stored in {name}_{well}_{field}_w{channel}{md_id}.tif.
    The *_thumb*.tif files, used by Molecular Devices as preview, are ignored.

    :param acquisition_dir: Path to acquisition directory.
    :param mode: whether to parse 'top-level' file only, 'z-steps' files only, or 'all' (default).
    :return: table of all acquired image data.
    """
    if mode == "top-level":
        root_pattern = _METASERIES_MAIN_FOLDER_PATTERN
    elif mode == "z-steps":
        root_pattern = _METASERIES_ZSTEP_FOLDER_PATTERN
    else:
        root_pattern = _METASERIES_FOLDER_PATTERN
    return pd.DataFrame(
        _list_dataset_files(
            root_dir=acquisition_dir,
            root_re=re.compile(root_pattern),
            filename_re=re.compile(_METASERIES_FILENAME_PATTERN),
        )
    )


def _list_dataset_files(
    root_dir: Union[Path, str], root_re: re.Pattern, filename_re: re.Pattern
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
