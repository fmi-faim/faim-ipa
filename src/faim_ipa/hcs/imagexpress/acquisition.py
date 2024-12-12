import os
import re
from abc import abstractmethod
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from tqdm import tqdm

from faim_ipa.hcs.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_ipa.io.metadata import ChannelMetadata
from faim_ipa.io.metaseries import load_metaseries_tiff_metadata
from faim_ipa.stitching.tile import Tile, TilePosition
from faim_ipa.utils import rgb_to_hex, wavelength_to_rgb


class ImageXpressWellAcquisition(WellAcquisition):
    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        z_spacing: float | None,
        background_correction_matrices: dict[str, Path | str] | None = None,
        illumination_correction_matrices: dict[str, Path | str] | None = None,
    ) -> None:
        self._z_spacing = z_spacing
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _assemble_tiles(self) -> list[Tile]:
        tiles = []
        for _i, row in self._files.iterrows():
            file = row["path"]
            time_point = row["t"] if "t" in row.index and row["t"] is not None else 0
            channel = row["channel"]
            metadata = load_metaseries_tiff_metadata(file)
            z = (
                1
                if self._z_spacing is None
                else row["z"] if row["z"] is not None else 1
            )

            bgcm = None
            if self._background_correction_matrices is not None:
                bgcm = self._background_correction_matrices[channel]

            icm = None
            if self._illumination_correction_matrices is not None:
                icm = self._illumination_correction_matrices[channel]

            tiles.append(
                Tile(
                    path=file,
                    shape=(metadata["pixel-size-y"], metadata["pixel-size-x"]),
                    position=TilePosition(
                        time=time_point,
                        channel=int(channel[1:]),
                        z=z,
                        y=int(
                            metadata["stage-position-y"]
                            / metadata["spatial-calibration-y"]
                        ),
                        x=int(
                            metadata["stage-position-x"]
                            / metadata["spatial-calibration-x"]
                        ),
                    ),
                    background_correction_matrix_path=bgcm,
                    illumination_correction_matrix_path=icm,
                )
            )
        return tiles

    def get_yx_spacing(self) -> tuple[float, float]:
        metadata = load_metaseries_tiff_metadata(self._files.iloc[0]["path"])
        return (metadata["spatial-calibration-y"], metadata["spatial-calibration-x"])

    def get_z_spacing(self) -> float | None:
        return self._z_spacing

    def get_axes(self) -> list[str]:
        axes = ["y", "x"]

        if "z" in self._files.columns:
            axes = ["z", *axes]

        if self._files["channel"].nunique() > 1:
            axes = ["c", *axes]

        if "t" in self._files.columns and self._files["t"].nunique() > 1:
            axes = ["t", *axes]

        return axes


class ImageXpressPlateAcquisition(PlateAcquisition):
    def __init__(
        self,
        acquisition_dir: Path | str,
        alignment: TileAlignmentOptions,
        background_correction_matrices: dict[str, Path | str] | None = None,
        illumination_correction_matrices: dict[str, Path | str] | None = None,
    ):
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _parse_files(self) -> pd.DataFrame:
        """Parse all files in the acquisition directory.

        Returns
        -------
        DataFrame
            Table of all files in the acquisition.
        """
        files = pd.DataFrame(
            ImageXpressPlateAcquisition._list_and_match_files(
                root_dir=self._acquisition_dir,
                root_re=self._get_root_re(),
                filename_re=self._get_filename_re(),
            )
        )

        # handle ImageXpress exported files:
        # MIPs are stored in ZStep_0 -> change 0 to None
        if "z" in files.columns:
            files.loc[files.z == "0", "z"] = None

        return files

    @staticmethod
    def _list_and_match_files(
        root_dir: Path | str,
        root_re: re.Pattern,
        filename_re: re.Pattern,
    ) -> list[list[dict[dict, str]]]:
        files = []
        for root, _, filenames in os.walk(root_dir):
            m_root = root_re.fullmatch(root)
            if m_root:
                for f in filenames:
                    m_filename = filename_re.fullmatch(f)
                    if m_filename:
                        row = m_root.groupdict()
                        row |= m_filename.groupdict()
                        if "channel" not in row or row["channel"] is None:
                            row["channel"] = "w1"
                        row["path"] = str(Path(root).joinpath(f))
                        files.append(row)
        return files

    @abstractmethod
    def _get_root_re(self) -> re.Pattern:
        """Regular expression for matching the root directory of the acquisition."""
        raise NotImplementedError

    @abstractmethod
    def _get_filename_re(self) -> re.Pattern:
        """Regular expression for matching the filename of the acquisition."""
        raise NotImplementedError

    def _build_well_acquisitions(self, files: pd.DataFrame) -> list[WellAcquisition]:
        return [
            ImageXpressWellAcquisition(
                files=files[files["well"] == well],
                alignment=self._alignment,
                z_spacing=self._get_z_spacing(),
                background_correction_matrices=self._background_correction_matrices,
                illumination_correction_matrices=self._illumination_correction_matrices,
            )
            for well in tqdm(files["well"].unique())
        ]

    @abstractmethod
    def _get_z_spacing(self) -> float | None:
        raise NotImplementedError

    def get_channel_metadata(self) -> dict[int, ChannelMetadata]:
        ch_metadata = {}
        _files = self._wells[0]._files
        for ch in _files["channel"].unique():
            channel_files = _files[_files["channel"] == ch]
            path = channel_files["path"].iloc[0]
            metadata = load_metaseries_tiff_metadata(path=path)
            index = int(ch[1:]) - 1
            if "Z Projection Method" in metadata:
                name = (
                    f"{metadata['Z Projection Method'].replace(' ', '-')}-Projection_"
                    f"{metadata['_IllumSetting_']}"
                )
            else:
                name = metadata["_IllumSetting_"]
            ch_metadata[index] = ChannelMetadata(
                channel_index=index,
                channel_name=name,
                display_color=rgb_to_hex(*wavelength_to_rgb(metadata["wavelength"])),
                spatial_calibration_x=metadata["spatial-calibration-x"],
                spatial_calibration_y=metadata["spatial-calibration-y"],
                spatial_calibration_units=metadata["spatial-calibration-units"],
                z_spacing=self._get_z_spacing(),
                wavelength=metadata["wavelength"],
                exposure_time=float(metadata["Exposure Time"].split(" ")[0]),
                exposure_time_unit=metadata["Exposure Time"].split(" ")[1],
                objective=metadata["_MagSetting_"],
            )

        assert min(ch_metadata.keys()) == 0, "Channel indices must start at 0."
        return ch_metadata


class SinglePlaneAcquisition(ImageXpressPlateAcquisition):
    """Parse top folder (single planes) of an acquisition of a MolecularDevices ImageXpress Micro Confocal system.

    Storage layout on disk for 2 wells with 2 fields and 2 channels:

    Option A (as stored by Microscope):

    MIP-2P-2sub --> {name} [Optional]
    └── 2022-07-05 --> {date} [Optional]
        └── 1075 --> {acquisition id}
            └── Timepoint_1 --> {t} [Optional]
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

    Option B (as exported via software):

    test_Plate_3420 --> {name}_Plate_{acquisition id}
    └── Timepoint_1 --> {t} [Optional]
        ├── test_C05_s1_w1.TIF
        ├── test_C05_s1_w2.TIF
        ├── test_C05_s2_w1.TIF
        ├── test_C05_s2_w2.TIF
        ├── test_C06_s1_w1.TIF
        ├── test_C06_s1_w2.TIF
        ├── test_C06_s2_w1.TIF
        └── test_C06_s2_w2.TIF

    Image data is stored in {name}_{well}_{field}_w{channel}.TIF.
    """

    def __init__(
        self,
        acquisition_dir: Path | str,
        alignment: TileAlignmentOptions,
        background_correction_matrices: dict[str, Path | str] | None = None,
        illumination_correction_matrices: dict[str, Path | str] | None = None,
    ):
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _get_root_re(self) -> re.Pattern:
        return re.compile(
            r".*(?:[\/\\](?P<date>\d{4}-\d{2}-\d{2}))?[\/\\](?:(?P<plate_name>.*)_Plate_)?(?P<acq_id>\d+)(?:[\/\\]TimePoint_(?P<t>\d+))?(?:[\/\\]ZStep_0)?"
        )

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_?(?P<field>s\d+)?_?(?P<channel>w[1-9]{1})?(?!_thumb)(?P<md_id>[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12})?(?P<ext>\.(?i:tif))"
        )

    def _get_z_spacing(self) -> float | None:
        return None


class StackAcquisition(ImageXpressPlateAcquisition):
    """Image stack acquisition with a Molecular Devices ImageXpress Micro
    Confocal system.

    Storage layout on disk for 2 wells with 2 fields, 4 channels and 9 z-steps:

    OPTION A (as stored by Microscope):

    MIP-2P-2sub-Stack --> {name} [Optional]
    └── 2023-02-21 --> {date} [Optional]
        └── 1334 --> {acquisition id}
            └── Timepoint_1 --> {t} [Optional]
                ├── ZStep_1 --> {z}
                │   ├── Projection-Mix_E07_s1_w1E78EB128-BD0D-4D94-A6AD-3FF28BB1B105.tif
                │   ├── Projection-Mix_E07_s1_w1_thumb187DE64B-038A-4671-BF6B-683721723769.tif
                │   ├── Projection-Mix_E07_s1_w2C0A49256-E289-4C0F-ADC9-F7728ABDB141.tif
                │   ├── Projection-Mix_E07_s1_w2_thumb57D4B151-71BF-480E-8CC4-C23A2690B763.tif
                │   ├── Projection-Mix_E07_s1_w427CCB2E4-1BF4-45E7-8BC7-264B48EF9C4A.tif
                │   ├── Projection-Mix_E07_s1_w4_thumb555647D0-77F1-4A43-9472-AE509F95E236.tif
                │   ├── ...
                │   └── Projection-Mix_E08_s2_w4_thumbD2785594-4F49-464F-9F80-1B82E30A560A.tif
                ├── ...
                └── ZStep_9
                    ├── Projection-Mix_E07_s1_w1091EB8A5-272A-466D-B8A0-7547C6BA392B.tif
                    ├── ...
                    └── Projection-Mix_E08_s2_w2_thumb210C0D5D-C20E-484D-AFB2-EFE669A56B84.tif

    Image data is stored in {name}_{well}_{field}_w{channel}{md_id}.tif.
    The *_thumb*.tif files, used by Molecular Devices as preview, are ignored.

    OPTION B (as exported via software):

    test_Plate_3433 --> {name}_Plate_{acquisition id}
    └── Timepoint_1 --> {t} [Optional]
        ├── ZStep_0 (contains MIPs)
            └── ...
        ├── ZStep_1 --> {z}
        │   ├── test_E07_s1_w1.TIF
        │   ├── test_E07_s1_w2.TIF
        │   ├── test_E07_s1_w3.TIF
        │   ├── test_E07_s1_w4.TIF
        │   ├── test_E07_s2_w1.TIF
        │   ├── ...
        │   └── test_E08_s2_w4.TIF
        ├── ...
        └── ZStep_9
            ├── test_E07_s1_w1.TIF
            ├── ...
            └── test_E08_s2_w4.TIF

    Image data is stored in {name}_{well}_{field}_w{channel}.TIF.
    ZStep_0 contains MIPs and is ignored.
    """

    _z_spacing: float = None

    def __init__(
        self,
        acquisition_dir: Path | str,
        alignment: TileAlignmentOptions,
        background_correction_matrices: dict[str, Path | str] | None = None,
        illumination_correction_matrices: dict[str, Path | str] | None = None,
    ):
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _parse_files(self) -> pd.DataFrame:
        files = super()._parse_files()
        self._z_spacing = self._compute_z_spacing(files)
        return files

    def _get_root_re(self) -> re.Pattern:
        return re.compile(
            r".*(?:[\/\\](?P<date>\d{4}-\d{2}-\d{2}))?[\/\\](?:(?P<plate_name>.*)_Plate_)?(?P<acq_id>\d+)(?:[\/\\]TimePoint_(?P<t>\d+))?(?:[\/\\]ZStep_(?P<z>[1-9]\d*))"
        )

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_?(?P<field>s\d+)?_?(?P<channel>w[1-9]{1})?(?!_thumb)(?P<md_id>[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12})?(?P<ext>\.(?i:tif))"
        )

    def _get_z_spacing(self) -> float | None:
        return self._z_spacing

    def _compute_z_spacing(self, files: pd.DataFrame) -> float | None:
        assert "z" in files.columns, "No z column in files DataFrame."
        # check if files were exported via software
        if files.plate_name.iloc[0] is not None:
            # single planes and MIPs are duplicated to the entire stack
            # -> need to check in metadata, if the channel contains a stack or not
            for c in np.sort(files.channel.unique()):
                file = files[(files.channel == c) & (files.z == "2")].iloc[0]
                metadata = load_metaseries_tiff_metadata(file.path)
                # projections have no "Z Step" attribute
                if "Z Step" in metadata.keys():
                    # single-planes have metadata["Z Step"] == 1, for each duplicate
                    if metadata["Z Step"] == 2:
                        channel_with_stack = c
                        break
        else:
            channel_with_stack = np.sort(files[files["z"] == "2"]["channel"].unique())[
                0
            ]
        subset = files[files["channel"] == channel_with_stack]
        subset = subset[subset["well"] == np.sort(subset["well"].unique())[0]]
        first_field = np.sort(subset["field"].unique())[0]
        if first_field is not None:
            subset = subset[subset["field"] == np.sort(subset["field"].unique())[0]]

        plane_positions = []

        for _i, row in subset.iterrows():
            file = row["path"]
            if "z" in row and row["z"] is not None:
                metadata = load_metaseries_tiff_metadata(file)
                z_position = metadata["stage-position-z"]
                plane_positions.append(z_position)

        plane_positions = np.array(sorted(plane_positions), dtype=np.float32)

        precision = -Decimal(str(plane_positions[0])).as_tuple().exponent
        return np.round(np.mean(np.diff(plane_positions)), decimals=precision)


class MixedAcquisition(StackAcquisition):
    """Image stack acquisition with Projectsion acquired with a Molecular
    Devices ImageXpress Micro Confocal system.

    OPTION A (as stored by Microscope):

    MIP-2P-2sub-Stack --> {name} [Optional]
    └── 2023-02-21 --> {date}
        └── 1334 --> {acquisition id}
            ├── Projection-Mix_E07_s1_w1E94C24BD-45E4-450A-9919-257C714278F7.tif
            ├── Projection-Mix_E07_s1_w1_thumb4BFD4018-E675-475E-B5AB-2E959E6B6DA1.tif
            ├── ...
            ├── Projection-Mix_E08_s2_w3CCE83D85-0912-429E-9F18-716A085BB5BC.tif
            ├── Projection-Mix_E08_s2_w3_thumb4D88636E-181E-4AF6-BC53-E7A435959C8F.tif
            ├── ZStep_1
            │   ├── Projection-Mix_E07_s1_w1E78EB128-BD0D-4D94-A6AD-3FF28BB1B105.tif
            │   ├── Projection-Mix_E07_s1_w1_thumb187DE64B-038A-4671-BF6B-683721723769.tif
            │   ├── Projection-Mix_E07_s1_w2C0A49256-E289-4C0F-ADC9-F7728ABDB141.tif
            │   ├── Projection-Mix_E07_s1_w2_thumb57D4B151-71BF-480E-8CC4-C23A2690B763.tif
            │   ├── Projection-Mix_E07_s1_w427CCB2E4-1BF4-45E7-8BC7-264B48EF9C4A.tif
            │   ├── Projection-Mix_E07_s1_w4_thumb555647D0-77F1-4A43-9472-AE509F95E236.tif
            │   ├── ...
            │   └── Projection-Mix_E08_s2_w4_thumbD2785594-4F49-464F-9F80-1B82E30A560A.tif
            ├── ...
            └── ZStep_9
                ├── Projection-Mix_E07_s1_w1091EB8A5-272A-466D-B8A0-7547C6BA392B.tif
                ├── ...
                └── Projection-Mix_E08_s2_w2_thumb210C0D5D-C20E-484D-AFB2-EFE669A56B84.tif

    Image data is stored in {name}_{well}_{field}_w{channel}{md_id}.tif.
    The *_thumb*.tif files, used by Molecular Devices as preview, are ignored.

    OPTION B (as exported via software) will always export mixed acquisitions
    as stacks. -> Use StackAcquisition instead.
    """

    def __init__(
        self,
        acquisition_dir: Path | str,
        alignment: TileAlignmentOptions,
        background_correction_matrix: dict[str, NDArray] | None = None,
        illumination_correction_matrix: NDArray | None = None,
    ):
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrix,
            illumination_correction_matrices=illumination_correction_matrix,
        )

    def _parse_files(self) -> pd.DataFrame:
        files = self._filter_mips(super()._parse_files())
        if files.plate_name.iloc[0] is not None:
            raise ValueError(
                "Data was exported via software. "
                "MixedAcquisition is not applicable in this case. "
                "Use StackAcquisition instead."
            )
        self._z_spacing = self._compute_z_spacing(files)
        return files

    def _filter_mips(self, files: pd.DataFrame) -> pd.DataFrame:
        """Remove MIP files if the whole stack was acquired."""
        _files = files.copy()
        for ch in _files["channel"].unique():
            channel_files = _files[_files["channel"] == ch]
            z_positions = channel_files["z"].unique()
            has_mip = None in z_positions
            has_stack = len(z_positions) > 1
            if has_mip and has_stack:
                _files.drop(
                    _files[(_files["channel"] == ch) & (_files["z"].isna())].index,
                    inplace=True,
                )
        return _files

    def _get_root_re(self) -> re.Pattern:
        return re.compile(
            r".*(?:[\/\\](?P<date>\d{4}-\d{2}-\d{2}))?[\/\\](?:(?P<plate_name>.*)_Plate_)?(?P<acq_id>\d+)(?:[\/\\]TimePoint_(?P<t>\d+))?(?:[\/\\]ZStep_(?P<z>[1-9]\d*))?.*"
        )

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_?(?P<field>s\d+)?_?(?P<channel>w[1-9]{1})?(?!_thumb)(?P<md_id>[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12})?(?P<ext>\.(?i:tif))"
        )
