from __future__ import annotations

import re
from copy import copy
from decimal import Decimal
from os.path import exists, join
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import pandas as pd
from defusedxml.ElementTree import parse
from tifffile import imread
from tqdm import tqdm

from faim_ipa.hcs.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_ipa.hcs.cellvoyager.tile import StackedTile
from faim_ipa.io.metadata import ChannelMetadata
from faim_ipa.stitching.tile import Tile, TilePosition

if TYPE_CHECKING:
    from pathlib import Path

BTS_NS = "{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}"


class CellVoyagerWellAcquisition(WellAcquisition):
    """
    Data structure for a CellVoyager well acquisition.
    """

    def __init__(
        self,
        files: pd.DataFrame,
        alignment: TileAlignmentOptions,
        metadata: pd.DataFrame,
        background_correction_matrices: dict[str, Path | str] | None = None,
        illumination_correction_matrices: dict[str, Path | str] | None = None,
        n_planes_in_stacked_tile: int = 1,
        dtype: np.dtype | None = None,
    ):
        self._metadata = metadata
        self._z_spacing = self._compute_z_spacing(files)
        self._dtype = dtype if dtype else self._get_dtype(files)
        self._n_planes_in_stacked_tile = n_planes_in_stacked_tile
        super().__init__(
            files=files,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def _get_dtype(self, files: pd.DataFrame) -> np.dtype:
        return imread(files["path"].iloc[0]).dtype

    def _compute_z_spacing(self, files: pd.DataFrame) -> float | None:
        if "ZIndex" in files.columns:
            z_steps = np.array(
                files.astype({"Z": float}).groupby("ZIndex", sort=True).mean("Z")["Z"]
            )

            precision = -Decimal(str(z_steps[0])).as_tuple().exponent
            return np.round(np.mean(np.diff(z_steps)), decimals=precision)
        return None

    def _assemble_tiles(self) -> list[Tile]:
        min_z_index = 0
        max_z_index = min_z_index + 1
        if "ZIndex" in self._files.columns:
            min_z_index = self._files["ZIndex"].min()
            max_z_index = self._files["ZIndex"].max() + 1

        tiles = {}
        for _i, row in self._files.iterrows():
            if "ZIndex" in row:
                tile_z_index = (
                    row["ZIndex"] - min_z_index
                ) // self._n_planes_in_stacked_tile
            else:
                tile_z_index = min_z_index
            tczyx_index = (
                row["TimePoint"],
                row["Ch"],
                tile_z_index,
                row["Y"],
                row["X"],
            )
            if tczyx_index not in tiles:
                tiles[tczyx_index] = [row]
            else:
                tiles[tczyx_index].append(row)

        stacked_tiles = []
        for tczyx_index, rows in tiles.items():
            row_dict = {}
            for r in rows:
                if "ZIndex" in r:
                    row_dict[r["ZIndex"]] = r["path"]
                else:
                    row_dict[min_z_index] = r["path"]

            files = []
            z_start = tczyx_index[2] * self._n_planes_in_stacked_tile + min_z_index
            for z in range(
                z_start, min(z_start + self._n_planes_in_stacked_tile, max_z_index)
            ):
                if z in row_dict:
                    files.append(row_dict[z])
                else:
                    files.append(None)

            time_point = tczyx_index[0]
            channel = tczyx_index[1]
            y, x = tczyx_index[3], tczyx_index[4]

            ch_metadata = self._metadata[self._metadata["Ch"] == channel].iloc[0]
            shape = (
                len(files),
                int(ch_metadata["VerticalPixels"]),
                int(ch_metadata["HorizontalPixels"]),
            )

            yx_spacing = self.get_yx_spacing()

            bgcm = None
            if self._background_correction_matrices is not None:
                bgcm = self._background_correction_matrices[str(channel)]

            icm = None
            if self._illumination_correction_matrices is not None:
                icm = self._illumination_correction_matrices[str(channel)]

            stacked_tiles.append(
                StackedTile(
                    paths=files,
                    shape=shape,
                    position=TilePosition(
                        time=time_point,
                        channel=int(channel),
                        z=z_start,
                        y=int(-float(y) / yx_spacing[0]),
                        x=int(float(x) / yx_spacing[1]),
                    ),
                    background_correction_matrix_path=bgcm,
                    illumination_correction_matrix_path=icm,
                    dtype=self._dtype,
                )
            )

        return stacked_tiles

    def get_axes(self) -> list[str]:
        if self._z_spacing is not None:
            return ["c", "z", "y", "x"]
        return ["c", "y", "x"]

    def get_yx_spacing(self) -> tuple[float, float]:
        ch_metadata = self._metadata.iloc[0]
        return (
            float(ch_metadata["VerticalPixelDimension"]),
            float(ch_metadata["HorizontalPixelDimension"]),
        )

    def get_z_spacing(self) -> float | None:
        return self._z_spacing


class StackAcquisition(PlateAcquisition):
    def __init__(
        self,
        acquisition_dir: Path | str,
        alignment: TileAlignmentOptions,
        background_correction_matrices: dict[str, Path | str] | None = None,
        illumination_correction_matrices: dict[str, Path | str] | None = None,
        n_planes_in_stacked_tile: int = 1,
    ):
        self._n_planes_in_stacked_tile = n_planes_in_stacked_tile
        super().__init__(
            acquisition_dir=acquisition_dir,
            alignment=alignment,
            background_correction_matrices=background_correction_matrices,
            illumination_correction_matrices=illumination_correction_matrices,
        )

    def get_channel_metadata(self) -> dict[int, ChannelMetadata]:
        metadata = self._parse_metadata()
        ch_metadata = {}

        for _i, row in metadata.iterrows():
            index = int(row["Ch"]) - 1
            ch_metadata[index] = ChannelMetadata(
                channel_index=index,
                channel_name=row["Ch"],
                display_color=row["Color"],
                spatial_calibration_x=row["HorizontalPixelDimension"],
                spatial_calibration_y=row["VerticalPixelDimension"],
                spatial_calibration_units="um",
                z_spacing=self.get_z_spacing(),
                wavelength=self.__parse_filter_wavelength(row["Acquisition"]),
                exposure_time=row["ExposureTime"],
                exposure_time_unit="ms",
                objective=row["Objective"],
            )

        assert min(ch_metadata.keys()) == 0, "Channel indices must start at 0."

        return ch_metadata

    def get_z_spacing(self) -> float:
        return self._wells[0].get_z_spacing()

    def _build_well_acquisitions(self, files: pd.DataFrame) -> list[WellAcquisition]:
        dtype = copy(imread(files.iloc[0]["path"]).dtype)
        return [
            CellVoyagerWellAcquisition(
                files=files[files["well"] == well],
                alignment=self._alignment,
                metadata=self._parse_metadata(),
                background_correction_matrices=self._background_correction_matrices,
                illumination_correction_matrices=self._illumination_correction_matrices,
                n_planes_in_stacked_tile=self._n_planes_in_stacked_tile,
                dtype=dtype,
            )
            for well in tqdm(files["well"].unique())
        ]

    @staticmethod
    def __parse_filter_wavelength(value) -> int:
        try:
            return int(re.match(r"BP(\d+)/", value).group(1))
        except AttributeError:  # no cov
            return 0

    def _parse_metadata(self) -> pd.DataFrame:
        mrf_file = join(self._acquisition_dir, "MeasurementDetail.mrf")
        if not exists(mrf_file):
            msg = f"MeasurementDetail.mrf not found in: {self._acquisition_dir}"
            raise ValueError(msg)
        mrf_tree = parse(mrf_file)
        mrf_root = mrf_tree.getroot()

        channels = []
        for channel in mrf_root.findall(BTS_NS + "MeasurementChannel"):
            row = {
                key.replace(BTS_NS, ""): value for key, value in channel.attrib.items()
            }
            channels.append(row)

        mes_file = join(
            self._acquisition_dir,
            mrf_root.attrib[BTS_NS + "MeasurementSettingFileName"],
        )
        if not exists(mes_file):
            msg = f"Settings file not found: {mes_file}"
            raise ValueError(msg)
        mes_tree = parse(mes_file)
        mes_root = mes_tree.getroot()

        channel_settings = []
        for channel in mes_root.find(BTS_NS + "ChannelList").findall(
            BTS_NS + "Channel"
        ):
            row = {
                key.replace(BTS_NS, ""): value for key, value in channel.attrib.items()
            }
            channel_settings.append(row)

        return pd.merge(
            pd.DataFrame(channels),
            pd.DataFrame(channel_settings),
            left_on="Ch",
            right_on="Ch",
        )

    def _parse_files(self) -> pd.DataFrame:
        mlf_file = join(self._acquisition_dir, "MeasurementData.mlf")
        if not exists(mlf_file):
            msg = f"MeasurementData.mlf not found in: {self._acquisition_dir}"
            raise ValueError(msg)
        mlf_tree = parse(mlf_file)
        mlf_root = mlf_tree.getroot()

        files = []
        for record in mlf_root.findall(BTS_NS + "MeasurementRecord"):
            row = {
                key.replace(BTS_NS, ""): value for key, value in record.attrib.items()
            }
            if row.pop("Type") == "IMG":
                row |= {
                    "path": join(self._acquisition_dir, record.text),
                    "well": chr(ord("@") + int(row.pop("Row")))
                    + row.pop("Column").zfill(2),
                }
                files.append(row)

            record.clear()

        files = pd.DataFrame(files)
        files["TimePoint"] = files["TimePoint"].astype(int)
        files["ZIndex"] = files["ZIndex"].astype(int)
        return files


class ZAdjustedStackAcquisition(StackAcquisition):
    _trace_log_files: list[str | Path]

    def __init__(
        self,
        acquisition_dir: Path | str,
        trace_log_files: list[Path | str],
        alignment: TileAlignmentOptions,
        background_correction_matrices: dict[str, Path | str] | None = None,
        illumination_correction_matrices: dict[str, Path | str] | None = None,
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

    def _parse_files(self) -> pd.DataFrame:
        files = super()._parse_files()
        z_mapping = self.create_z_mapping(self._trace_log_files, self._acquisition_dir)
        # merge files left with mapping on path
        merged = files.merge(z_mapping, how="left", left_on=["path"], right_on=["path"])
        if np.any(merged["z_pos"].isna()):
            msg = "At least one invalid z position."
            raise ValueError(msg)
        min_z = np.min(merged["z_pos"].astype(float))
        z_spacing = np.mean(
            merged[merged["ZIndex"].astype(int) == 2]["Z"].astype(float)
        ) - np.mean(merged[merged["ZIndex"].astype(int) == 1]["Z"].astype(float))
        # Shift ZIndex for each field in each well according to the auto-focus value
        for key, selection in tqdm(
            merged.groupby(["well", "FieldIndex", "Ch"]), desc="Adjust Z", leave=False
        ):
            z_index_offset = int(
                np.round((np.min(selection["z_pos"].astype(float)) - min_z) / z_spacing)
            )
            merged.loc[
                (merged["well"] == key[0])
                & (merged["FieldIndex"] == key[1])
                & (merged["Ch"] == key[2]),
                "ZIndex",
            ] += z_index_offset

        # Start at 0
        merged["ZIndex"] = merged["ZIndex"] - merged["ZIndex"].min()
        # update Z
        merged["Z"] = merged["z_pos"]
        return merged

    @staticmethod
    def create_z_mapping(
        trace_log_files: list[str | Path], acquisition_dir: Path | str | None = None
    ) -> pd.DataFrame:
        z_pos = []
        filenames = []
        missing = []
        value = None
        for trace_file in trace_log_files:
            with open(trace_file) as log:
                for line in log:
                    tokens = line.split(",")
                    if (
                        (len(tokens) > 14)
                        and (tokens[7] == "--->")
                        and (tokens[8] == "MS_MANU")
                    ):
                        value = float(tokens[14])
                    elif (
                        (len(tokens) > 12)
                        and (tokens[7] == "--->")
                        and (tokens[8] == "AF_MANU")
                        and (tokens[9] == "34")
                    ):
                        value = float(tokens[12])
                    elif (
                        (len(tokens) > 8)
                        and (tokens[4] == "Measurement")
                        and (tokens[7] == "_init_frame_save")
                    ):
                        filename = tokens[8]
                        if value is None:
                            missing.append(
                                join(acquisition_dir, filename)
                                if acquisition_dir
                                else filename
                            )
                        else:
                            filenames.append(
                                join(acquisition_dir, filename)
                                if acquisition_dir
                                else filename
                            )
                            z_pos.append(value)
                    elif (
                        (len(tokens) > 7)
                        and (tokens[6] == "EndPeriod")
                        and (tokens[7] == "acquire frames")
                    ):
                        value = None

        if len(missing) > 0:
            warn("Z position information missing for some files.", stacklevel=2)
            warn(
                f"First file without z position information: {missing[0]}", stacklevel=2
            )
        return pd.DataFrame(
            {
                "path": filenames,
                "z_pos": z_pos,
            }
        )
