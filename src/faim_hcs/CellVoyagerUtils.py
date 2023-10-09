from decimal import Decimal
from typing import Callable, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from tifffile.tifffile import imread

from faim_hcs.MetaSeriesUtils import (
    montage_stage_pos_image_YX,  # FIXME factor out into common file
)
from faim_hcs.UIntHistogram import UIntHistogram


def _build_ch_metadata(source: pd.DataFrame, channel: str):
    ch_meta = source.set_index("Ch")
    metadata = {
        "wavelength": ch_meta.at[channel, "Target"],
        # "power": power,
        "exposure-time": float(ch_meta.at[channel, "ExposureTime"]),
        "exposure-time-unit": "ms",  # hard-coded
        # "shading-correction": metaseries_ch_metadata["ShadingCorrection"] == "On",
        "channel-name": ch_meta.at[channel, "Target"],
        # "objective-numerical-aperture": metaseries_ch_metadata["_MagNA_"],
        "objective": ch_meta.at[channel, "Objective"],
        "display-color": ch_meta.at[channel, "Color"][-6:],
    }
    return metadata


def _build_px_metadata(source: pd.DataFrame, channel: str, dtype: np.dtype):
    ch_meta = source.set_index("Ch")
    return {
        "pixel-type": dtype,
        "spatial-calibration-units": "um",
        "spatial-calibration-x": float(ch_meta.at[channel, "HorizontalPixelDimension"]),
        "spatial-calibration-y": float(ch_meta.at[channel, "VerticalPixelDimension"]),
    }


def build_stack(imgs: list):
    # Build zero-stack
    stack = None
    for img in imgs:
        if img is not None:
            shape = img.shape
            stack = np.zeros((len(imgs), *shape), dtype=img.dtype)
            break

    # Fill in existing data
    for i, img in enumerate(imgs):
        if img is not None:
            stack[i] = img

    return stack


def compute_z_sampling(ch_z_positions: list[list[Union[None, float]]]):
    z_samplings = []
    for z_positions in ch_z_positions:
        if z_positions is not None and None not in z_positions:
            precision = -Decimal(str(z_positions[0])).as_tuple().exponent
            z_step = np.round(np.mean(np.diff(z_positions)), decimals=precision)
            z_samplings.append(z_step)

    return np.mean(z_samplings)


def get_well_image_CZYX(
    well_files: pd.DataFrame,
    channel_metadata_source: pd.DataFrame,
    channels: list[str],
    assemble_fn: Callable = montage_stage_pos_image_YX,
) -> tuple[ArrayLike, list[UIntHistogram], list[dict], dict, dict]:
    """Assemble image data for the given well-files."""
    planes = well_files["ZIndex"].unique()

    stacks = []
    channel_histograms = []
    ch_metadata = []
    z_positions = []
    roi_tables = {}

    for ch in channels:
        channel_files = well_files[well_files["Ch"] == ch]
        if len(channel_files) > 0:
            plane_imgs = []
            z_plane_positions = []
            for z in sorted(planes, key=int):
                plane_files = channel_files[channel_files["ZIndex"] == z]
                if len(plane_files) > 0:
                    img, z_pos, roi_tables = get_img_YX(
                        assemble_fn=assemble_fn,
                        files=plane_files,
                        channel_metadata=channel_metadata_source,
                    )
                    plane_imgs.append(img)
                    z_plane_positions.append(z_pos)
                else:
                    plane_imgs.append(None)
                    z_plane_positions.append(None)
            zyx = build_stack(plane_imgs)
            stacks.append(zyx)
            channel_histograms.append(UIntHistogram(zyx))
            ch_metadata.append(
                _build_ch_metadata(source=channel_metadata_source, channel=ch)
            )
            z_positions.append(z_plane_positions)
        else:
            stacks.append(None)
            channel_histograms.append(UIntHistogram())
            ch_metadata.append(
                {
                    "channel-name": "empty",
                    "display-color": "000000",
                }
            )
            z_positions.append(None)

    czyx = build_stack(stacks)

    px_metadata = _build_px_metadata(
        source=channel_metadata_source,
        channel=ch,
        dtype=czyx.dtype,
    )

    z_sampling = compute_z_sampling(z_positions)
    px_metadata["z-scaling"] = z_sampling

    return czyx, channel_histograms, ch_metadata, px_metadata, roi_tables


def get_well_image_CYX(
    well_files: pd.DataFrame,
    channel_metadata_source: pd.DataFrame,
    channels: list[str],
    assemble_fn: Callable = montage_stage_pos_image_YX,
) -> tuple[ArrayLike, list[UIntHistogram], list[dict], dict, dict]:
    """Assemble image data for the given well files.

    For each channel, a single 2D image is computed. If the well has multiple
    fields per channel, the `assemble_fn` has to montage or stitch the fields
    accordingly.

    :param well_files: all files corresponding to the well
    :param channels: list of required channels
    :param assemble_fn: creates a single image for each channel
    :return: CYX image, channel-histograms, channel-metadata, general-metadata,
                roi-tables dictionary
    """
    channel_imgs = {}
    channel_histograms = {}
    channel_metadata = {}
    roi_tables = {}

    for ch in channels:
        channel_files = well_files[well_files["Ch"] == ch]
        if len(channel_files) > 0:
            img, _, roi_tables = get_img_YX(
                assemble_fn=assemble_fn,
                files=channel_files,
                channel_metadata=channel_metadata_source,
            )
            channel_imgs[ch] = img
            channel_histograms[ch] = UIntHistogram(img)
            channel_metadata[ch] = _build_ch_metadata(channel_metadata_source, ch)

    cyx = np.zeros((len(channels), *img.shape), dtype=img.dtype)

    channel_hists = []
    channel_meta = []
    for i, ch in enumerate(channels):
        if ch in channel_imgs.keys():
            cyx[i] = channel_imgs[ch]
            channel_hists.append(channel_histograms[ch])
            channel_meta.append(channel_metadata[ch])
        else:
            channel_hists.append(UIntHistogram())
            channel_meta.append(
                {
                    "channel-name": "empty",
                    "display-color": "000000",
                }
            )

    px_metadata = _build_px_metadata(
        source=channel_metadata_source,
        channel=ch,
        dtype=cyx.dtype,
    )

    return cyx, channel_hists, channel_meta, px_metadata, roi_tables


def get_img_YX(assemble_fn, files: pd.DataFrame, channel_metadata: pd.DataFrame):
    """Assemble single 2D image for all fields.

    Assumes that all files are from a single channel.

    img: 2D pixel array (yx)
    z_position
    roi_tables: dict[DataFrame]
    """
    channel = np.unique(files["Ch"])
    if len(channel) != 1:
        raise ValueError(
            f"get_img_YX requires files for a single channel only, got: {channel}"
        )
    channel = channel[0]

    z_position = np.mean(files["Z"].astype(float))  # assumed to be all same
    ch_meta = channel_metadata.set_index("Ch")
    general_metadata = {
        "spatial-calibration-x": float(ch_meta.at[channel, "HorizontalPixelDimension"]),
        "spatial-calibration-y": float(ch_meta.at[channel, "VerticalPixelDimension"]),
        "spatial-calibration-units": "um",  # hard-coded unit
        "pixel-type": None,
    }

    imgs = []
    for file_row in files.sort_values(by="path").itertuples():
        img = imread(file_row.path)
        # img = da.from_zarr(imread(file_row.path, aszarr=True), chunks=(-1,-1))
        if general_metadata["pixel-type"] is None:
            general_metadata["pixel-type"] = img.dtype
        img_metadata = {
            "stage-position-x": float(file_row.X),
            "stage-position-y": float(file_row.Y),
            "spatial-calibration-x": general_metadata["spatial-calibration-x"],
            "spatial-calibration-y": general_metadata["spatial-calibration-y"],
            "stage-label": "F"
            + file_row.FieldIndex,  # TODO do we get a meaningful name from CV7000/CV8000?
        }
        imgs.append((img, img_metadata))
    img, roi_tables = assemble_fn(imgs)
    # return general_metadata, img, metadata, z_position, roi_tables
    return img, z_position, roi_tables
