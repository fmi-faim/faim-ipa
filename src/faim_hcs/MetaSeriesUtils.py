from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from numpy._typing import ArrayLike

from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff
from faim_hcs.UIntHistogram import UIntHistogram
from faim_hcs.utils import rgb_to_hex, wavelength_to_rgb


def _build_ch_metadata(metaseries_ch_metadata: dict):
    """Build channel metadata from metaseries metadata."""

    def get_wavelength_power(channel):
        if channel["Lumencor Cyan Intensity"] > 0.0:
            wl = "cyan"
            power = channel["Lumencor Cyan Intensity"]
            assert channel["Lumencor Green Intensity"] == 0.0
            assert channel["Lumencor Red Intensity"] == 0.0
            assert channel["Lumencor Violet Intensity"] == 0.0
            assert channel["Lumencor Yellow Intensity"] == 0.0
            return wl, power
        if channel["Lumencor Green Intensity"] > 0.0:
            wl = "green"
            power = channel["Lumencor Green Intensity"]
            assert channel["Lumencor Cyan Intensity"] == 0.0
            assert channel["Lumencor Red Intensity"] == 0.0
            assert channel["Lumencor Violet Intensity"] == 0.0
            assert channel["Lumencor Yellow Intensity"] == 0.0
            return wl, power
        if channel["Lumencor Red Intensity"] > 0.0:
            wl = "red"
            power = channel["Lumencor Red Intensity"]
            assert channel["Lumencor Cyan Intensity"] == 0.0
            assert channel["Lumencor Green Intensity"] == 0.0
            assert channel["Lumencor Violet Intensity"] == 0.0
            assert channel["Lumencor Yellow Intensity"] == 0.0
            return wl, power
        if channel["Lumencor Violet Intensity"] > 0.0:
            wl = "violet"
            power = channel["Lumencor Violet Intensity"]
            assert channel["Lumencor Cyan Intensity"] == 0.0
            assert channel["Lumencor Green Intensity"] == 0.0
            assert channel["Lumencor Red Intensity"] == 0.0
            assert channel["Lumencor Yellow Intensity"] == 0.0
            return wl, power
        if channel["Lumencor Yellow Intensity"] > 0.0:
            wl = "yellow"
            power = channel["Lumencor Yellow Intensity"]
            assert channel["Lumencor Cyan Intensity"] == 0.0
            assert channel["Lumencor Green Intensity"] == 0.0
            assert channel["Lumencor Red Intensity"] == 0.0
            assert channel["Lumencor Violet Intensity"] == 0.0
            return wl, power
        return None, None

    def get_exposure_time_unit(ch):
        time, unit = ch["Exposure Time"].split(" ")
        time = float(time)
        return time, unit

    wavelength, power = get_wavelength_power(metaseries_ch_metadata)
    time, unit = get_exposure_time_unit(metaseries_ch_metadata)
    display_color = rgb_to_hex(*wavelength_to_rgb(metaseries_ch_metadata["wavelength"]))
    return {
        "wavelength": wavelength,
        "power": power,
        "exposure-time": time,
        "exposure-time-unit": unit,
        "shading-correction": metaseries_ch_metadata["ShadingCorrection"] == "On",
        "channel-name": metaseries_ch_metadata["_IllumSetting_"],
        "objective-NA": metaseries_ch_metadata["_MagNA_"],
        "objective": metaseries_ch_metadata["_MagSetting_"],
        "display-color": display_color,
    }


def _z_metadata(metaseries_ch_metadata: dict):
    return {"z-position": metaseries_ch_metadata["z-position"]}


def _get_molecular_devices_well_bbox_2D(
    data: list[tuple[ArrayLike, dict]]
) -> tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """Compute well-shape based on stage position metadata."""
    assert "stage-position-x" in data[0][1].keys(), "Missing metaseries metadata."
    assert "stage-position-y" in data[0][1].keys(), "Missing metaseries metadata."
    assert "spatial-calibration-x" in data[0][1].keys(), "Missing metaseries metadata."
    assert "spatial-calibration-y" in data[0][1].keys(), "Missing metaseries metadata."

    min_x, max_x, min_y, max_y = None, None, None, None
    for d in data:
        pos_x = d[1]["stage-position-x"]
        pos_y = d[1]["stage-position-y"]
        res_x = d[1]["spatial-calibration-x"]
        res_y = d[1]["spatial-calibration-y"]

        if min_x is None:
            min_x = pos_x / res_x
            max_x = min_x + d[0].shape[1]
        elif min_x > (pos_x / res_x):
            min_x = pos_x / res_x

        if max_x < (pos_x / res_x) + d[0].shape[1]:
            max_x = (pos_x / res_x) + d[0].shape[1]

        if min_y is None:
            min_y = pos_y / res_y
            max_y = min_y + d[0].shape[0]
        elif min_y > pos_y / res_y:
            min_y = pos_y / res_y

        if max_y < (pos_y / res_y) + d[0].shape[0]:
            max_y = (pos_y / res_y) + d[0].shape[0]

    return min_y, min_x, max_y, max_x


def _montage_image_YX(data):
    """Montage 2D fields based on stage position metadata."""

    def sort_key(d):
        label = d[1]["stage-label"]

        label = label.split(":")

        if len(label) == 1:
            return label
        else:
            return int(label[1].replace("Site", ""))

    data.sort(key=sort_key, reverse=True)

    min_y, min_x, max_y, max_x = _get_molecular_devices_well_bbox_2D(data)

    shape = (int(np.round(max_y - min_y)), int(np.round(max_x - min_x)))

    img = np.zeros(shape, dtype=data[0][0].dtype)

    for d in data:
        pos_y = int(
            np.round(d[1]["stage-position-y"] / d[1]["spatial-calibration-y"] - min_y)
        )
        pos_x = int(
            np.round(d[1]["stage-position-x"] / d[1]["spatial-calibration-x"] - min_x)
        )

        img[pos_y : pos_y + d[0].shape[0], pos_x : pos_x + d[0].shape[1]] = d[0]

    return img


def _pixel_pos(dim: str, data: dict):
    return np.round(data[f"stage-position-{dim}"] / data[f"spatial-calibration-{dim}"])


def _montage_grid_image_YX(data):
    """Montage 2D fields into fixed grid, based on stage position metadata."""
    min_y = min(_pixel_pos("y", d[1]) for d in data)
    min_x = min(_pixel_pos("x", d[1]) for d in data)
    max_y = max(_pixel_pos("y", d[1]) for d in data)
    max_x = max(_pixel_pos("x", d[1]) for d in data)

    assert all([d[0].shape for d in data])
    step_y = data[0][0].shape[0]
    step_x = data[0][0].shape[1]

    shape = (
        int(np.round((max_y - min_y) / step_y + 1) * step_y),
        int(np.round((max_x - min_x) / step_x + 1) * step_x),
    )
    img = np.zeros(shape, dtype=data[0][0].dtype)

    for d in data:
        pos_x = int(np.round((_pixel_pos("x", d[1]) - min_x) / step_x))
        pos_y = int(np.round((_pixel_pos("y", d[1]) - min_y) / step_y))
        img[
            pos_y * step_y : (pos_y + 1) * step_y, pos_x * step_x : (pos_x + 1) * step_x
        ] = d[0]

    return img


def verify_integrity(field_metadata: list[dict]):
    metadata = field_metadata[0]
    for fm in field_metadata:
        assert fm == metadata, "Metadata is not consistent across fields."

    return metadata


def get_well_image_ZCYX(
    well_files: pd.DataFrame, assemble_fn: Callable = _montage_grid_image_YX
) -> tuple[ArrayLike, list[UIntHistogram], list[dict], dict]:
    """Assemble image data for the given well-files."""
    planes = well_files["z"].unique()

    plane_imgs = []
    channel_histograms = None
    channel_metadata = None
    general_metadata = None
    z_positions = []

    for z in sorted(planes, key=int):
        plane_files = well_files[well_files["z"] == z]
        img, ch_hists, ch_metas, meta = get_well_image_CYX(
            plane_files,
            assemble_fn=assemble_fn,
            include_z_position=True,
        )
        plane_imgs.append(img)
        z_positions.append(meta["z-position"])
        if not channel_histograms:
            channel_histograms = ch_hists
        else:
            channel_histograms = [
                hist1.combine(hist2)
                for hist1, hist2 in zip(channel_histograms, ch_hists)
            ]
        if not channel_metadata:
            channel_metadata = ch_metas
        if not general_metadata:
            general_metadata = meta

    zcyx = np.array(plane_imgs)

    # add z scaling (computed from slices) to general_metadata
    if len(z_positions) > 1:
        z_step = np.min(np.diff(z_positions))
        general_metadata["z-scaling"] = z_step

    return zcyx, channel_histograms, channel_metadata, general_metadata


def get_well_image_CYX(
    well_files: pd.DataFrame,
    assemble_fn: Callable = _montage_grid_image_YX,
    include_z_position: bool = False,
) -> tuple[ArrayLike, list[UIntHistogram], list[dict], dict]:
    """Assemble image data for the given well-files.

    For each channel a single 2D image is computed. If the well has multiple
    fields per channel the `assemble_fn` has to montage or stitch the fields
    accordingly.

    :param well_files: all files corresponding to the well
    :param assemble_fn: creates a single image for each channel
    :return: CYX image, channel-histograms, channel-metadata, general-metadata
    """
    channels = well_files["channel"].unique()
    channels.sort()

    channel_imgs = []
    channel_histograms = []
    channel_metadata = []
    general_metadata = None
    zpos_metadata = []
    for ch in channels:
        channel_files = well_files[well_files["channel"] == ch]

        imgs = []
        field_metadata = []
        for f in channel_files["path"]:
            img, ms_metadata = load_metaseries_tiff(f)
            ch_metadata = _build_ch_metadata(ms_metadata)
            zpos_metadata.append(_z_metadata(ms_metadata))
            imgs.append((img, ms_metadata))
            field_metadata.append(ch_metadata)
            if general_metadata is None:
                general_metadata = {
                    "spatial-calibration-x": ms_metadata["spatial-calibration-x"],
                    "spatial-calibration-y": ms_metadata["spatial-calibration-y"],
                    "spatial-calibration-units": ms_metadata[
                        "spatial-calibration-units"
                    ],
                    "pixel-type": ms_metadata["PixelType"],
                }

        img = assemble_fn(imgs)
        metadata = verify_integrity(field_metadata)

        channel_imgs.append(img)
        channel_histograms.append(UIntHistogram(img))

        channel_metadata.append(metadata)

    cyx = np.array(channel_imgs)
    # NB: z-position metadata can be inconsistent for MIPs
    # z_position = verify_integrity(zpos_metadata)
    z_position = zpos_metadata[0]
    if include_z_position:
        general_metadata.update(z_position)

    return cyx, channel_histograms, channel_metadata, general_metadata
