from decimal import Decimal
from typing import Any, Callable, Optional, Union

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
    metadata = {
        "wavelength": wavelength,
        "power": power,
        "exposure-time": time,
        "exposure-time-unit": unit,
        "shading-correction": metaseries_ch_metadata["ShadingCorrection"] == "On",
        "channel-name": metaseries_ch_metadata["_IllumSetting_"],
        "objective-numerical-aperture": metaseries_ch_metadata["_MagNA_"],
        "objective": metaseries_ch_metadata["_MagSetting_"],
        "display-color": display_color,
    }
    if "Z Projection Method" in metaseries_ch_metadata:
        metadata["z-projection-method"] = metaseries_ch_metadata["Z Projection Method"]
    return metadata


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


def montage_stage_pos_image_YX(data):
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


def _stage_label(data: dict):
    """Get the FOV string for a given FOV dict"""
    try:
        return data["stage-label"].split(":")[-1][1:]
    # Return an empty string if the metadata does not contain stage-label
    except KeyError:
        return ""


def montage_grid_image_YX(data):
    """Montage 2D fields into fixed grid, based on stage position metadata.
    
    :param data: list of tuples of (image, metadata)
    :return: img (stitched 2D np array), fov_df (dataframe with region of 
             interest information for the fields of view)
    """
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
    fov_rois = []

    for d in data:
        pos_x = int(np.round((_pixel_pos("x", d[1]) - min_x) / step_x))
        pos_y = int(np.round((_pixel_pos("y", d[1]) - min_y) / step_y))
        img[
            pos_y * step_y : (pos_y + 1) * step_y, pos_x * step_x : (pos_x + 1) * step_x
        ] = d[0]
        # Create the FOV ROI table for the site in physical units
        fov_rois.append((
            _stage_label(d[1]), 
            pos_y * step_y * d[1]['spatial-calibration-y'], 
            pos_x * step_x * d[1]['spatial-calibration-x'], 
            0.0, # Hard-coded z starting position
            step_y * d[1]['spatial-calibration-y'], 
            step_x * d[1]['spatial-calibration-x'],
            1.0, # Hard-coded z length (for 2D planes), to be overwritten if 
                 # the 2D planes are assembled into a 3D stack
            ))
    
    # Generate the ROI tables
    roi_tables = {}
    columns = [
        "FieldIndex", 
        "x_micrometer", 
        "y_micrometer", 
        "z_micrometer", 
        "len_x_micrometer", 
        "len_y_micrometer", 
        "len_z_micrometer"
        ]

    roi_table = pd.DataFrame(fov_rois, columns=columns).set_index("FieldIndex")
    roi_tables["FOV_ROI_table"] = roi_table

    # Generate a well ROI table
    well_roi = [
        "well_1", 
        0.0, 
        0.0, 
        0.0, 
        shape[1] * d[1]['spatial-calibration-x'], 
        shape[0] * d[1]['spatial-calibration-y'], 
        1.0
        ]
    well_roi_table = pd.DataFrame(well_roi).T
    well_roi_table.columns=columns
    well_roi_table.set_index("FieldIndex", inplace=True)
    roi_tables["well_ROI_table"] = well_roi_table

    return img, roi_tables


def verify_integrity(field_metadata: list[dict]):
    metadata = field_metadata[0]
    for fm in field_metadata:
        assert fm == metadata, "Metadata is not consistent across fields."

    return metadata


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


def roll_single_plane(stacks, ch_z_positions):
    min_z, max_z = [], []
    for z_positions in ch_z_positions:
        if z_positions is not None and None not in z_positions:
            min_z.append(np.min(z_positions))
            max_z.append(np.max(z_positions))

    min_z, max_z = np.mean(min_z), np.mean(max_z)

    for i, z_positions in enumerate(ch_z_positions):
        if z_positions is not None and None in z_positions:
            # Single planes are always acquired in first z-step
            step_size = (max_z - min_z) / len(z_positions)
            shift_z = int((z_positions[0] - min_z) // step_size)
            stacks[i] = np.roll(stacks[i], shift_z, axis=0)


def get_well_image_CZYX(
    well_files: pd.DataFrame,
    channels: list[str],
    assemble_fn: Callable = montage_grid_image_YX,
) -> tuple[ArrayLike, list[UIntHistogram], list[dict], dict]:
    """Assemble image data for the given well-files."""
    planes = well_files["z"].unique()

    stacks = []
    channel_histograms = []
    channel_metadata = []
    px_metadata = None
    z_positions = []

    for ch in channels:
        channel_files = well_files[well_files["channel"] == ch]
        if len(channel_files) > 0:
            plane_imgs = []
            z_plane_positions = []
            ch_metadata = None
            for z in sorted(planes, key=int):
                plane_files = channel_files[channel_files["z"] == z]

                if len(plane_files) > 0:
                    px_metadata, img, ch_meta, z_position, roi_tables = get_img_YX(
                        assemble_fn=assemble_fn, files=plane_files
                    )

                    if ch_metadata is None:
                        ch_metadata = ch_meta

                    plane_imgs.append(img)
                    z_plane_positions.append(z_position)
                else:
                    plane_imgs.append(None)
                    z_plane_positions.append(None)

            zyx = build_stack(plane_imgs)
            stacks.append(zyx)
            channel_histograms.append(UIntHistogram(stacks[-1]))
            channel_metadata.append(ch_metadata)
            z_positions.append(z_plane_positions)
        else:
            stacks.append(None)
            channel_histograms.append(UIntHistogram())
            channel_metadata.append(
                {
                    "channel-name": "empty",
                    "display-color": "000000",
                }
            )
            z_positions.append(None)

    z_sampling = compute_z_sampling(z_positions)
    px_metadata["z-scaling"] = z_sampling

    roll_single_plane(stacks, z_positions)

    czyx = build_stack(stacks)

    for i in range(len(channel_histograms)):
        if channel_histograms[i] is None:
            channel_histograms[i] = UIntHistogram()
            assert channel_metadata[i] is None
            channel_metadata[i] = {
                "channel-name": "empty",
                "display-color": "000000",
            }

    return czyx, channel_histograms, channel_metadata, px_metadata


def get_well_image_CYX(
    well_files: pd.DataFrame,
    channels: list[str],
    assemble_fn: Callable = montage_grid_image_YX,
    include_z_position: bool = False,
) -> tuple[ArrayLike, list[UIntHistogram], list[dict], dict]:
    """Assemble image data for the given well-files.

    For each channel a single 2D image is computed. If the well has multiple
    fields per channel the `assemble_fn` has to montage or stitch the fields
    accordingly.

    :param well_files: all files corresponding to the well
    :param channels: list of required channels
    :param assemble_fn: creates a single image for each channel
    :param include_z_position: whether to include z-position metadata
    :return: CYX image, channel-histograms, channel-metadata, general-metadata, 
                roi-tables dictionary
    """
    channel_imgs = {}
    channel_histograms = {}
    channel_metadata = {}
    px_metadata = None
    roi_tables = {}
    for ch in channels:
        channel_files = well_files[well_files["channel"] == ch]

        if len(channel_files) > 0:
            px_metadata, img, ch_metadata, z_position, roi_tables = get_img_YX(
                assemble_fn, channel_files
            )

            channel_imgs[ch] = img
            channel_histograms[ch] = UIntHistogram(img)

            channel_metadata[ch] = ch_metadata

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

    return cyx, channel_hists, channel_meta, px_metadata, roi_tables


def get_img_YX(assemble_fn, files):
    imgs = []
    field_metadata = []
    z_positions = []
    general_metadata = None
    for f in sorted(files["path"]):
        img, ms_metadata = load_metaseries_tiff(f)
        ch_metadata = _build_ch_metadata(ms_metadata)
        z_positions.append(_z_metadata(ms_metadata))
        imgs.append((img, ms_metadata))
        field_metadata.append(ch_metadata)
        if general_metadata is None:
            general_metadata = {
                "spatial-calibration-x": ms_metadata["spatial-calibration-x"],
                "spatial-calibration-y": ms_metadata["spatial-calibration-y"],
                "spatial-calibration-units": ms_metadata["spatial-calibration-units"],
                "pixel-type": ms_metadata["PixelType"],
            }
    img, roi_tables = assemble_fn(imgs)
    metadata = verify_integrity(field_metadata)
    zs = [z["z-position"] for z in z_positions]
    return general_metadata, img, metadata, np.mean(zs), roi_tables
