from decimal import Decimal
from typing import Callable, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff
from faim_hcs.MontageUtils import montage_grid_image_YX
from faim_hcs.UIntHistogram import UIntHistogram
from faim_hcs.utils import rgb_to_hex, wavelength_to_rgb


def extract_z_position(row: pd.Series) -> float:
    if "z" in row.keys() and row["z"] is not None:
        return int(row["z"])
    else:
        # ImageXpress starts counting at 1.
        return 1


def _build_ch_metadata(metaseries_ch_metadata: dict):
    """Build channel metadata from metaseries metadata."""

    def get_wavelength_power(channel):
        # Custom channel names
        custom_channel_dict = {
            "Lumencor Cyan Intensity": "cyan",
            "Lumencor Green Intensity": "green",
            "Lumencor Red Intensity": "red",
            "Lumencor Violet Intensity": "violet",
            "Lumencor Yellow Intensity": "yellow",
        }

        # Find the intensity channnels:
        wavelengths = []
        for key in channel.keys():
            if key.endswith("Intensity"):
                wavelengths.append(key)

        for wavelength in wavelengths:
            if channel[wavelength] > 0.0:
                if wavelength in custom_channel_dict.keys():
                    wl = custom_channel_dict[wavelength]
                else:
                    wl = wavelength
                power = channel[wavelength]

                # Assert all other power values are zero
                for other_wavelength in wavelengths:
                    if other_wavelength != wavelength:
                        assert channel[other_wavelength] == 0.0

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
) -> tuple[ArrayLike, list[UIntHistogram], list[dict], dict, dict]:
    """Assemble image data for the given well-files."""
    planes = well_files["z"].unique()

    stacks = []
    channel_histograms = []
    channel_metadata = []
    px_metadata = None
    z_positions = []
    roi_tables = {}

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
    max_stack_size = max([x.shape[0] for x in stacks if x is not None])
    for roi_table in roi_tables.values():
        roi_table["len_z_micrometer"] = z_sampling * (max_stack_size - 1)

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

    return czyx, channel_histograms, channel_metadata, px_metadata, roi_tables


def get_well_image_CYX(
    well_files: pd.DataFrame,
    channels: list[str],
    assemble_fn: Callable = montage_grid_image_YX,
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
    px_metadata = None
    roi_tables = {}
    for ch in channels:
        channel_files = well_files[well_files["channel"] == ch]

        if len(channel_files) > 0:
            px_metadata, img, ch_metadata, _, roi_tables = get_img_YX(
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
    """Assemble single 2D image for all fields.

    general_metadata: spatial-calibration-x, spatial-calibration-y, spatial-calibration-units, pixel-type
    img: 2D pixel array (yx)
    metadata: list[dict] (wavelength, power, exposure-time, exposure-time-unit, shading-correction,
        channel-name, objective-numerical-aperture, objective, display-color)
    z_position
    roi_tables: dict[DataFrame]
    """
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
