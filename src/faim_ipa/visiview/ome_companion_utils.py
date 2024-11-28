from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from defusedxml import ElementTree
from defusedxml.ElementTree import parse

from faim_ipa.io.metadata import ChannelMetadata
from faim_ipa.utils import rgb_to_hex, wavelength_to_rgb

SCHEMA = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"


def get_z_spacing(metadata: ElementTree) -> float | None:
    """
    Get the Z spacing from the first image in the XML metadata.

    Parameters
    ----------
    metadata :
        XML metadata

    Returns
    -------
        Z spacing or None if there is only one Z plane
    """
    image_0 = next(metadata.iter(f"{SCHEMA}Image"))
    pixels = next(image_0.iter(f"{SCHEMA}Pixels"))
    z_positions = {}
    for plane in pixels.iter(f"{SCHEMA}Plane"):
        z_positions[plane.get("TheZ")] = float(plane.get("PositionZ"))

    if len(z_positions) == 1:
        return None
    z_positions = list(z_positions.values())
    precision = -Decimal(str(z_positions[0])).as_tuple().exponent
    z_spacing = np.round(np.diff(z_positions).mean(), precision)
    return z_spacing


def get_yx_spacing(metadata: ElementTree) -> tuple[float, float]:
    """
    Get the YX spacing from the first image in the XML metadata.

    Parameters
    ----------
    metadata :
        XML metadata

    Returns
    -------
        YX spacing
    """
    image_0 = next(metadata.iter(f"{SCHEMA}Image"))
    pixels = next(image_0.iter(f"{SCHEMA}Pixels"))
    return (
        float(pixels.get("PhysicalSizeY")),
        float(pixels.get("PhysicalSizeX")),
    )


def get_exposure_time(metadata: ElementTree) -> tuple[float, str]:
    """
    Get the exposure time and unit from the first image in the XML metadata.

    Parameters
    ----------
    metadata :
        XML metadata

    Returns
    -------
        Exposure time and unit
    """
    image_0 = next(metadata.iter(f"{SCHEMA}Image"))
    pixels = next(image_0.iter(f"{SCHEMA}Pixels"))
    plane_0 = next(pixels.iter(f"{SCHEMA}Plane"))
    return plane_0.get("ExposureTime"), plane_0.get("ExposureTimeUnit")


def get_channels(metadata: ElementTree) -> dict[str, ChannelMetadata]:
    """
    Get the channel metadata from the XML metadata.

    Parameters
    ----------
    metadata :
        XML metadata

    Returns
    -------
        Channel metadata for each channel.
    """
    image_0 = next(metadata.iter(f"{SCHEMA}Image"))
    pixels = next(image_0.iter(f"{SCHEMA}Pixels"))
    channels = [channel.attrib for channel in pixels.iter(f"{SCHEMA}Channel")]
    objective = next(image_0.iter(f"{SCHEMA}ObjectiveSettings")).get("ID")
    yx_spacing = get_yx_spacing(metadata)
    exposure_time, exposure_time_unit = get_exposure_time(metadata)

    ch_metadata = {}
    for channel in channels:
        idx = int(channel["ID"].split(":")[-1])
        if "EmissionWavelength" in channel:
            wavelength = int(channel["EmissionWavelength"])
            display_color = rgb_to_hex(*wavelength_to_rgb(wavelength))
        else:
            wavelength = None
            display_color = "#ffffff"
        ch_metadata[f"w{idx+1}"] = ChannelMetadata(
            channel_index=idx,
            channel_name=channel["Name"],
            display_color=display_color,
            spatial_calibration_x=yx_spacing[1],
            spatial_calibration_y=yx_spacing[0],
            spatial_calibration_units="um",
            z_spacing=get_z_spacing(metadata),
            wavelength=wavelength,
            exposure_time=exposure_time,
            exposure_time_unit=exposure_time_unit,
            objective=objective,
        )

    return ch_metadata


def get_stage_positions(
    metadata: ElementTree,
) -> dict[str, tuple[float, float]]:
    """
    Get the stage positions for each image from the XML metadata.

    Parameters
    ----------
    metadata :
        XML metadata

    Returns
    -------
        Stage positions for each image.
    """
    positions = {}
    for i, image in enumerate(metadata.iter(f"{SCHEMA}Image")):
        image_id = image.get("ID")
        index = int(image_id.split(":")[-1])
        assert index == i, f"Expected index {i} but got {index}"

        try:
            stage_label = next(image.iter(f"{SCHEMA}StageLabel")).attrib

            y_pos = float(stage_label["Y"])
            x_pos = float(stage_label["X"])
            positions[str(i + 1)] = (y_pos, x_pos)
        except (StopIteration, KeyError):
            # No stage positions in metadata
            positions[str(i + 1)] = (0, 0)

    return positions


def parse_basic_metadata(companion_file: Path | str) -> dict[str, Any]:
    """
    Parse the basic metadata from the XML companion file.

    Note: Only the first entries are parsed, and it is assumed that the
    extracted metadata is the same for all images.
    If you want to parse the complete metadata, use ome-types instead.

    ome-types: https://ome-types.readthedocs.io/en/latest/usage/

    Parameters
    ----------
    companion_file :
        Path to the XML companion file.

    Returns
    -------
        Basic metadata:
        - z_spacing
        - yx_spacing
        - channels
        - stage_positions
    """
    with open(companion_file, "rb") as f:
        root = parse(f).getroot()
        try:
            z_spacing = get_z_spacing(root)
        except StopIteration:
            z_spacing = None

        return {
            "z_spacing": z_spacing,
            "yx_spacing": get_yx_spacing(root),
            "channels": get_channels(root),
            "stage_positions": get_stage_positions(root),
        }
