from decimal import Decimal
from pathlib import Path
from typing import Union

import numpy as np
from lxml import etree

from faim_hcs.io.ChannelMetadata import ChannelMetadata
from faim_hcs.utils import rgb_to_hex, wavelength_to_rgb

SCHEMA = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"


def get_z_spacing(metadata):
    image_0 = next(metadata.iterchildren(f"{SCHEMA}Image"))
    pixels = next(image_0.iterchildren(f"{SCHEMA}Pixels"))
    z_positions = set()
    for plane in pixels.iterchildren(f"{SCHEMA}Plane"):
        z_positions.add(float(plane.get("PositionZ")))

    if len(z_positions) == 1:
        return None
    else:
        z_positions = np.array(sorted(z_positions))
        precision = -Decimal(str(z_positions[0])).as_tuple().exponent
        return np.round(np.diff(z_positions).mean(), precision)


def get_yx_spacing(metadata):
    image_0 = next(metadata.iterchildren(f"{SCHEMA}Image"))
    pixels = next(image_0.iterchildren(f"{SCHEMA}Pixels"))
    return (
        float(pixels.get("PhysicalSizeY")),
        float(pixels.get("PhysicalSizeX")),
    )


def get_exposure_time(metadata):
    image_0 = next(metadata.iterchildren(f"{SCHEMA}Image"))
    pixels = next(image_0.iterchildren(f"{SCHEMA}Pixels"))
    plane_0 = next(pixels.iterchildren(f"{SCHEMA}Plane"))
    return plane_0.get("ExposureTime"), plane_0.get("ExposureTimeUnit")


def get_channels(metadata):
    image_0 = next(metadata.iterchildren(f"{SCHEMA}Image"))
    pixels = next(image_0.iterchildren(f"{SCHEMA}Pixels"))
    channels = [channel.attrib for channel in pixels.iterchildren(f"{SCHEMA}Channel")]
    objective = next(image_0.iterchildren(f"{SCHEMA}ObjectiveSettings")).get("ID")
    yx_spacing = get_yx_spacing(metadata)
    exposure_time, exposure_time_unit = get_exposure_time(metadata)

    ch_metadata = {}
    for channel in channels:
        idx = int(channel["ID"].split(":")[-1])
        if "EmissionWavelength" in channel:
            wavelength = int(channel["EmissionWavelength"])
            display_color = rgb_to_hex(*wavelength_to_rgb(wavelength))
        else:
            wavelength = channel["Name"]
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


def get_stage_positions(metadata):
    positions = {}
    for i, image in enumerate(metadata.iterchildren(f"{SCHEMA}Image")):
        id = image.get("ID")
        index = int(id.split(":")[-1])
        assert index == i, f"Expected index {i} but got {index}"

        stage_label = next(image.iterchildren(f"{SCHEMA}StageLabel")).attrib

        y_pos = float(stage_label["Y"])
        x_pos = float(stage_label["X"])
        positions[i + 1] = (y_pos, x_pos)

    return positions


def parse_basic_metadata(companion_file: Union[Path, str]):
    with open(companion_file, "rb") as f:
        root = etree.parse(f).getroot()
        metadata = dict(
            z_spacing=get_z_spacing(root),
            yx_spacing=get_yx_spacing(root),
            channels=get_channels(root),
            stage_positions=get_stage_positions(root),
        )
    return metadata


if __name__ == "__main__":
    from datetime import datetime

    start = datetime.now()
    print(
        parse_basic_metadata(
            "/tungstenfs/scratch/gturco/calvida/240112_crispr_timelapse/240112_crispr_timelapse_1.companion.ome"
        )
    )
    end = datetime.now()
    print(end - start)
