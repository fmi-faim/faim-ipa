import re
import xml.etree.ElementTree as ET
from glob import glob
from os.path import basename, join
from typing import Dict, List

import numpy as np
import pandas as pd
from dateutil import parser
from tifffile import imread


def get_metadata(input_dir: str):
    """
    Extract acquisition date, pixel-size information, acquisition camera
    index, and channel information (laser, filter and objective) from a
    Yokogawa CV7000 or CV8000 acquisition. The information is extracted from
    mse- and mrf-files written by Yokogawa.

    :param input_dir: location of the Yokogawa acquisition
    :return: acquisition_date, pixel_size, pixel_size_unit, channel_information
    """
    mrf_file = join(input_dir, "MeasurementDetail.mrf")

    mrf_tree = ET.parse(mrf_file)
    mrf_root = mrf_tree.getroot()
    mrf_ns = mrf_root.tag.replace("MeasurementDetail", "")

    date_format_str = mrf_root.attrib[mrf_ns + "BeginTime"]
    date = parser.parse(date_format_str)
    acquisition_date_str = date.strftime("%Y-%m-%d")

    channels = {}
    for child in mrf_root:
        ch = child.get(mrf_ns + "Ch")
        if ch is not None:
            channels[ch] = {
                "pixel_size": child.get(mrf_ns + "HorizontalPixelDimension"),
                "cam_index": child.get(mrf_ns + "CameraNumber"),
            }

    pixel_size = float(mrf_root[1].attrib.get(mrf_ns + "HorizontalPixelDimension"))
    pixel_size_unit = "micron"

    mes_file = glob(join(input_dir, "*.mes"))[0]

    mes_tree = ET.parse(mes_file)
    mes_root = mes_tree.getroot()
    mes_ns = mes_root.tag.replace("MeasurementSetting", "")

    for child in mes_root[2]:
        ch = child.get(mes_ns + "Ch")
        if ch is not None:
            channel_dict = channels[ch]
            channel_dict["objective"] = child.get(mes_ns + "Objective").replace(
                " ", "-"
            )
            channel_dict["filter"] = child.get(mes_ns + "Acquisition").replace("/", "-")
            channel_dict["laser"] = child[0].text

    return acquisition_date_str, pixel_size, pixel_size_unit, channels


def parse_filename(file_name: str, plate_name: str):
    """
    Parse CellVoyager image file names.
    File name pattern:
    {plate_name}_{well}_{timepoint}{field}{L}{action}{Z-plane}{channel}.tif

    :param file_name:
    :param plate_name:
    :return: plate_name, well, time_point, field, line, action, z, channel
    """
    tmp = basename(file_name)
    tmp = tmp.replace(plate_name, "")

    regex = re.compile(
        "_([A-Z]{1}[0-9]{2})_(T[0-9]{4})F([0-9]+)(L[0-9]{2})(A[0-9]+)Z([0-9]+)(C[0-9]{2}).tif"
    )

    well, time_point, field, line, action, z, channel = regex.findall(tmp)[0]

    return plate_name, well, time_point, int(field), line, action, int(z), channel


def create_table(files: List[str], plate_name: str) -> pd.DataFrame:
    """
    Create table of file-names with columns for plate, well, time-point,
    field, L, action, Z and channel.

    :param files: image file list
    :param plate_name: Name of the plate
    :return: table
    """
    plate = []
    well = []
    timepoint = []
    field = []
    lines = []
    action = []
    z = []
    channel = []
    path = []
    for file in files:
        p, w, t, f, line, a, z_, c = parse_filename(file, plate_name)
        plate.append(p)
        well.append(w)
        timepoint.append(t)
        field.append(f)
        lines.append(line)
        action.append(a)
        z.append(z_)
        channel.append(c)
        path.append(file)

    return pd.DataFrame(
        {
            "plate": plate,
            "well": well,
            "timepoint": timepoint,
            "field": field,
            "line": lines,
            "action": action,
            "z": z,
            "channel": channel,
            "path": path,
        }
    )


def build_field_stacks_for_channels(table: pd.DataFrame, z_plane: int) -> Dict:
    """
    Stack all fields of a given z-plane for every channel and return them.

    :param table: of all image files
    :param z_plane: to stack
    :return: field stack for each found channel
    """
    z_table = table[table["z"] == z_plane]
    acquired_channels = z_table.channel.unique().tolist()
    acquired_channels.sort()

    stacks = {}
    for ch in acquired_channels:
        sub_table = z_table[z_table["channel"] == ch]
        fields = []
        for file_path in sub_table["path"]:
            fields.append(imread(file_path))

        stacks[ch] = np.array(fields)

    return stacks


def subtract_dark_images(stacks: Dict, channel_metadata: Dict, input_dir: str) -> Dict:
    """
    Subtract camera dark images from channels.

    :param stacks: field stacks for each channel
    :param channel_metadata: metadata about channels
    :param input_dir: location of the Yokogawa acquisition
    :return: dark image subtracted stacks
    """
    dark_img_subtracted = {}
    for key, value in stacks.items():
        channel_idx = str(int(key[1:]))
        cam_idx = channel_metadata[channel_idx]["cam_index"]
        dark_img = imread(glob(join(input_dir, f"*_CAM{cam_idx}.tif"))[0])

        dark_img_subtracted[key] = value - dark_img

    return dark_img_subtracted


def compute_median_projection(stacks: Dict) -> Dict:
    """
    Compute median projection of the fields for each channel.

    :param stacks: field stacks
    :return: median projection for each channel
    """
    projections = {}
    for key, value in stacks.items():
        projections[key] = np.median(value, axis=0)

    return projections


def get_output_name(acquisition_date: str, channel: Dict) -> str:
    """
    Create output name for reference images.

    :param acquisition_date:
    :param channel:
    :return: "{acquisition_date}_{objective}_{laser}_{filter}.tif"
    """
    obj = channel["objective"]
    laser = channel["laser"]
    filter = channel["filter"]
    return f"{acquisition_date}_{obj}_{laser}_{filter}.tif"
