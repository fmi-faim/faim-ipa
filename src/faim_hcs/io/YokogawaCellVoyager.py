from os.path import exists, join
from pathlib import Path
from typing import Union
from xml.etree import ElementTree as ET

import pandas as pd

BTS_NS = "{http://www.yokogawa.co.jp/BTS/BTSSchema/1.0}"


def parse_files(
    acquisition_dir: Union[Path, str],
):
    mlf_file = join(acquisition_dir, "MeasurementData.mlf")
    if not exists(mlf_file):
        raise ValueError(f"MeasurementData.mlf not found in: {acquisition_dir}")
    mlf_tree = ET.parse(mlf_file)
    mlf_root = mlf_tree.getroot()

    files = []
    for record in mlf_root.findall(BTS_NS + "MeasurementRecord"):
        row = {key.replace(BTS_NS, ""): value for key, value in record.attrib.items()}
        if row.pop("Type") == "IMG":
            row |= {
                "path": join(acquisition_dir, record.text),
                "well": chr(ord("@") + int(row.pop("Row")))
                + row.pop("Column").zfill(2),
            }
            files.append(row)

    return pd.DataFrame(files)


def parse_metadata(
    acquistion_dir: Union[Path, str],
):
    mrf_file = join(acquistion_dir, "MeasurementDetail.mrf")
    if not exists(mrf_file):
        raise ValueError(f"MeasurementDetail.mrf not found in: {acquistion_dir}")
    mrf_tree = ET.parse(mrf_file)
    mrf_root = mrf_tree.getroot()

    channels = []
    for channel in mrf_root.findall(BTS_NS + "MeasurementChannel"):
        row = {key.replace(BTS_NS, ""): value for key, value in channel.attrib.items()}
        channels.append(row)

    mes_file = join(
        acquistion_dir, mrf_root.attrib[BTS_NS + "MeasurementSettingFileName"]
    )
    if not exists(mes_file):
        raise ValueError(f"Settings file not found: {mes_file}")
    mes_tree = ET.parse(mes_file)
    mes_root = mes_tree.getroot()

    channel_settings = []
    for channel in mes_root.find(BTS_NS + "ChannelList").findall(BTS_NS + "Channel"):
        row = {key.replace(BTS_NS, ""): value for key, value in channel.attrib.items()}
        channel_settings.append(row)

    plate = mrf_root.find(BTS_NS + "MeasurementSamplePlate")
    wpi_file = join(acquistion_dir, plate.attrib[BTS_NS + "WellPlateFileName"])
    if not exists(wpi_file):
        raise ValueError(f"Plate information file not found: {wpi_file}")
    wpi_tree = ET.parse(wpi_file)
    wpi_root = wpi_tree.getroot()
    name = wpi_root.attrib[BTS_NS + "Name"]

    # NB: we probably do not need to parse the well plate product file
    # wpp_file = join(acquistion_dir, plate.attrib[BTS_NS + "WellPlateProductFileName"])

    return name, pd.merge(
        pd.DataFrame(channels),
        pd.DataFrame(channel_settings),
        left_on="Ch",
        right_on="Ch",
    )
