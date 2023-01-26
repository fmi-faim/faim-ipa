from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from numpy._typing import ArrayLike

from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff
from faim_hcs.UIntHistogram import UIntHistogram


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

    metadata = data[0][1]

    for d in data:
        pos_y = int(
            np.round(d[1]["stage-position-y"] / d[1]["spatial-calibration-y"] - min_y)
        )
        pos_x = int(
            np.round(d[1]["stage-position-x"] / d[1]["spatial-calibration-x"] - min_x)
        )

        img[pos_y : pos_y + d[0].shape[0], pos_x : pos_x + d[0].shape[1]] = d[0]

        assert (
            d[1]["_IllumSetting_"] == metadata["_IllumSetting_"]
        ), "Metadata is not consistent."
        assert (
            d[1]["spatial-calibration-x"] == metadata["spatial-calibration-x"]
        ), "Metadata is not consistent."
        assert (
            d[1]["spatial-calibration-y"] == metadata["spatial-calibration-y"]
        ), "Metadata is not consistent."
        assert (
            d[1]["spatial-calibration-units"] == metadata["spatial-calibration-units"]
        ), "Metadata is not consistent."
        assert d[1]["PixelType"] == metadata["PixelType"], "Metadata is not consistent."
        assert d[1]["_MagNA_"] == metadata["_MagNA_"], "Metadata is not consistent."
        assert (
            d[1]["_MagSetting_"] == metadata["_MagSetting_"]
        ), "Metadata is not consistent."
        assert (
            d[1]["Exposure Time"] == metadata["Exposure Time"]
        ), "Metadata is not consistent."
        assert (
            d[1]["Lumencor Cyan Intensity"] == metadata["Lumencor Cyan Intensity"]
        ), "Metadata is not consistent."
        assert (
            d[1]["Lumencor Green Intensity"] == metadata["Lumencor Green Intensity"]
        ), "Metadata is not consistent."
        assert (
            d[1]["Lumencor Red Intensity"] == metadata["Lumencor Red Intensity"]
        ), "Metadata is not consistent."
        assert (
            d[1]["Lumencor Violet Intensity"] == metadata["Lumencor Violet Intensity"]
        ), "Metadata is not consistent."
        assert (
            d[1]["Lumencor Yellow Intensity"] == metadata["Lumencor Yellow Intensity"]
        ), "Metadata is not consistent."
        assert (
            d[1]["ShadingCorrection"] == metadata["ShadingCorrection"]
        ), "Metadata is not consistent."
        assert (
            d[1]["_IllumSetting_"] == metadata["_IllumSetting_"]
        ), "Metadata is not consistent."
        assert (
            d[1]["_IllumSetting_"] == metadata["_IllumSetting_"]
        ), "Metadata is not consistent."
        assert (
            d[1]["_IllumSetting_"] == metadata["_IllumSetting_"]
        ), "Metadata is not consistent."

    metadata.pop("stage-position-x")
    metadata.pop("stage-position-y")
    metadata.pop("stage-label")
    metadata.pop("SiteX")
    metadata.pop("SiteY")

    return img, metadata


def get_well_image_CYX(
    well_files: pd.DataFrame, assemble_fn: Callable = _montage_image_YX
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
    for ch in channels:
        channel_files = well_files[well_files["channel"] == ch]

        data = []
        for f in channel_files["path"]:
            data.append(load_metaseries_tiff(f))

        img, metadata = assemble_fn(data)
        channel_imgs.append(img)
        channel_histograms.append(UIntHistogram(img))

        if general_metadata is None:
            general_metadata = {
                "spatial-calibration-x": metadata["spatial-calibration-x"],
                "spatial-calibration-y": metadata["spatial-calibration-y"],
                "spatial-calibration-units": metadata["spatial-calibration-units"],
                "PixelType": metadata["PixelType"],
            }

        metadata.pop("spatial-calibration-x")
        metadata.pop("spatial-calibration-y")
        metadata.pop("spatial-calibration-units")
        metadata.pop("PixelType")
        channel_metadata.append(metadata)

    cyx = np.array(channel_imgs)

    return cyx, channel_histograms, channel_metadata, general_metadata
