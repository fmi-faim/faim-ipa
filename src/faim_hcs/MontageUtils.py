from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _pixel_pos(dim: str, data: dict):
    return np.round(data[f"stage-position-{dim}"] / data[f"spatial-calibration-{dim}"])


def montage_grid_image_YX(data):
    """Montage 2D fields into fixed grid, based on stage position metadata.

    Uses the stage position coordinates to decide which grid cell to put the
    image in. Always writes images into a grid, thus avoiding overwriting
    partially overwriting parts of images. Not well suited for arbitarily
    positioned fields. In that case, use `montage_stage_pos_image_YX`.

    Also calculates ROI tables for the whole well and the field of views.
    Given that Fractal ROI tables are always 3D, but we only stitch the xy
    planes here, the z starting position is always 0 and the
    z extent is set to 1. This is overwritten downsteam if the 2D planes are
    assembled into a 3D stack.

    :param data: list of tuples of (image, metadata)
    :return: img (stitched 2D np array), fov_df (dataframe with region of
             interest information for the fields of view)
    """
    min_y = min(_pixel_pos("y", d[1]) for d in data)
    min_x = min(_pixel_pos("x", d[1]) for d in data)
    max_y = max(_pixel_pos("y", d[1]) for d in data)
    max_x = max(_pixel_pos("x", d[1]) for d in data)

    assert all([d[0].shape == data[0][0].shape for d in data])
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
        fov_rois.append(
            (
                _stage_label(d[1]),
                pos_x * step_x * d[1]["spatial-calibration-x"],
                pos_y * step_y * d[1]["spatial-calibration-y"],
                0.0,
                step_x * d[1]["spatial-calibration-x"],
                step_y * d[1]["spatial-calibration-y"],
                1.0,
            )
        )

    roi_tables = create_ROI_tables(fov_rois, shape, calibration_dict=d[1])

    return img, roi_tables


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
    """Montage 2D fields based on stage position metadata.

    Montages 2D fields based on stage position metadata. If the stage position
    specifies overlapping images, the overlapping part is overwritten
    (=> just uses the data of one image). Not well suited for regular grids,
    as the stage position can show overlap, but overwriting of data at the
    edge is not the intended behavior. In that case, use
    `montage_grid_image_YX`.

    Also calculates ROI tables for the whole well and the field of views in
    the Fractal ROI table format. We only stitch the xy planes here.
    Therefore, the z starting position is always 0 and the z extent is set to
    1. This is overwritten downsteam if the 2D planes are assembled into a
    3D stack.

    :param data: list of tuples (image, metadata)
    :return: img (stitched 2D np array), fov_df (dataframe with region of
            interest information for the fields of view)
    """

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

    fov_rois = []

    for d in data:
        pos_y = int(
            np.round(d[1]["stage-position-y"] / d[1]["spatial-calibration-y"] - min_y)
        )
        pos_x = int(
            np.round(d[1]["stage-position-x"] / d[1]["spatial-calibration-x"] - min_x)
        )

        img[pos_y : pos_y + d[0].shape[0], pos_x : pos_x + d[0].shape[1]] = d[0]

        # Create the FOV ROI table for the site in physical units
        fov_rois.append(
            (
                _stage_label(d[1]),
                pos_x * d[1]["spatial-calibration-x"],
                pos_y * d[1]["spatial-calibration-y"],
                0.0,
                d[0].shape[1] * d[1]["spatial-calibration-x"],
                d[0].shape[0] * d[1]["spatial-calibration-y"],
                1.0,
            )
        )

    roi_tables = create_ROI_tables(fov_rois, shape, calibration_dict=d[1])

    return img, roi_tables


def _stage_label(data: dict):
    """Get the field of view (FOV) string for a given FOV dict"""
    try:
        return data["stage-label"].split(":")[-1][1:]
    # Return an empty string if the metadata does not contain stage-label
    except KeyError:
        return ""


def create_ROI_tables(fov_rois, shape, calibration_dict):
    columns = [
        "FieldIndex",
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    roi_tables = {}
    roi_tables["FOV_ROI_table"] = create_fov_ROI_table(fov_rois, columns)
    roi_tables["well_ROI_table"] = create_well_ROI_table(
        shape[1],
        shape[0],
        calibration_dict["spatial-calibration-x"],
        calibration_dict["spatial-calibration-y"],
        columns,
    )
    return roi_tables


def create_well_ROI_table(shape_x, shape_y, pixel_size_x, pixel_size_y, columns):
    well_roi = [
        "well_1",
        0.0,
        0.0,
        0.0,
        shape_x * pixel_size_x,
        shape_y * pixel_size_y,
        1.0,
    ]
    well_roi_table = pd.DataFrame(well_roi).T
    well_roi_table.columns = columns
    well_roi_table.set_index("FieldIndex", inplace=True)
    return well_roi_table


def create_fov_ROI_table(fov_rois, columns):
    roi_table = pd.DataFrame(fov_rois, columns=columns).set_index("FieldIndex")
    return roi_table
