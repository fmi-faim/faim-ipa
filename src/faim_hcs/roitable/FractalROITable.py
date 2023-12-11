import pandas as pd

from faim_hcs.hcs.acquisition import PlateAcquisition, WellAcquisition
from faim_hcs.stitching import Tile


def create_ROI_tables(plate_acquistion: PlateAcquisition, calibration_dict):
    columns = [
        "FieldIndex",
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    plate_roi_tables = {}
    for well_acquisition in plate_acquistion.get_well_acquisitions():
        plate_roi_tables[well_acquisition.name] = dict(
            FOV_ROI_table=create_fov_ROI_table(
                well_acquisition.get_tiles(),
                columns,
                calibration_dict,
            ),
            well_ROI_table=create_well_ROI_table(
                well_acquisition,
                columns,
                calibration_dict,
            ),
        )

    return plate_roi_tables


def create_well_ROI_table(
    well_acquisition: WellAcquisition,
    columns,
    calibration_dict,
):
    well_roi = [
        "well_1",
        0.0,
        0.0,
        0.0,
        well_acquisition.get_shape()[-1] * calibration_dict["spatial-calibration-x"],
        well_acquisition.get_shape()[-2] * calibration_dict["spatial-calibration-y"],
        1.0,
    ]
    well_roi_table = pd.DataFrame(well_roi).T
    well_roi_table.columns = columns
    well_roi_table.set_index("FieldIndex", inplace=True)
    return well_roi_table


def create_fov_ROI_table(
    tiles: list[Tile], columns, calibration_dict: dict[str, float]
):
    fov_rois = []
    for tile in tiles:
        fov_rois.append(
            (
                "",
                tile.position.x * calibration_dict["spatial-calibration-x"],
                tile.position.y * calibration_dict["spatial-calibration-y"],
                0.0,
                tile.shape[-1] * calibration_dict["spatial-calibration-x"],
                tile.shape[-2] * calibration_dict["spatial-calibration-y"],
                1.0,
            )
        )
    roi_table = pd.DataFrame(fov_rois, columns=columns).set_index("FieldIndex")
    return roi_table
