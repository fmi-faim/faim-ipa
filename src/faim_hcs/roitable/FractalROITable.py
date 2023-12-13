import anndata as ad
import numpy as np
import pandas as pd
from zarr import Group

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
        well_acquisition.get_shape()[-3] * calibration_dict["spatial-calibration-z"],
    ]
    well_roi_table = pd.DataFrame(well_roi).T
    well_roi_table.columns = columns
    well_roi_table.set_index("FieldIndex", inplace=True)
    return well_roi_table


def create_fov_ROI_table(
    tiles: list[Tile], columns, calibration_dict: dict[str, float]
):
    fov_rois = []
    tile = tiles[0]
    min_z = tile.position.z * calibration_dict["spatial-calibration-z"]
    max_z = (tile.position.z + 1) * calibration_dict["spatial-calibration-z"]
    for tile in tiles:
        z_start = tile.position.z * calibration_dict["spatial-calibration-z"]
        z_end = (tile.position.z + 1) * calibration_dict["spatial-calibration-z"]
        if z_start < min_z:
            min_z = z_start

        if z_end > max_z:
            max_z = z_end

        if tile.position.z == 0 and tile.position.channel == 0:
            fov_rois.append(
                (
                    "",
                    tile.position.x * calibration_dict["spatial-calibration-x"],
                    tile.position.y * calibration_dict["spatial-calibration-y"],
                    tile.position.z * calibration_dict["spatial-calibration-z"],
                    tile.shape[-1] * calibration_dict["spatial-calibration-x"],
                    tile.shape[-2] * calibration_dict["spatial-calibration-y"],
                    (tile.position.z + 1) * calibration_dict["spatial-calibration-z"],
                )
            )
    roi_table = pd.DataFrame(fov_rois, columns=columns).set_index("FieldIndex")

    roi_table["z_micrometer"] = min_z
    roi_table["len_z_micrometer"] = max_z
    return roi_table


def write_roi_table(
    roi_table: pd.DataFrame,
    table_name: str,
    group: Group,
):
    """Writes a roi table to an OME-Zarr image. If no table folder exists, it is created."""
    group_tables = group.require_group("tables")

    # Assign dtype explicitly, to avoid
    # >> UserWarning: X converted to numpy array with dtype float64
    # when creating AnnData object
    df_roi = roi_table.astype(np.float32)

    adata = ad.AnnData(X=df_roi)
    adata.obs_names = roi_table.index
    adata.var_names = list(map(str, roi_table.columns))
    ad._io.specs.write_elem(group_tables, table_name, adata)
    update_table_metadata(group_tables, table_name)


def update_table_metadata(group_tables, table_name):
    if "tables" not in group_tables.attrs:
        group_tables.attrs["tables"] = [table_name]
    elif table_name not in group_tables.attrs["tables"]:
        group_tables.attrs["tables"] = group_tables.attrs["tables"] + [table_name]
