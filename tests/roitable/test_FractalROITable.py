from os.path import join
from pathlib import Path

import pytest
import zarr
from numpy.testing import assert_array_almost_equal
from ome_zarr.io import parse_url

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.imagexpress import StackAcquisition
from faim_hcs.roitable.FractalROITable import (
    create_fov_ROI_table,
    create_ROI_tables,
    create_well_ROI_table,
    write_roi_table,
)


@pytest.fixture
def plate_acquisition():
    return StackAcquisition(
        acquisition_dir=Path(__file__).parent.parent.parent
        / "resources"
        / "Projection-Mix",
        alignment=TileAlignmentOptions.GRID,
    )


def test_create_fov_ROI_table(plate_acquisition):
    tiles = next(plate_acquisition.get_well_acquisitions()).get_tiles()
    roi_table = create_fov_ROI_table(
        tiles=tiles,
        columns=[
            "FieldIndex",
            "x_micrometer",
            "y_micrometer",
            "z_micrometer",
            "len_x_micrometer",
            "len_y_micrometer",
            "len_z_micrometer",
        ],
        calibration_dict={
            "spatial-calibration-x": 1.3668,
            "spatial-calibration-y": 1.3668,
            "spatial-calibration-z": 5.0,
        },
    )
    target_values = [
        0.0,
        0.0,
        0.0,
        699.8016,
        699.8016,
        50.0,
    ]
    assert roi_table.iloc[0].values.flatten().tolist() == target_values
    target_values = [
        699.8016,
        0.0,
        0.0,
        699.8016,
        699.8016,
        50.0,
    ]
    assert roi_table.iloc[1].values.flatten().tolist() == target_values


def test_create_well_ROI_table(plate_acquisition):
    well_acquisition = next(plate_acquisition.get_well_acquisitions())
    roi_table = create_well_ROI_table(
        well_acquisition=well_acquisition,
        columns=[
            "FieldIndex",
            "x_micrometer",
            "y_micrometer",
            "z_micrometer",
            "len_x_micrometer",
            "len_y_micrometer",
            "len_z_micrometer",
        ],
        calibration_dict={
            "spatial-calibration-x": 1.3668,
            "spatial-calibration-y": 1.3668,
            "spatial-calibration-z": 5.0,
        },
    )
    target_values = [
        0.0,
        0.0,
        0.0,
        1399.6032,
        699.8016,
        50.0,
    ]
    assert roi_table.iloc[0].values.flatten().tolist() == target_values


def test_create_ROI_tables(plate_acquisition):
    roi_tables = create_ROI_tables(
        plate_acquistion=plate_acquisition,
        calibration_dict={
            "spatial-calibration-x": 1.3668,
            "spatial-calibration-y": 1.3668,
            "spatial-calibration-z": 5.0,
        },
    )
    assert len(roi_tables) == 2
    assert roi_tables["E07"]["FOV_ROI_table"].shape == (2, 6)
    assert roi_tables["E07"]["well_ROI_table"].shape == (1, 6)
    assert roi_tables["E08"]["FOV_ROI_table"].shape == (2, 6)
    assert roi_tables["E08"]["well_ROI_table"].shape == (1, 6)


@pytest.fixture
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("hcs_plate")


def test_write_roi_table(tmp_dir, plate_acquisition):
    roi_tables = create_ROI_tables(
        plate_acquistion=plate_acquisition,
        calibration_dict={
            "spatial-calibration-x": 1.3668,
            "spatial-calibration-y": 1.3668,
            "spatial-calibration-z": 5.0,
        },
    )

    store = parse_url(join(tmp_dir, "tmp.zarr"), mode="w").store
    group = zarr.group(store=store)

    write_roi_table(
        roi_table=roi_tables["E07"]["FOV_ROI_table"],
        table_name="FOV_ROI_table",
        group=group,
    )

    import anndata as ad

    df_fov = ad.read_zarr(
        join(
            tmp_dir,
            "tmp.zarr",
            "tables",
            "FOV_ROI_table",
        )
    ).to_df()
    assert_array_almost_equal(
        df_fov.to_numpy(), roi_tables["E07"]["FOV_ROI_table"].to_numpy(), decimal=4
    )
