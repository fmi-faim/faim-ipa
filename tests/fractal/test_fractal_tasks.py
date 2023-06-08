# Fractal example scripts

import tempfile
from os.path import join
from pathlib import Path

import anndata as ad
import pytest

from faim_hcs.fractal.fractal_create_ome_zarr_md import create_ome_zarr_md
from faim_hcs.fractal.fractal_md_to_ome_zarr import md_to_ome_zarr


def test_ome_zarr_conversion_mode_all_fractal_tasks():
    ROOT_DIR = Path(__file__).parent
    input_paths = [join(ROOT_DIR.parent.parent, "resources", "Projection-Mix")]
    tmp_dir = tempfile.mkdtemp()
    zarr_root = Path(tmp_dir, "zarr-files")
    zarr_root.mkdir()

    mode = "all"

    order_name = "example-order"
    barcode = "example-barcode"
    overwrite = True

    output_name = "OME-Zarr"

    metatada_update = create_ome_zarr_md(
        input_paths=input_paths,
        output_path=zarr_root,
        metadata={},
        zarr_name=output_name,
        mode=mode,
        order_name=order_name,
        barcode=barcode,
        overwrite=overwrite,
    )

    for component in metatada_update["image"]:
        md_to_ome_zarr(
            input_paths=[zarr_root],
            output_path=zarr_root,
            component=component,
            metadata=metatada_update,
        )

    assert (
        zarr_root
        / f"{output_name}.zarr"
        / "E"
        / "7"
        / "0"
        / "C00_FITC_05_histogram.npz"
    ).exists()
    assert (
        zarr_root
        / f"{output_name}.zarr"
        / "E"
        / "7"
        / "0"
        / "C01_FITC_05_histogram.npz"
    ).exists()
    assert (
        zarr_root / f"{output_name}.zarr" / "E" / "7" / "0" / "C02_empty_histogram.npz"
    ).exists()
    assert (
        zarr_root
        / f"{output_name}.zarr"
        / "E"
        / "7"
        / "0"
        / "C03_FITC_05_histogram.npz"
    ).exists()

    assert (
        zarr_root
        / f"{output_name}.zarr"
        / "E"
        / "7"
        / "0"
        / "tables"
        / "well_ROI_table"
    ).exists()
    assert (
        zarr_root / f"{output_name}.zarr" / "E" / "7" / "0" / "tables" / "FOV_ROI_table"
    ).exists()

    # Check ROI table content
    table = ad.read_zarr(
        zarr_root
        / f"{output_name}.zarr"
        / "E"
        / "7"
        / "0"
        / "tables"
        / "well_ROI_table"
    )
    df_well = table.to_df()
    roi_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    assert list(df_well.columns) == roi_columns
    assert len(df_well) == 1
    target_values = [
        0.0,
        0.0,
        0.0,
        1399.6031494140625,
        699.8015747070312,
        45.290000915527344,
    ]
    assert df_well.loc["well_1"].values.flatten().tolist() == target_values

    table = ad.read_zarr(
        zarr_root / f"{output_name}.zarr" / "E" / "7" / "0" / "tables" / "FOV_ROI_table"
    )
    df_fov = table.to_df()
    assert list(df_fov.columns) == roi_columns
    assert len(df_fov) == 2
    target_values = [
        0.0,
        699.8015747070312,
        0.0,
        699.8015747070312,
        699.8015747070312,
        45.290000915527344,
    ]
    assert df_fov.loc["Site 2"].values.flatten().tolist() == target_values


def test_ome_zarr_conversion_mode_2D_fractal_tasks():
    ROOT_DIR = Path(__file__).parent
    input_paths = [join(ROOT_DIR.parent.parent, "resources", "Projection-Mix")]
    tmp_dir = tempfile.mkdtemp()
    zarr_root = Path(tmp_dir, "zarr-files")
    zarr_root.mkdir()

    mode = "top-level"

    order_name = "example-order"
    barcode = "example-barcode"
    overwrite = True

    output_name = "OME-Zarr"

    metatada_update = create_ome_zarr_md(
        input_paths=input_paths,
        output_path=zarr_root,
        metadata={},
        zarr_name=output_name,
        mode=mode,
        order_name=order_name,
        barcode=barcode,
        overwrite=overwrite,
    )

    for component in metatada_update["image"]:
        md_to_ome_zarr(
            input_paths=[zarr_root],
            output_path=zarr_root,
            component=component,
            metadata=metatada_update,
        )

    # Check ROI table content
    table = ad.read_zarr(
        zarr_root
        / f"{output_name}.zarr"
        / "E"
        / "7"
        / "0"
        / "tables"
        / "well_ROI_table"
    )
    df_well = table.to_df()
    roi_columns = [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
    ]
    assert list(df_well.columns) == roi_columns
    assert len(df_well) == 1
    target_values = [0.0, 0.0, 0.0, 1399.6031494140625, 699.8015747070312, 1.0]
    assert df_well.loc["well_1"].values.flatten().tolist() == target_values

    table = ad.read_zarr(
        zarr_root / f"{output_name}.zarr" / "E" / "7" / "0" / "tables" / "FOV_ROI_table"
    )
    df_fov = table.to_df()
    assert list(df_fov.columns) == roi_columns
    assert len(df_fov) == 2
    target_values = [
        0.0,
        699.8015747070312,
        0.0,
        699.8015747070312,
        699.8015747070312,
        1.0,
    ]
    assert df_fov.loc["Site 2"].values.flatten().tolist() == target_values


def test_fractal_task_overwrite_false():
    ROOT_DIR = Path(__file__).parent
    input_paths = [join(ROOT_DIR.parent.parent, "resources", "Projection-Mix")]
    tmp_dir = tempfile.mkdtemp()
    zarr_root = Path(tmp_dir, "zarr-files")
    zarr_root.mkdir()

    mode = "all"

    order_name = "example-order"
    barcode = "example-barcode"
    overwrite = True

    output_name = "OME-Zarr"

    metatada_update = create_ome_zarr_md(
        input_paths=input_paths,
        output_path=zarr_root,
        metadata={},
        zarr_name=output_name,
        mode=mode,
        order_name=order_name,
        barcode=barcode,
        overwrite=overwrite,
    )

    overwrite = False

    with pytest.raises(FileExistsError):
        metatada_update = create_ome_zarr_md(
            input_paths=input_paths,
            output_path=zarr_root,
            metadata={},
            zarr_name=output_name,
            mode=mode,
            order_name=order_name,
            barcode=barcode,
            overwrite=overwrite,
        )
