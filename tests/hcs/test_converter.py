from os.path import exists, join
from pathlib import Path

import dask
import numpy as np
import pytest
import zarr
from distributed.utils_test import (
    cleanup,  # noqa: F401
    client,  # noqa: F401
    cluster_fixture,  # noqa: F401
    loop,  # noqa: F401
    loop_in_thread,  # noqa: F401
)
from numcodecs import Blosc

from faim_ipa import dask_utils
from faim_ipa.hcs.acquisition import TileAlignmentOptions
from faim_ipa.hcs.cellvoyager import StackAcquisition
from faim_ipa.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_ipa.hcs.plate import PlateLayout
from faim_ipa.stitching.tile import Tile, TilePosition


def test_ngff_plate():
    root_dir = "/path/to/root_dir"
    name = "plate_name"
    layout = PlateLayout.I18
    order_name = "order_name"
    barcode = "barcode"
    plate = NGFFPlate(
        root_dir=root_dir,
        name=name,
        layout=layout,
        order_name=order_name,
        barcode=barcode,
    )
    assert Path(plate.root_dir) == Path(root_dir)
    assert plate.name == name
    assert plate.layout == layout
    assert plate.order_name == order_name
    assert plate.barcode == barcode


@pytest.fixture
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("hcs_plate")


@pytest.fixture
def hcs_plate(tmp_dir):
    return NGFFPlate(
        root_dir=tmp_dir,
        name="plate_name",
        layout=PlateLayout.I96,
        order_name="order_name",
        barcode="barcode",
    )


@pytest.fixture
def plate_acquisition():
    return StackAcquisition(
        acquisition_dir=Path(__file__).parent.parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack",
        alignment=TileAlignmentOptions.GRID,
    )


@pytest.fixture
def plate_acquisition_2d():
    acq = StackAcquisition(
        acquisition_dir=Path(__file__).parent.parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack",
        alignment=TileAlignmentOptions.GRID,
    )

    for well in acq.get_well_acquisitions():
        tiles = well.get_tiles()
        new_tiles = [
            (
                Tile(
                    path=tile._paths[0],
                    shape=(2000, 2000),
                    position=TilePosition(
                        time=tile.position.time,
                        channel=tile.position.channel,
                        z=tile.position.z,
                        y=tile.position.y,
                        x=tile.position.x,
                    ),
                )
            )
            for tile in tiles
        ]

        well._tiles = new_tiles

    return acq


def test__create_zarr_plate(tmp_dir, plate_acquisition, hcs_plate, client):
    converter = ConvertToNGFFPlate(hcs_plate, client=client)
    zarr_plate = converter.create_zarr_plate(plate_acquisition, wells=None)

    assert exists(join(tmp_dir, "plate_name.zarr"))
    assert zarr_plate.attrs["plate"]["name"] == "plate_name"
    assert zarr_plate.attrs["order_name"] == "order_name"
    assert zarr_plate.attrs["barcode"] == "barcode"
    assert zarr_plate.attrs["plate"]["field_count"] == 1
    assert zarr_plate.attrs["plate"]["wells"] == [
        {"columnIndex": 7, "path": "D/08", "rowIndex": 3},
        {"columnIndex": 2, "path": "E/03", "rowIndex": 4},
        {"columnIndex": 7, "path": "F/08", "rowIndex": 5},
    ]
    assert exists(join(tmp_dir, "plate_name.zarr", ".zgroup"))
    assert exists(join(tmp_dir, "plate_name.zarr", ".zattrs"))
    assert not exists(join(tmp_dir, "plate_name.zarr", "D"))
    assert not exists(join(tmp_dir, "plate_name.zarr", "E"))
    assert not exists(join(tmp_dir, "plate_name.zarr", "F"))

    zarr_plate_1 = converter.create_zarr_plate(plate_acquisition)
    assert zarr_plate_1 == zarr_plate


def test__out_chunks():
    out_chunks = ConvertToNGFFPlate._out_chunks(
        shape=(1, 2, 5, 10, 10),
        chunks=(1, 1, 5, 10, 5),
    )
    assert out_chunks == (1, 1, 5, 10, 5)

    out_chunks = ConvertToNGFFPlate._out_chunks(
        shape=(1, 2, 5, 10, 10),
        chunks=(5, 10, 10),
    )
    assert out_chunks == (1, 1, 5, 10, 10)


def test__get_storage_options():
    storage_options = ConvertToNGFFPlate._get_storage_options(
        storage_options=None,
        output_shape=(1, 2, 5, 10, 10),
        chunks=(5, 10, 5),
    )
    assert storage_options == {
        "dimension_separator": "/",
        "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        "chunks": (1, 1, 5, 10, 5),
        "write_empty_chunks": False,
    }

    storage_options = ConvertToNGFFPlate._get_storage_options(
        storage_options={
            "dimension_separator": ".",
        },
        output_shape=(1, 2, 5, 10, 10),
        chunks=(5, 10, 5),
    )
    assert storage_options == {
        "dimension_separator": ".",
    }


def test__mean_cast_to():
    mean_cast_to = dask_utils.mean_cast_to(np.uint8)
    input_array = np.array([1.0, 2.0], dtype=np.float32)
    assert input_array.dtype == np.float32
    assert mean_cast_to(input_array).dtype == np.uint8
    assert mean_cast_to(input_array) == 1


def test__create_well_group(tmp_dir, plate_acquisition, hcs_plate, client):
    converter = ConvertToNGFFPlate(hcs_plate, client=client)
    zarr_plate = converter.create_zarr_plate(plate_acquisition)
    well_group = converter._create_well_group(
        plate=zarr_plate,
        well_acquisition=plate_acquisition.get_well_acquisitions()[0],
        well_sub_group="0",
    )
    assert exists(join(tmp_dir, "plate_name.zarr", "D", "08", "0"))
    assert isinstance(well_group, zarr.Group)

    mask_group = converter._create_well_group(
        plate=zarr_plate,
        well_acquisition=plate_acquisition.get_well_acquisitions()[0],
        well_sub_group="0/mask",
        add_to_well_images=False,
    )
    assert exists(join(tmp_dir, "plate_name.zarr", "D", "08", "0", "mask"))
    assert isinstance(mask_group, zarr.Group)

    assert mask_group.attrs.asdict()["well"]["images"] == [{"path": "0"}]

    another_group = converter._create_well_group(
        plate=zarr_plate,
        well_acquisition=plate_acquisition.get_well_acquisitions()[0],
        well_sub_group="0/another",
        add_to_well_images=True,
    )
    assert exists(join(tmp_dir, "plate_name.zarr", "D", "08", "0", "another"))
    assert isinstance(another_group, zarr.Group)
    assert another_group.attrs.asdict()["well"]["images"] == [
        {"path": "0"},
        {"path": "0/another"},
    ]


def test__stitch_well_image_2d(plate_acquisition_2d, hcs_plate, client):
    converter = ConvertToNGFFPlate(hcs_plate, client=client)
    well_acquisition = plate_acquisition_2d.get_well_acquisitions()[0]
    well_img_da = converter._stitch_well_image(
        chunks=(1, 1, 10, 1000, 1000),
        well_acquisition=well_acquisition,
        output_shape=plate_acquisition_2d.get_common_well_shape(),
        build_acquisition_mask=False,
    )
    assert isinstance(well_img_da, dask.array.core.Array)
    assert well_img_da.shape == (1, 2, 4, 4000, 4000)
    assert well_img_da.dtype == np.uint16


def test__stitch_well_image_mask_2d(plate_acquisition_2d, hcs_plate, client):
    converter = ConvertToNGFFPlate(hcs_plate, client=client)
    well_acquisition = plate_acquisition_2d.get_well_acquisitions()[0]
    well_img_da = converter._stitch_well_image(
        chunks=(1, 1, 10, 1000, 1000),
        well_acquisition=well_acquisition,
        output_shape=plate_acquisition_2d.get_common_well_shape(),
        build_acquisition_mask=True,
    )
    assert isinstance(well_img_da, dask.array.core.Array)
    assert well_img_da.shape == (1, 2, 4, 4000, 4000)
    assert well_img_da.dtype == bool


def test__stitch_well_image_3d(plate_acquisition, hcs_plate, client):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        client=client,
    )
    well_acquisition = plate_acquisition.get_well_acquisitions()[0]
    well_img_da = converter._stitch_well_image(
        chunks=(1, 1, 10, 1000, 1000),
        well_acquisition=well_acquisition,
        output_shape=plate_acquisition.get_common_well_shape(),
        build_acquisition_mask=False,
    )
    assert isinstance(well_img_da, dask.array.core.Array)
    assert well_img_da.shape == (1, 2, 4, 4000, 4000)
    assert well_img_da.dtype == np.uint16


def test__stitch_well_image_mask_3d(plate_acquisition, hcs_plate, client):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        client=client,
    )
    well_acquisition = plate_acquisition.get_well_acquisitions()[0]
    well_img_da = converter._stitch_well_image(
        chunks=(1, 1, 10, 1000, 1000),
        well_acquisition=well_acquisition,
        output_shape=plate_acquisition.get_common_well_shape(),
        build_acquisition_mask=True,
    )
    assert isinstance(well_img_da, dask.array.core.Array)
    assert well_img_da.shape == (1, 2, 4, 4000, 4000)
    assert well_img_da.dtype == bool


def test__bin_yx(plate_acquisition, hcs_plate, client):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        yx_binning=2,
        client=client,
    )
    well_acquisition = plate_acquisition.get_well_acquisitions()[0]
    well_img_da = converter._stitch_well_image(
        chunks=(1, 1, 10, 1000, 1000),
        well_acquisition=well_acquisition,
        output_shape=plate_acquisition.get_common_well_shape(),
        build_acquisition_mask=False,
    )
    binned_yx = converter._bin_yx(well_img_da)
    assert isinstance(binned_yx, dask.array.core.Array)
    assert binned_yx.shape == (1, 2, 4, 2000, 2000)
    assert binned_yx.dtype == np.uint16

    converter._yx_binning = 1
    binned_yx = converter._bin_yx(well_img_da)
    assert isinstance(binned_yx, dask.array.core.Array)
    assert binned_yx.shape == (1, 2, 4, 4000, 4000)
    assert binned_yx.dtype == np.uint16


def test_run(tmp_dir, plate_acquisition, hcs_plate, client):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        yx_binning=2,
        client=client,
    )
    plate = converter.create_zarr_plate(plate_acquisition)
    plate = converter.run(
        plate=plate,
        plate_acquisition=plate_acquisition,
        max_layer=2,
        chunks=(1, 2000, 2000),
    )
    assert plate.attrs["plate"]["wells"] == [
        {"columnIndex": 7, "path": "D/08", "rowIndex": 3},
        {"columnIndex": 2, "path": "E/03", "rowIndex": 4},
        {"columnIndex": 7, "path": "F/08", "rowIndex": 5},
    ]
    for well in ["D08", "E03", "F08"]:
        row, col = well[0], well[1:]
        path = join(tmp_dir, "plate_name.zarr", row, col, "0")
        assert exists(path)

        assert exists(join(path, "0"))
        assert exists(join(path, "1"))
        assert exists(join(path, ".zattrs"))
        assert exists(join(path, ".zgroup"))

        assert "acquisition_metadata" in plate[row][col]["0"].attrs
        assert "multiscales" in plate[row][col]["0"].attrs
        assert "omero" in plate[row][col]["0"].attrs

        axes = plate[row][col]["0"].attrs["multiscales"][0]["axes"]
        for axis in axes:
            if axis["type"] == "space":
                assert "unit" in axis
                assert axis["unit"] == "micrometer"

        assert exists(join(path, "0", ".zarray"))
        assert exists(join(path, "1", ".zarray"))

        assert plate[row][col]["0"]["0"].shape == (2, 4, 2000, 2000)
        assert plate[row][col]["0"]["1"].shape == (2, 4, 1000, 1000)


def test_run_selection(tmp_dir, plate_acquisition, hcs_plate, client):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        yx_binning=2,
        client=client,
    )
    plate = converter.create_zarr_plate(plate_acquisition)
    plate = converter.run(
        plate=plate,
        plate_acquisition=plate_acquisition,
        max_layer=2,
        wells=["D08"],
        chunks=(1, 2000, 2000),
    )
    assert plate.attrs["plate"]["wells"] == [
        {"columnIndex": 7, "path": "D/08", "rowIndex": 3},
        {"columnIndex": 2, "path": "E/03", "rowIndex": 4},
        {"columnIndex": 7, "path": "F/08", "rowIndex": 5},
    ]
    for well in ["D08"]:
        row, col = well[0], well[1:]
        path = join(tmp_dir, "plate_name.zarr", row, col, "0")
        assert exists(path)

        assert exists(join(path, "0"))
        assert exists(join(path, "1"))
        assert exists(join(path, ".zattrs"))
        assert exists(join(path, ".zgroup"))

        assert "acquisition_metadata" in plate[row][col]["0"].attrs
        assert "multiscales" in plate[row][col]["0"].attrs
        assert "omero" in plate[row][col]["0"].attrs

        assert exists(join(path, "0", ".zarray"))
        assert exists(join(path, "1", ".zarray"))

        assert plate[row][col]["0"]["0"].shape == (2, 4, 2000, 2000)
        assert plate[row][col]["0"]["1"].shape == (2, 4, 1000, 1000)


def test_provide_client(hcs_plate, client):
    converter = ConvertToNGFFPlate(
        hcs_plate,
        yx_binning=2,
        client=client,
    )
    assert converter._client is not None


def test__drop_missing_axes(plate_acquisition_2d, hcs_plate, client):
    converter = ConvertToNGFFPlate(hcs_plate, client=client)
    well_acquisition = plate_acquisition_2d.get_well_acquisitions()[0]
    well_img_da = converter._stitch_well_image(
        chunks=(1, 1, 10, 1000, 1000),
        well_acquisition=well_acquisition,
        output_shape=plate_acquisition_2d.get_common_well_shape(),
        build_acquisition_mask=False,
    )

    well_acquisition.get_axes = lambda: ["t", "c", "z", "y", "x"]
    well_img_da = converter._drop_missing_axes(well_img_da, well_acquisition)
    assert well_img_da.shape == (1, 2, 4, 4000, 4000)

    well_acquisition.get_axes = lambda: ["c", "z", "y", "x"]
    well_img_da = converter._drop_missing_axes(well_img_da, well_acquisition)
    assert well_img_da.shape == (2, 4, 4000, 4000)
