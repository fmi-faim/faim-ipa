import os
from os.path import join
from pathlib import Path

import pandas as pd
import pytest

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.cellvoyager.CellVoyagerWellAcquisition import (
    CellVoyagerWellAcquisition,
)


@pytest.fixture
def files() -> pd.DataFrame:
    resource_dir = Path(__file__).parent.parent.parent.parent

    files = pd.read_csv(join(Path(__file__).parent, "files.csv"), index_col=0)

    files["path"] = files.apply(lambda row: join(resource_dir, row["path"]), axis=1)

    return files


@pytest.fixture
def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ch": [1, 2],
            "VerticalPixels": [2000, 2000],
            "HorizontalPixels": [2000, 2000],
            "VerticalPixelDimension": [0.65, 0.65],
            "HorizontalPixelDimension": [0.65, 0.65],
        }
    )


def test__assemble_tiles(files, metadata):
    cv_well_acquisition = CellVoyagerWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        metadata=metadata,
    )

    tiles = cv_well_acquisition._assemble_tiles()
    assert len(tiles) == 32
    for tile in tiles:
        assert os.path.exists(tile.path)
        assert tile.shape == (2000, 2000)
        assert tile.position.channel in [1, 2]
        assert tile.position.time == 1
        assert tile.position.z in [1, 2, 3, 4]
        assert tile.position.y in list(-(files["Y"].unique() / 0.65).astype(int))
        assert tile.position.x in list((files["X"].unique() / 0.65).astype(int))


def test_get_axes(files, metadata):
    cv_well_acquisition = CellVoyagerWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        metadata=metadata,
    )
    axes = cv_well_acquisition.get_axes()
    assert axes == ["c", "z", "y", "x"]

    cv_well_acquisition = CellVoyagerWellAcquisition(
        files=files.drop(["Z", "ZIndex"], axis=1),
        alignment=TileAlignmentOptions.GRID,
        metadata=metadata,
    )
    axes = cv_well_acquisition.get_axes()
    assert axes == ["c", "y", "x"]


def test_get_yx_spacing(files, metadata):
    cv_well_acquisition = CellVoyagerWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        metadata=metadata,
    )
    yx_spacing = cv_well_acquisition.get_yx_spacing()
    assert yx_spacing == (0.65, 0.65)


def test__compute_z_spacing(files, metadata):
    cv_well_acquisition = CellVoyagerWellAcquisition(
        files=files, alignment=TileAlignmentOptions.GRID, metadata=metadata
    )
    z_spacing = cv_well_acquisition._compute_z_spacing(files)
    assert z_spacing == 3.0

    yx_spacing = cv_well_acquisition.get_yx_spacing()
    assert yx_spacing == (0.65, 0.65)


def test_get_z_spacing(files, metadata):
    cv_well_acquisition = CellVoyagerWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        metadata=metadata,
    )

    z_spacing = cv_well_acquisition.get_z_spacing()
    assert z_spacing == 3.0


def test_bgcm(files, metadata):
    cv_well_acquisition = CellVoyagerWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        metadata=metadata,
        background_correction_matrices={"1": "bgcm1", "2": "bgcm2"},
    )

    tiles = cv_well_acquisition._assemble_tiles()
    for tile in tiles:
        if tile.position.channel == 1:
            assert tile.background_correction_matrix_path == "bgcm1"

        if tile.position.channel == 2:
            assert tile.background_correction_matrix_path == "bgcm2"


def test_icm(files, metadata):
    cv_well_acquisition = CellVoyagerWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        metadata=metadata,
        illumination_correction_matrices={"1": "icm1", "2": "icm2"},
    )

    tiles = cv_well_acquisition._assemble_tiles()
    for tile in tiles:
        if tile.position.channel == 1:
            assert tile.illumination_correction_matrix_path == "icm1"

        if tile.position.channel == 2:
            assert tile.illumination_correction_matrix_path == "icm2"
