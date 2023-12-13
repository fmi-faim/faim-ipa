import os
from os.path import join
from pathlib import Path

import pandas as pd
import pytest

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.imagexpress import ImageXpressWellAcquisition


@pytest.fixture
def files() -> pd.DataFrame:
    resource_dir = Path(__file__).parent.parent.parent.parent
    files = pd.read_csv(join(Path(__file__).parent, "files.csv"), index_col=0)

    files["path"] = files.apply(lambda row: join(resource_dir, row["path"]), axis=1)

    return files


def test__assemble_tiles(files):
    ix_well_acquisition = ImageXpressWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        z_spacing=3.0,
    )

    tiles = ix_well_acquisition._assemble_tiles()
    assert len(tiles) == 42
    for tile in tiles:
        assert os.path.exists(tile.path)
        assert tile.shape == (512, 512)
        assert tile.position.channel in [1, 2, 4]
        assert tile.position.time == 0
        assert tile.position.z in [
            3106,
            3107,
            3109,
            3111,
            3112,
            3114,
            3116,
            3117,
            3119,
            3121,
        ]


def test_get_axes(files):
    ix_well_acquisition = ImageXpressWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        z_spacing=3.0,
    )

    axes = ix_well_acquisition.get_axes()
    assert axes == ["c", "z", "y", "x"]

    ix_well_acquisition = ImageXpressWellAcquisition(
        files=files.drop("z", axis=1),
        alignment=TileAlignmentOptions.GRID,
        z_spacing=None,
    )

    axes = ix_well_acquisition.get_axes()
    assert axes == ["c", "y", "x"]


def test_get_yx_spacing(files):
    ix_well_acquisition = ImageXpressWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        z_spacing=3.0,
    )

    yx_spacing = ix_well_acquisition.get_yx_spacing()
    assert yx_spacing == (1.3668, 1.3668)


def test_get_z_spacing(files):
    ix_well_acquisition = ImageXpressWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        z_spacing=3.0,
    )
    assert ix_well_acquisition.get_z_spacing() == 3.0


def test_bgcm(files):
    ix_well_acquisition = ImageXpressWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        z_spacing=3.0,
        background_correction_matrices={"w1": "bgcm1", "w2": "bgcm2", "w4": "bgcm4"},
    )
    tiles = ix_well_acquisition._assemble_tiles()
    for tile in tiles:
        if tile.position.channel == 1:
            assert tile.background_correction_matrix_path == "bgcm1"
        elif tile.position.channel == 2:
            assert tile.background_correction_matrix_path == "bgcm2"
        elif tile.position.channel == 4:
            assert tile.background_correction_matrix_path == "bgcm4"
        else:
            assert tile.background_correction_matrix_path is None


def test_icm(files):
    ix_well_acquisition = ImageXpressWellAcquisition(
        files=files,
        alignment=TileAlignmentOptions.GRID,
        z_spacing=3.0,
        illumination_correction_matrices={"w1": "icm1", "w2": "icm2", "w4": "icm4"},
    )
    tiles = ix_well_acquisition._assemble_tiles()
    for tile in tiles:
        if tile.position.channel == 1:
            assert tile.illumination_correction_matrix_path == "icm1"
        elif tile.position.channel == 2:
            assert tile.illumination_correction_matrix_path == "icm2"
        elif tile.position.channel == 4:
            assert tile.illumination_correction_matrix_path == "icm4"
        else:
            assert tile.illumination_correction_matrix_path is None
