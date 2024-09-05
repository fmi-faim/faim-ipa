from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from faim_ipa.hcs.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_ipa.hcs.source import FileSource
from faim_ipa.io.metadata import ChannelMetadata
from faim_ipa.stitching.tile import Tile, TilePosition


@pytest.fixture
def dummy():
    PlateAcquisition.__abstractmethods__ = set()

    @dataclass
    class Plate(PlateAcquisition):
        _acquisition_dir = "acquisition_dir"
        _files = pd.read_csv(
            Path(__file__).parent / "imagexpress" / "files.csv", index_col=0
        )
        _alignment = TileAlignmentOptions.STAGE_POSITION

    return Plate()


def test_get_well_acquisitions(dummy):
    class DummyWell(WellAcquisition):
        name = "E07"

    dummy._wells = [DummyWell]
    wells = dummy.get_well_acquisitions()
    assert len(wells) == 1
    assert wells[0].name == "E07"
    wells = dummy.get_well_acquisitions(["E08"])
    assert len(wells) == 0


def test_raise_not_implemented_error(dummy):
    with pytest.raises(NotImplementedError):
        dummy._parse_files()

    with pytest.raises(NotImplementedError):
        dummy._build_well_acquisitions(files=pd.DataFrame())

    with pytest.raises(NotImplementedError):
        dummy.get_channel_metadata()


def test_get_well_names(dummy):
    WellAcquisition.__abstractmethods__ = set()

    class DummyWell(WellAcquisition):
        def _align_tiles(self, tiles: list[Tile]) -> list[Tile]:
            pass

        def _assemble_tiles(self) -> list[Tile]:
            pass

    dummy._wells = [
        DummyWell(
            files=pd.read_csv(
                Path(__file__).parent / "imagexpress" / "files.csv", index_col=0
            ),
            alignment=TileAlignmentOptions.GRID,
            background_correction_matrices=None,
            illumination_correction_matrices=None,
        )
    ]

    assert list(dummy.get_well_names()) == ["E07"]
    assert list(dummy.get_well_names(["E07"])) == ["E07"]
    assert list(dummy.get_well_names(["E08"])) == []


def test_get_omero_channel_metadata(dummy):
    dummy.get_channel_metadata = lambda: {
        1: ChannelMetadata(
            channel_index=1,
            channel_name="name",
            display_color="FFFFFF",
            spatial_calibration_x=1,
            spatial_calibration_y=2,
            spatial_calibration_units="um",
            z_spacing=3,
            wavelength=432,
            exposure_time=0.1,
            exposure_time_unit="ms",
            objective="20x",
        ),
        2: ChannelMetadata(
            channel_index=2,
            channel_name="no-name",
            display_color="FFFFAA",
            spatial_calibration_x=1,
            spatial_calibration_y=2,
            spatial_calibration_units="um",
            z_spacing=3,
            wavelength=432,
            exposure_time=0.1,
            exposure_time_unit="ms",
            objective="20x",
        ),
    }

    ome_metadata = dummy.get_omero_channel_metadata()
    assert len(ome_metadata) == 3
    assert ome_metadata[0] == {
        "active": False,
        "coefficient": 1,
        "color": "#000000",
        "family": "linear",
        "inverted": False,
        "label": "empty",
        "wavelength_id": "C01",
        "window": {
            "min": np.iinfo(np.uint16).min,
            "max": np.iinfo(np.uint16).max,
            "start": np.iinfo(np.uint16).min,
            "end": np.iinfo(np.uint16).max,
        },
    }
    assert ome_metadata[1] == {
        "active": True,
        "coefficient": 1,
        "color": "FFFFFF",
        "family": "linear",
        "inverted": False,
        "label": "name",
        "wavelength_id": "C02",
        "window": {
            "min": np.iinfo(np.uint16).min,
            "max": np.iinfo(np.uint16).max,
            "start": np.iinfo(np.uint16).min,
            "end": np.iinfo(np.uint16).max,
        },
    }
    assert ome_metadata[2] == {
        "active": True,
        "coefficient": 1,
        "color": "FFFFAA",
        "family": "linear",
        "inverted": False,
        "label": "no-name",
        "wavelength_id": "C03",
        "window": {
            "min": np.iinfo(np.uint16).min,
            "max": np.iinfo(np.uint16).max,
            "start": np.iinfo(np.uint16).min,
            "end": np.iinfo(np.uint16).max,
        },
    }


def test_get_common_well_shape(dummy):
    WellAcquisition.__abstractmethods__ = set()

    class DummyWellA(WellAcquisition):
        name = "A01"

        def _align_tiles(self, tiles: list[Tile]) -> list[Tile]:
            pass

        def _assemble_tiles(self) -> list[Tile]:
            pass

        def get_shape(self):
            return (1, 1, 3, 11, 13)

    class DummyWellB(WellAcquisition):
        name = "A01"

        def _align_tiles(self, tiles: list[Tile]) -> list[Tile]:
            pass

        def _assemble_tiles(self) -> list[Tile]:
            pass

        def get_shape(self):
            return (1, 1, 3, 10, 23)

    dummy._wells = [
        DummyWellA(
            files=pd.read_csv(
                Path(__file__).parent / "imagexpress" / "files.csv", index_col=0
            ),
            alignment=TileAlignmentOptions.GRID,
            background_correction_matrices=None,
            illumination_correction_matrices=None,
        ),
        DummyWellB(
            files=pd.read_csv(
                Path(__file__).parent / "imagexpress" / "files.csv", index_col=0
            ),
            alignment=TileAlignmentOptions.GRID,
            background_correction_matrices=None,
            illumination_correction_matrices=None,
        ),
    ]

    assert dummy.get_common_well_shape() == (1, 1, 3, 11, 23)


@pytest.fixture
def dummy_well():
    WellAcquisition.__abstractmethods__ = set()

    @dataclass
    class Well(WellAcquisition):
        name = "A01"
        _files = pd.read_csv(
            Path(__file__).parent / "imagexpress" / "files.csv", index_col=0
        )
        _alignment = TileAlignmentOptions.STAGE_POSITION

    return Well()


def test_raise_not_implemented_errors(dummy_well):
    with pytest.raises(NotImplementedError):
        dummy_well._assemble_tiles()

    with pytest.raises(NotImplementedError):
        dummy_well.get_axes()

    with pytest.raises(NotImplementedError):
        dummy_well.get_yx_spacing()

    with pytest.raises(NotImplementedError):
        dummy_well.get_z_spacing()


def test_get_coordinate_transformations_3d(dummy_well):
    dummy_well.get_z_spacing = lambda: 1.0
    dummy_well.get_yx_spacing = lambda: (2.0, 3.0)
    ct = dummy_well.get_coordinate_transformations(
        max_layer=2,
        yx_binning=1,
        ndim=4,
    )
    assert len(ct) == 3
    assert ct[0] == [
        {
            "scale": [
                1.0,
                1.0,
                2.0,
                3.0,
            ],
            "type": "scale",
        }
    ]
    assert ct[1] == [
        {
            "scale": [
                1.0,
                1.0,
                4.0,
                6.0,
            ],
            "type": "scale",
        }
    ]
    assert ct[2] == [
        {
            "scale": [
                1.0,
                1.0,
                8.0,
                12.0,
            ],
            "type": "scale",
        }
    ]


def test_get_coordinate_transformations_2d(dummy_well):
    dummy_well.get_z_spacing = lambda: None
    dummy_well.get_yx_spacing = lambda: (2.0, 3.0)
    ct = dummy_well.get_coordinate_transformations(max_layer=2, yx_binning=1, ndim=3)
    assert len(ct) == 3
    assert ct[0] == [
        {
            "scale": [
                1.0,
                2.0,
                3.0,
            ],
            "type": "scale",
        }
    ]
    assert ct[1] == [
        {
            "scale": [
                1.0,
                4.0,
                6.0,
            ],
            "type": "scale",
        }
    ]
    assert ct[2] == [
        {
            "scale": [
                1.0,
                8.0,
                12.0,
            ],
            "type": "scale",
        }
    ]


def test_get_dtype(dummy_well):
    dummy_well._tiles = [
        Tile(
            source=FileSource(
                directory=Path(__file__).parent.parent.parent
                / "resources"
                / "CV8000"
                / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
                / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack"
            ),
            path="CV8000-Minimal-DataSet-2C-3W-4S-FP2-"
            "stack_D08_T0001F001L01A01Z01C01.tif",
            shape=(2000, 2000),
            position=TilePosition(time=0, channel=0, z=0, y=0, x=0),
        )
    ]

    assert dummy_well.get_dtype() == np.uint16


def test_get_row_col(dummy_well):
    assert dummy_well.get_row_col() == ("A", "01")


def test_align_tiles(dummy_well):
    dummy_well._alignment = TileAlignmentOptions.STAGE_POSITION
    dummy_well._tiles = [
        Tile(
            source=FileSource(
                directory=Path(__file__).parent.parent.parent
                / "resources"
                / "CV8000"
                / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
                / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack"
            ),
            path="CV8000-Minimal-DataSet-2C-3W-4S-FP2"
            "-stack_D08_T0001F001L01A01Z01C01.tif",
            shape=(2000, 2000),
            position=TilePosition(time=1, channel=2, z=0, y=3, x=-70),
        )
    ]

    aligned = dummy_well._align_tiles(dummy_well._tiles)

    assert len(aligned) == len(dummy_well._tiles)
    assert aligned[0].position.time == 0
    assert aligned[0].position.channel == 0
    assert aligned[0].position.z == 0
    assert aligned[0].position.x == 0
    assert aligned[0].position.y == 0

    assert dummy_well.get_dtype() == np.uint16

    dummy_well._alignment = TileAlignmentOptions.GRID
    dummy_well._tiles = [
        Tile(
            source=FileSource(
                directory=Path(__file__).parent.parent.parent
                / "resources"
                / "CV8000"
                / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
                / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack"
            ),
            path="CV8000-Minimal-DataSet-2C-3W-4S-FP2"
            "-stack_D08_T0001F001L01A01Z01C01.tif",
            shape=(2000, 2000),
            position=TilePosition(time=1, channel=2, z=0, y=3, x=-70),
        )
    ]

    aligned = dummy_well._align_tiles(dummy_well._tiles)

    assert len(aligned) == len(dummy_well._tiles)
    assert aligned[0].position.time == 0
    assert aligned[0].position.channel == 0
    assert aligned[0].position.z == 0
    assert aligned[0].position.x == 0
    assert aligned[0].position.y == 0

    assert dummy_well.get_dtype() == np.uint16


def test_alignment_not_implemented(dummy_well):
    dummy_well._alignment = "Unknown"
    with pytest.raises(ValueError, match="Unknown alignment option"):
        dummy_well._align_tiles(dummy_well._tiles)


def test_get_shape(dummy_well):

    dummy_well._tiles = [
        Tile(
            source=FileSource(
                directory=Path(__file__).parent.parent.parent
                / "resources"
                / "CV8000"
                / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
                / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack"
            ),
            path="CV8000-Minimal-DataSet-2C-3W-4S-FP2"
            "-stack_D08_T0001F001L01A01Z01C01.tif",
            shape=(2000, 2000),
            position=TilePosition(time=0, channel=0, z=0, y=0, x=0),
        )
    ]
    assert dummy_well.get_shape() == (1, 1, 1, 2000, 2000)
