from pathlib import Path

import pytest

from faim_hcs.io.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_hcs.io.ImageXpress import ImageXpressPlateAcquisition


@pytest.fixture
def acquisition_dir():
    return Path(__file__).parent.parent.parent / "resources" / "Projection-Mix"


@pytest.fixture
def top_level_acquisition(acquisition_dir):
    return ImageXpressPlateAcquisition(
        acquisition_dir, alignment=TileAlignmentOptions.GRID, mode="top-level"
    )


def test_top_level_acquistion(top_level_acquisition: PlateAcquisition):
    wells = top_level_acquisition.get_well_acquisitions()

    assert wells is not None
    assert len(wells) == 2
    assert len(top_level_acquisition._files) == 12

    channels = top_level_acquisition.get_channel_metadata()
    assert len(channels) == 3
    ch = channels["w1"]
    assert ch.channel_index == 0
    assert ch.channel_name == "w1"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == "cyan"
    assert ch.z_scaling is None

    ch = channels["w2"]
    assert ch.channel_index == 1
    assert ch.channel_name == "w2"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == "cyan"
    assert ch.z_scaling is None

    ch = channels["w3"]
    assert ch.channel_index == 2
    assert ch.channel_name == "w3"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == "cyan"
    assert ch.z_scaling is None

    for well in top_level_acquisition.get_well_acquisitions():
        assert isinstance(well, WellAcquisition)
        assert len(well.get_tiles()) == 6
        for tile in well.get_tiles():
            assert tile.position.time == 0
            assert tile.position.channel in [0, 1, 2]
            assert tile.position.z == 0
            assert tile.position.y in [0]
            assert tile.position.x in [0, 512]
            assert tile.shape == (512, 512)


@pytest.fixture
def z_step_acquisition(acquisition_dir):
    return ImageXpressPlateAcquisition(
        acquisition_dir, alignment=TileAlignmentOptions.GRID, mode="z-step"
    )


def test_z_step_acquisition(z_step_acquisition: PlateAcquisition):
    wells = z_step_acquisition.get_well_acquisitions()

    assert wells is not None
    assert len(wells) == 2
    assert len(z_step_acquisition._files) == 96

    channels = z_step_acquisition.get_channel_metadata()
    assert len(channels) == 4
    ch = channels["w1"]
    assert ch.channel_index == 0
    assert ch.channel_name == "w1"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == "cyan"
    assert ch.z_scaling == 5.0

    ch = channels["w2"]
    assert ch.channel_index == 1
    assert ch.channel_name == "w2"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == "cyan"
    assert ch.z_scaling == 5.0

    ch = channels["w3"]
    assert ch.channel_index == 2
    assert ch.channel_name == "w3"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == "cyan"
    assert ch.z_scaling is None

    ch = channels["w4"]
    assert ch.channel_index == 3
    assert ch.channel_name == "w4"
    assert ch.display_color == "73ff00"
    assert ch.exposure_time == 15.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "20X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3668
    assert ch.spatial_calibration_y == 1.3668
    assert ch.wavelength == "cyan"
    assert ch.z_scaling is None

    for well in z_step_acquisition.get_well_acquisitions():
        assert isinstance(well, WellAcquisition)
        assert len(well.get_tiles()) == 48
        for tile in well.get_tiles():
            assert tile.position.time == 0
            assert tile.position.channel in [0, 1, 2, 3]
            assert tile.position.z in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            assert tile.position.y in [0]
            assert tile.position.x in [0, 512]
            assert tile.shape == (512, 512)
