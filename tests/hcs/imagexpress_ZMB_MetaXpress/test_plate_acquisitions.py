from dataclasses import dataclass
from pathlib import Path

import pytest
from numpy.testing import assert_almost_equal

from faim_ipa.hcs.acquisition import (
    PlateAcquisition,
    TileAlignmentOptions,
    WellAcquisition,
)
from faim_ipa.hcs.imagexpress import (
    ImageXpressPlateAcquisition,
    MixedAcquisition,
    SinglePlaneAcquisition,
    StackAcquisition,
)


@pytest.fixture
def acquisition_dir():
    return (
        Path(__file__).parent.parent.parent.parent
        / "resources"
        / "ImageXpress_ZMB"
        / "MetaXpress"
        / "9987_Plate_3434"
    )


@pytest.fixture
def single_plane_acquisition(acquisition_dir):
    return SinglePlaneAcquisition(acquisition_dir, alignment=TileAlignmentOptions.GRID)


def test_single_plane_acquisition(single_plane_acquisition: PlateAcquisition):
    wells = single_plane_acquisition.get_well_acquisitions()

    assert wells is not None
    assert len(wells) == 2
    # MIPs: 1 well has 2 fields * 2 channels = 4 files
    # single planes: 1 well has 2 fields * 2 channels = 4 files
    assert len(wells[0]._files) == 8

    channels = single_plane_acquisition.get_channel_metadata()
    assert len(channels) == 4
    ch = channels[0]
    assert ch.channel_index == 0
    assert ch.channel_name == "Maximum-Projection_DAPI"
    assert ch.display_color == "0051ff"
    assert ch.exposure_time == 100.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "40X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3672
    assert ch.spatial_calibration_y == 1.3672
    assert ch.wavelength == 452
    assert ch.z_spacing is None

    ch = channels[1]
    assert ch.channel_index == 1
    assert ch.channel_name == "Maximum-Projection_FITC"
    assert ch.display_color == "35ff00"
    assert ch.exposure_time == 100.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "40X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3672
    assert ch.spatial_calibration_y == 1.3672
    assert ch.wavelength == 520
    assert ch.z_spacing is None

    ch = channels[2]
    assert ch.channel_index == 2
    assert ch.channel_name == "Texas Red"
    assert ch.display_color == "ff6700"
    assert ch.exposure_time == 100.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "40X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3672
    assert ch.spatial_calibration_y == 1.3672
    assert ch.wavelength == 624
    assert ch.z_spacing is None

    ch = channels[3]
    assert ch.channel_index == 3
    assert ch.channel_name == "Cy5"
    assert ch.display_color == "bc0000"
    assert ch.exposure_time == 100.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "40X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3672
    assert ch.spatial_calibration_y == 1.3672
    assert ch.wavelength == 692
    assert ch.z_spacing is None

    for well in single_plane_acquisition.get_well_acquisitions():
        assert isinstance(well, WellAcquisition)
        assert len(well.get_tiles()) == 8
        for tile in well.get_tiles():
            assert tile.position.time == 0
            assert tile.position.channel in [0, 1, 2, 3]
            assert tile.position.z == 0
            assert tile.position.y in [0]
            assert tile.position.x in [0, 256]
            assert tile.shape == (256, 256)


@pytest.fixture
def stack_acquisition(acquisition_dir):
    return StackAcquisition(acquisition_dir, alignment=TileAlignmentOptions.GRID)


def test_stack_acquisition(stack_acquisition: PlateAcquisition):
    wells = stack_acquisition.get_well_acquisitions()

    assert wells is not None
    assert len(wells) == 2
    # Full Stacks: 2 wells * 2 fields * 1 channels * 3 planes = 12 files
    # Single plane stacks: 2 wells * 2 fields * 2 channels * 3 plane = 24 files
    # MIP stacks: 2 wells * 2 fields * 1 channels * 3 plane = 12 files
    # Total of 48 files.
    # There are additionally 16 MIP files in the z=0 directory, but these are
    # ignored in this setup.
    assert len(wells[0]._files) + len(wells[1]._files) == 48

    channels = stack_acquisition.get_channel_metadata()
    assert len(channels) == 4
    ch = channels[0]
    assert ch.channel_index == 0
    assert ch.channel_name == "DAPI"
    assert ch.display_color == "0051ff"
    assert ch.exposure_time == 100.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "40X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3672
    assert ch.spatial_calibration_y == 1.3672
    assert ch.wavelength == 452
    assert_almost_equal(ch.z_spacing, 2.99, decimal=4)

    ch = channels[1]
    assert ch.channel_index == 1
    assert ch.channel_name == "Maximum-Projection_FITC"
    assert ch.display_color == "35ff00"
    assert ch.exposure_time == 100.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "40X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3672
    assert ch.spatial_calibration_y == 1.3672
    assert ch.wavelength == 520
    assert_almost_equal(ch.z_spacing, 2.99, decimal=4)

    ch = channels[2]
    assert ch.channel_index == 2
    assert ch.channel_name == "Texas Red"
    assert ch.display_color == "ff6700"
    assert ch.exposure_time == 100.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "40X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3672
    assert ch.spatial_calibration_y == 1.3672
    assert ch.wavelength == 624
    assert_almost_equal(ch.z_spacing, 2.99, decimal=4)

    ch = channels[3]
    assert ch.channel_index == 3
    assert ch.channel_name == "Cy5"
    assert ch.display_color == "bc0000"
    assert ch.exposure_time == 100.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "40X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3672
    assert ch.spatial_calibration_y == 1.3672
    assert ch.wavelength == 692
    assert_almost_equal(ch.z_spacing, 2.99, decimal=4)

    for well in stack_acquisition.get_well_acquisitions():
        assert isinstance(well, WellAcquisition)
        assert len(well.get_tiles()) == 24
        for tile in well.get_tiles():
            assert tile.position.time == 0
            assert tile.position.channel in [0, 1, 2, 3]
            assert tile.position.channel not in [4]
            assert tile.position.z in [0, 1, 2]
            assert tile.position.y in [0]
            assert tile.position.x in [0, 256]
            assert tile.shape == (256, 256)


def test_mixed_acquisition(acquisition_dir: Path):
    with pytest.raises(ValueError):
        MixedAcquisition(
            acquisition_dir,
            alignment=TileAlignmentOptions.GRID,
        )


@pytest.fixture
def dummy_plate():
    ImageXpressPlateAcquisition.__abstractmethods__ = set()

    @dataclass
    class Plate(ImageXpressPlateAcquisition):
        pass

    return Plate()


def test_raise_not_implemented_error(dummy_plate):
    with pytest.raises(NotImplementedError):
        dummy_plate._get_root_re()

    with pytest.raises(NotImplementedError):
        dummy_plate._get_filename_re()

    with pytest.raises(NotImplementedError):
        dummy_plate._get_z_spacing()


@pytest.fixture
def acquisition_dir_single_channel():
    return (
        Path(__file__).parent.parent.parent.parent
        / "resources"
        / "ImageXpress_ZMB"
        / "MetaXpress"
        / "9987_Plate_3420"
    )


@pytest.fixture
def single_channel_acquisition(acquisition_dir_single_channel):
    return SinglePlaneAcquisition(
        acquisition_dir_single_channel, alignment=TileAlignmentOptions.GRID
    )


def test_single_channel_acquisition(single_channel_acquisition: PlateAcquisition):
    wells = single_channel_acquisition.get_well_acquisitions()

    assert wells is not None
    assert len(wells) == 1
    # MIPs: 1 well has 1 fields * 1 channels = 1 files
    assert len(wells[0]._files) == 1

    channels = single_channel_acquisition.get_channel_metadata()
    assert len(channels) == 1
    ch = channels[0]
    assert ch.channel_index == 0
    assert ch.channel_name == "DAPI"
    assert ch.display_color == "0051ff"
    assert ch.exposure_time == 100.0
    assert ch.exposure_time_unit == "ms"
    assert ch.objective == "40X Plan Apo Lambda"
    assert ch.spatial_calibration_units == "um"
    assert ch.spatial_calibration_x == 1.3672
    assert ch.spatial_calibration_y == 1.3672
    assert ch.wavelength == 452
    assert ch.z_spacing is None

    for well in single_channel_acquisition.get_well_acquisitions():
        assert isinstance(well, WellAcquisition)
        assert len(well.get_tiles()) == 1
        for tile in well.get_tiles():
            assert tile.position.time == 0
            assert tile.position.channel in [0]
            assert tile.position.z == 0
            assert tile.position.y in [0]
            assert tile.position.x in [0]
            assert tile.shape == (256, 256)


# TODO: add test for time-lapse


def test_single_field_stack_acquisition(stack_acquisition: PlateAcquisition):
    # Regular z spacing in dataset
    files = stack_acquisition._parse_files()
    z_step = stack_acquisition._compute_z_spacing(files)
    assert_almost_equal(z_step, 2.99, decimal=4)

    # Select the subset of only a single field to process
    files = files[files["field"] == "s1"]
    # When only a single field is available per well, the files dataframe
    # has it set to None
    files["field"] = None
    z_step = stack_acquisition._compute_z_spacing(files)
    assert_almost_equal(z_step, 2.99, decimal=4)
