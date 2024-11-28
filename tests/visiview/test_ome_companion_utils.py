from pathlib import Path

import pytest
from defusedxml.ElementTree import parse

from faim_ipa.io.metadata import ChannelMetadata
from faim_ipa.visiview.ome_companion_utils import (
    get_channels,
    get_exposure_time,
    get_stage_positions,
    get_yx_spacing,
    get_z_spacing,
    parse_basic_metadata,
)


@pytest.fixture
def ome_companion_file():
    return (
        Path(__file__).parent.parent.parent
        / "resources"
        / "ome-tiff"
        / "20241003_experiment_488_7_640_3_1.companion.ome"
    )


@pytest.fixture
def ome_companion(ome_companion_file):
    with open(ome_companion_file, "rb") as f:
        root = parse(f).getroot()

    return root


@pytest.fixture
def ome_companion_file2():
    return (
        Path(__file__).parent.parent.parent
        / "resources"
        / "ome-tiff"
        / "c3_z33.companion.ome"
    )


@pytest.fixture
def ome_companion2(ome_companion_file2):
    with open(ome_companion_file2, "rb") as f:
        root = parse(f).getroot()

    return root


def test_get_z_spacing(ome_companion):
    z_spacing = get_z_spacing(ome_companion)

    assert z_spacing is None


def test_get_z_spacing2(ome_companion2):
    z_spacing = get_z_spacing(ome_companion2)

    assert z_spacing == 0.5


def test_get_yx_spacing(ome_companion):
    yx_spacing = get_yx_spacing(ome_companion)

    assert yx_spacing == (0.16, 0.16)


def test_get_exposure_time(ome_companion):
    # For some ome-companion files this is not set.
    exposure_time, exposure_time_unit = get_exposure_time(ome_companion)

    assert exposure_time is None
    assert exposure_time_unit is None


def test_get_channels(ome_companion):
    channels = get_channels(ome_companion)

    assert len(channels) == 2
    print(channels)

    assert channels["w1"] == ChannelMetadata(
        channel_index=0,
        channel_name="Channel1",
        display_color="4aff00",
        spatial_calibration_x=0.16,
        spatial_calibration_y=0.16,
        spatial_calibration_units="um",
        z_spacing=None,
        wavelength=525,
        exposure_time=None,
        exposure_time_unit=None,
        objective="Objective:100x/1.49",
    )
    assert channels["w2"] == ChannelMetadata(
        channel_index=1,
        channel_name="Channel2",
        display_color="ce0000",
        spatial_calibration_x=0.16,
        spatial_calibration_y=0.16,
        spatial_calibration_units="um",
        z_spacing=None,
        wavelength=680,
        exposure_time=None,
        exposure_time_unit=None,
        objective="Objective:100x/1.49",
    )


def test_get_stage_positions(ome_companion):
    positions = get_stage_positions(ome_companion)

    assert positions == {"1": (0, 0)}


def test_parse_basic_metadata(ome_companion_file, ome_companion):
    metadata = parse_basic_metadata(ome_companion_file)

    assert metadata == {
        "z_spacing": get_z_spacing(ome_companion),
        "yx_spacing": get_yx_spacing(ome_companion),
        "channels": get_channels(ome_companion),
        "stage_positions": get_stage_positions(ome_companion),
    }
