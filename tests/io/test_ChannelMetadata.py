from pathlib import Path

import pytest

from faim_ipa.io.metadata import ChannelMetadata
from faim_ipa.io.metaseries import load_metaseries_tiff_metadata


@pytest.fixture
def transmission_sample_image_path():
    return (
        Path(__file__).parent.parent.parent
        / "resources"
        / "ImageXpress"
        / "exp126-d0_C03_thumbA6B0784C-19ED-4F35-9E6A-0A306794BB11.tif"
    )


def test_transmission_metadata(transmission_sample_image_path):
    metadata = load_metaseries_tiff_metadata(transmission_sample_image_path)
    channel_metadata = ChannelMetadata(
        channel_index=0,
        channel_name="test_channel",
        display_color="FFFFFF",
        spatial_calibration_x=metadata["spatial-calibration-x"],
        spatial_calibration_y=metadata["spatial-calibration-y"],
        spatial_calibration_units=metadata["spatial-calibration-units"],
        objective="4x",
        **metadata,
    )
    assert channel_metadata.spatial_calibration_x == 43.3613
    assert channel_metadata.spatial_calibration_y == 54.2016
    assert channel_metadata.wavelength == 0
