from pathlib import Path

import pytest

from faim_hcs.CellVoyagerUtils import (
    get_img_YX,
    get_well_image_CYX,
    get_well_image_CZYX,
)
from faim_hcs.io.YokogawaCellVoyager import parse_files, parse_metadata
from faim_hcs.MontageUtils import montage_stage_pos_image_YX


@pytest.fixture
def acquisition_dir():
    return (
        Path(__file__).parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack"
    )


@pytest.fixture
def files(acquisition_dir):
    return parse_files(acquisition_dir=acquisition_dir)


@pytest.fixture
def channel_metadata(acquisition_dir):
    _, channel_metadata = parse_metadata(acquistion_dir=acquisition_dir)
    return channel_metadata


def test_get_img_YX(files, channel_metadata):
    with pytest.raises(
        ValueError, match=".*get_img_YX requires files for a single channel only.*"
    ):
        get_img_YX(
            assemble_fn=montage_stage_pos_image_YX,
            files=files,
            channel_metadata=channel_metadata,
        )

    img, z_position, roi_tables = get_img_YX(
        assemble_fn=montage_stage_pos_image_YX,
        files=files[
            (files["well"] == "E03") & (files["ZIndex"] == "2") & (files["Ch"] == "1")
        ],
        channel_metadata=channel_metadata,
    )
    assert img.shape == (4000, 4000)
    assert z_position == 3.0
    assert roi_tables


def test_get_well_image_CYX(files, channel_metadata):
    files_Z2 = files[files["ZIndex"] == "2"]
    for well in files_Z2["well"].unique():
        img, hists, ch_metadata, metadata, roi_tables = get_well_image_CYX(
            well_files=files_Z2[files_Z2["well"] == well],
            channel_metadata_source=channel_metadata,
            channels=[
                "1",
                "2",
                "4",  # channel 4 is absent from files, but present in channel metadata
            ],
        )
        assert img.shape == (3, 4000, 4000)
        assert len(hists) == 3
        assert len(ch_metadata) == 3
        assert ch_metadata[0] == {
            "wavelength": "405",
            "exposure-time": 100.0,
            "exposure-time-unit": "ms",
            "channel-name": "405",
            "objective": "20x v2",
            "display-color": "002FFF",
        }
        assert ch_metadata[1] == {
            "wavelength": "488",
            "exposure-time": 100.0,
            "exposure-time-unit": "ms",
            "channel-name": "488",
            "objective": "20x v2",
            "display-color": "00FFA1",
        }
        assert ch_metadata[2] == {
            "channel-name": "empty",
            "display-color": "000000",
        }
        assert metadata == {
            "pixel-type": "uint16",
            "spatial-calibration-units": "um",
            "spatial-calibration-x": 0.325,
            "spatial-calibration-y": 0.325,
        }
        assert roi_tables


def test_get_well_image_ZCYX(files, channel_metadata):
    for well in files["well"].unique():
        well_files = files[files["well"] == well]
        img, hists, ch_metadata, metadata, roi_tables = get_well_image_CZYX(
            well_files=well_files,
            channel_metadata_source=channel_metadata,
            channels=["1", "2"],
        )
        assert img.shape == (2, 4, 4000, 4000)
        assert len(hists) == 2
        assert len(ch_metadata) == 2
        assert metadata == {
            "pixel-type": "uint16",
            "spatial-calibration-units": "um",
            "spatial-calibration-x": 0.325,
            "spatial-calibration-y": 0.325,
            "z-scaling": 3.0,
        }
        assert "FOV_ROI_table" in roi_tables
        assert "well_ROI_table" in roi_tables
