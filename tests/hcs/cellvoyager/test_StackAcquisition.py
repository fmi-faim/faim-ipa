import re
from pathlib import Path

import pytest
from tifffile import imread

from faim_ipa.hcs.acquisition import TileAlignmentOptions
from faim_ipa.hcs.cellvoyager import StackAcquisition
from faim_ipa.hcs.cellvoyager.StackedTile import StackedTile


@pytest.fixture
def cv_acquisition() -> Path:
    dir = (
        Path(__file__).parent.parent.parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack"
    )
    return dir


def test_get_channel_metadata(cv_acquisition):
    plate = StackAcquisition(
        acquisition_dir=cv_acquisition,
        alignment=TileAlignmentOptions.GRID,
    )

    ch_metadata = plate.get_channel_metadata()
    assert len(ch_metadata) == 4
    assert ch_metadata[0].channel_name == "1"
    assert ch_metadata[0].display_color == "#FF002FFF"
    assert ch_metadata[0].spatial_calibration_x == 0.325
    assert ch_metadata[0].spatial_calibration_y == 0.325
    assert ch_metadata[0].spatial_calibration_units == "um"
    assert ch_metadata[0].z_spacing == 3.0
    assert ch_metadata[0].wavelength == 445
    assert ch_metadata[0].exposure_time == 100
    assert ch_metadata[0].exposure_time_unit == "ms"
    assert ch_metadata[0].objective == "20x v2"

    assert ch_metadata[1].channel_name == "2"
    assert ch_metadata[1].display_color == "#FF00FFA1"
    assert ch_metadata[1].spatial_calibration_x == 0.325
    assert ch_metadata[1].spatial_calibration_y == 0.325
    assert ch_metadata[1].spatial_calibration_units == "um"
    assert ch_metadata[1].z_spacing == 3.0
    assert ch_metadata[1].wavelength == 525
    assert ch_metadata[1].exposure_time == 100
    assert ch_metadata[1].exposure_time_unit == "ms"
    assert ch_metadata[1].objective == "20x v2"

    assert ch_metadata[2].channel_name == "3"
    assert ch_metadata[2].display_color == "#FFFF8200"
    assert ch_metadata[2].spatial_calibration_x == 0.325
    assert ch_metadata[2].spatial_calibration_y == 0.325
    assert ch_metadata[2].spatial_calibration_units == "um"
    assert ch_metadata[2].z_spacing == 3.0
    assert ch_metadata[2].wavelength == 600
    assert ch_metadata[2].exposure_time == 250
    assert ch_metadata[2].exposure_time_unit == "ms"
    assert ch_metadata[2].objective == "20x v2"

    assert ch_metadata[3].channel_name == "4"
    assert ch_metadata[3].display_color == "#FFFF1B00"
    assert ch_metadata[3].spatial_calibration_x == 0.325
    assert ch_metadata[3].spatial_calibration_y == 0.325
    assert ch_metadata[3].spatial_calibration_units == "um"
    assert ch_metadata[3].z_spacing == 3.0
    assert ch_metadata[3].wavelength == 676
    assert ch_metadata[3].exposure_time == 250
    assert ch_metadata[3].exposure_time_unit == "ms"
    assert ch_metadata[3].objective == "20x v2"


def test__parse_files(cv_acquisition):
    plate = StackAcquisition(
        acquisition_dir=cv_acquisition,
        alignment=TileAlignmentOptions.GRID,
    )

    files = plate._parse_files()
    assert len(files) == 96
    assert files["well"].unique().tolist() == ["D08", "E03", "F08"]
    assert files["Ch"].unique().tolist() == ["1", "2"]
    assert files["Z"].unique().tolist() == ["0.0", "3.0", "6.0", "9.0"]

    assert files.columns.tolist() == [
        "Time",
        "TimePoint",
        "FieldIndex",
        "ZIndex",
        "TimelineIndex",
        "ActionIndex",
        "Action",
        "X",
        "Y",
        "Z",
        "Ch",
        "path",
        "well",
    ]


def test_get_well_acquisitions(cv_acquisition):
    plate = StackAcquisition(
        acquisition_dir=cv_acquisition,
        alignment=TileAlignmentOptions.GRID,
    )

    wells = plate.get_well_acquisitions()
    assert len(wells) == 3
    for well in wells:
        for tile in well.get_tiles():
            file_name = (
                f".*[/\\\\]CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_"
                f"{well.name}_T"
                f"{str(tile.position.time + 1).zfill(4)}F.*L.*A.*Z"
                f"{str(tile.position.z + 1).zfill(2)}C"
                f"{str(tile.position.channel + 1).zfill(2)}.tif"
            )
            re_file_name = re.compile(file_name)
            assert isinstance(tile, StackedTile)
            assert re_file_name.match(tile._paths[0])
            assert tile.shape[1:] == imread(tile._paths[0]).shape
            assert tile.illumination_correction_matrix_path is None
            assert tile.background_correction_matrix_path is None
            assert tile.position.x in [0, 2000]
            assert tile.position.y in [0, 2000]


def test_raise_value_errors(cv_acquisition):
    with pytest.raises(ValueError):
        plate = StackAcquisition(
            acquisition_dir=".",
            alignment=TileAlignmentOptions.GRID,
        )

    with pytest.raises(ValueError):
        plate = StackAcquisition(
            acquisition_dir=cv_acquisition,
            alignment=TileAlignmentOptions.GRID,
        )
        # Change acquisition_dir to mock missing mrf and mes files.
        plate._acquisition_dir = "."
        plate._parse_metadata()
