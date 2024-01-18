import re
from pathlib import Path

import pytest
from tifffile import imread

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.cellvoyager import ZAdjustedStackAcquisition


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


@pytest.fixture
def trace_log_file() -> Path:
    return (
        Path(__file__).parent.parent.parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "TRACE2023-XX-XX.log"
    )


def test__parse_files(cv_acquisition, trace_log_file):
    plate = ZAdjustedStackAcquisition(
        acquisition_dir=cv_acquisition,
        trace_log_file=trace_log_file,
        alignment=TileAlignmentOptions.GRID,
    )

    files = plate._parse_files()
    assert len(files) == 96
    assert files["well"].unique().tolist() == ["D08", "E03", "F08"]
    assert files["Ch"].unique().tolist() == ["1", "2"]
    assert files["ZIndex"].unique().tolist() == [0, 1, 2, 3, 4, 5]
    assert files["Z"].unique().tolist() == [
        0.0,
        3.0,
        6.0,
        9.0,
        1.0,
        4.0,
        7.0,
        10.0,
        12.0,
        15.0,
    ]

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
        "z_pos",
    ]


def test_get_well_acquisitions(cv_acquisition, trace_log_file):
    plate = ZAdjustedStackAcquisition(
        acquisition_dir=cv_acquisition,
        trace_log_file=trace_log_file,
        alignment=TileAlignmentOptions.GRID,
    )

    wells = plate.get_well_acquisitions()
    assert len(wells) == 3
    for well in wells:
        for tile in well.get_tiles():
            file_name = (
                f".*[/\\\\]CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_"
                f"{well.name}_T"
                f"{str(tile.position.time + 1).zfill(4)}F.*L.*A.*Z.*C"
                f"{str(tile.position.channel + 1).zfill(2)}\\.tif"
            )
            re_file_name = re.compile(file_name)
            assert re_file_name.match(tile.path)
            assert tile.shape == imread(tile.path).shape
            assert tile.illumination_correction_matrix_path is None
            assert tile.background_correction_matrix_path is None
            assert tile.position.x in [0, 2000]
            assert tile.position.y in [0, 2000]
            assert tile.position.z in [0, 1, 2, 3, 4, 5]
