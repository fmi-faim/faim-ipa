import re
from pathlib import Path

import pytest

from faim_ipa.hcs.acquisition import TileAlignmentOptions
from faim_ipa.hcs.cellvoyager import ZAdjustedStackAcquisition
from faim_ipa.hcs.cellvoyager.source import CVSourceFS


@pytest.fixture
def cv_acquisition() -> Path:
    return (
        Path(__file__).parent.parent.parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack"
    )


@pytest.fixture
def trace_log_file() -> Path:
    return (
        Path(__file__).parent.parent.parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "TRACE2023-XX-XX.log"
    )


@pytest.fixture
def cv_source_filesystem(cv_acquisition, trace_log_file) -> CVSourceFS:
    return CVSourceFS(
        acquisition_dir=cv_acquisition,
        trace_log_files=[trace_log_file],
    )


@pytest.fixture
def incomplete_trace_log_file() -> Path:
    return (
        Path(__file__).parent.parent.parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "TRACE-incomplete.log"
    )


def test__parse_files(cv_source_filesystem):
    with pytest.warns() as record:
        plate = ZAdjustedStackAcquisition(
            source=cv_source_filesystem,
            alignment=TileAlignmentOptions.GRID,
        )
    assert len(record) == 2
    assert str(record[0].message) == "Z position information missing for some files."
    assert str(record[1].message).startswith(
        "First file without z position information"
    )

    with pytest.warns() as record2:
        files = plate._parse_files()
    assert len(record2) == 2
    assert str(record2[0].message) == "Z position information missing for some files."
    assert str(record2[1].message).startswith(
        "First file without z position information"
    )

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


def test_get_well_acquisitions(cv_source_filesystem):
    with pytest.warns() as record:
        plate = ZAdjustedStackAcquisition(
            source=cv_source_filesystem,
            alignment=TileAlignmentOptions.GRID,
        )
    assert len(record) == 2
    assert str(record[0].message) == "Z position information missing for some files."
    assert str(record[1].message).startswith(
        "First file without z position information"
    )

    wells = plate.get_well_acquisitions()
    assert len(wells) == 3
    for well in wells:
        for tile in well.get_tiles():
            file_name = (
                f"CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_"
                f"{well.name}_T"
                f"{str(tile.position.time + 1).zfill(4)}F.*L.*A.*Z.*C"
                f"{str(tile.position.channel + 1).zfill(2)}\\.tif"
            )
            re_file_name = re.compile(file_name)
            from faim_ipa.hcs.cellvoyager.tile import StackedTile

            assert isinstance(tile, StackedTile)
            assert re_file_name.match(tile.tiles[0].path)
            assert tile.shape[1:] == plate._source.get_image(tile.tiles[0].path).shape
            assert tile.tiles[0].illumination_correction_matrix_path is None
            assert tile.tiles[0].background_correction_matrix_path is None
            assert tile.position.x in [0, 2000]
            assert tile.position.y in [0, 2000]
            assert tile.position.z in [0, 1, 2, 3, 4, 5]


def test_create_z_mapping(cv_source_filesystem):
    with pytest.warns():
        zasa = ZAdjustedStackAcquisition(
            source=cv_source_filesystem,
            alignment=TileAlignmentOptions.GRID,
        )
        mapping = zasa.create_z_mapping()
    assert len(mapping) == 96
    assert set(mapping["z_pos"]) == {0, 1, 3, 4, 6, 7, 9, 10, 12, 15}


def test_incomplete_tracelog(cv_acquisition, incomplete_trace_log_file):
    with pytest.raises(ValueError, match="At least one invalid z position"):
        ZAdjustedStackAcquisition(
            source=CVSourceFS(
                cv_acquisition, trace_log_files=[incomplete_trace_log_file]
            ),
            alignment=TileAlignmentOptions.GRID,
        )
