from pathlib import Path

import pytest

from faim_hcs.io.YokogawaCellVoyager import parse_files


@pytest.fixture
def acquisition_dir():
    return (
        Path(__file__).parent.parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack"
    )


def test_parse_files(acquisition_dir):
    files = parse_files(acquisition_dir=acquisition_dir)
    assert len(files) == 96
    assert len(files.columns) == 13
