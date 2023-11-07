from pathlib import Path

import pytest

from faim_hcs.io.acquisition import Plate_Acquisition, Well_Acquisition
from faim_hcs.io.ImageXpress import ImageXpress_Plate_Acquisition


@pytest.fixture
def acquisition_dir():
    return Path(__file__).parent.parent.parent / "resources" / "Projection-Mix"


@pytest.fixture
def acquisition(acquisition_dir):
    return ImageXpress_Plate_Acquisition(acquisition_dir)


def test_default(acquisition: Plate_Acquisition):
    wells = acquisition.wells()

    assert wells is not None
    assert len(wells) == 2
    assert len(acquisition._files) == 96

    well_acquisitions = acquisition.well_acquisitions()

    for well_acquisition in well_acquisitions:
        assert isinstance(well_acquisition, Well_Acquisition)
        assert len(well_acquisition.files()) == 48
        files = well_acquisition.files()

        assert len(files[files["z"].isnull()]) == 2 * 3  # 2 fields, 3 channels (1,2,3)
        assert len(files[files["z"] == "1"]) == 2 * 3  # 2 fields, 3 channels (1,2,4)
        assert len(files[files["z"] == "10"]) == 2 * 2  # 2 fields, 2 channels (1,2)

        positions = well_acquisition.positions()

        assert positions is not None
        assert len(positions) == len(files)
