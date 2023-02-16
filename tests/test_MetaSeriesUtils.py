from pathlib import Path

import pytest

from faim_hcs.io.MolecularDevicesImageXpress import parse_single_plane_multi_fields
from faim_hcs.MetaSeriesUtils import _montage_grid_image_YX, get_well_image_CYX

ROOT_DIR = Path(__file__).parent


@pytest.fixture
def files():
    return parse_single_plane_multi_fields(
        ROOT_DIR.parent / "resources" / "MIP-2P-2sub"
    )


def test_get_well_image_CYX(files):
    for well in files["well"].unique():
        img, hists, ch_metadata, metadata = get_well_image_CYX(
            files[files["well"] == well], assemble_fn=_montage_grid_image_YX
        )
        assert img.shape == (2, 2048, 4096)
