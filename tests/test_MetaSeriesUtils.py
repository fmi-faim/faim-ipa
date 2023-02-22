from pathlib import Path

import pytest

from faim_hcs.io.MolecularDevicesImageXpress import parse_single_plane_multi_fields
from faim_hcs.MetaSeriesUtils import (
    get_well_image_CYX,
    get_well_image_ZCYX,
    montage_grid_image_YX,
    montage_stage_pos_image_YX,
)

ROOT_DIR = Path(__file__).parent


@pytest.fixture
def files2d():
    return parse_single_plane_multi_fields(
        ROOT_DIR.parent / "resources" / "MIP-2P-2sub"
    )


@pytest.fixture
def files3d():
    files = parse_single_plane_multi_fields(
        ROOT_DIR.parent / "resources" / "MIP-2P-2sub"
    )
    return files.assign(z=1)


def test_get_well_image_CYX(files2d):
    for well in files2d["well"].unique():
        img, hists, ch_metadata, metadata = get_well_image_CYX(
            files2d[files2d["well"] == well], assemble_fn=montage_stage_pos_image_YX
        )
        assert img.shape == (2, 2048, 4096)


def test_get_well_image_ZCYX(files3d):
    for well in files3d["well"].unique():
        img, hists, ch_metadata, metadata = get_well_image_ZCYX(
            files3d[files3d["well"] == well], assemble_fn=montage_grid_image_YX
        )
        assert img.shape == (1, 2, 2048, 4096)
