# SPDX-FileCopyrightText: 2023 Friedrich Miescher Institute for Biomedical Research (FMI), Basel (Switzerland)
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest

from faim_hcs.io.MolecularDevicesImageXpress import parse_files
from faim_hcs.MetaSeriesUtils import (
    get_well_image_CYX,
    get_well_image_ZCYX,
    montage_grid_image_YX,
    montage_stage_pos_image_YX,
)

ROOT_DIR = Path(__file__).parent


@pytest.fixture
def files():
    return parse_files(ROOT_DIR / "resources" / "Projection-Mix")


def test_get_well_image_CYX(files):
    files2d = files[(files["z"].isnull()) & (files["channel"].isin(["w1", "w2"]))]
    for well in files2d["well"].unique():
        img, hists, ch_metadata, metadata = get_well_image_CYX(
            files2d[files2d["well"] == well], assemble_fn=montage_stage_pos_image_YX
        )
        assert img.shape == (2, 512, 1024)
        assert len(hists) == 2
        assert "z-scaling" not in metadata
        for ch_meta in ch_metadata:
            assert "Z Projection Method" in ch_meta


def test_get_well_image_ZCYX(files):
    files3d = files[(~files["z"].isnull()) & (files["channel"].isin(["w1", "w2"]))]
    for well in files3d["well"].unique():
        img, hists, ch_metadata, metadata = get_well_image_ZCYX(
            files3d[files3d["well"] == well], assemble_fn=montage_grid_image_YX
        )
        assert img.shape == (10, 2, 512, 1024)
        assert len(hists) == 2
        assert "z-scaling" in metadata
