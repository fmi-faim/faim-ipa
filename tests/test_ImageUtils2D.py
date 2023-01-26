import shutil
import tempfile
import unittest
from os.path import join
from pathlib import Path

import numpy as np

from faim_hcs.ImageUtils2D import get_well_image_CYX
from faim_hcs.io.MolecularDevicesImageXpress import parse_single_plane_multi_fields

ROOT_DIR = Path(__file__).parent


class TestZarr(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.files = parse_single_plane_multi_fields(
            join(ROOT_DIR.parent, "resources", "MIP-2P-2sub")
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_get_well_image_CYX(self):
        cyx, hists, ch_meta, metadata = get_well_image_CYX(
            well_files=self.files[self.files["well"] == "C05"]
        )

        assert cyx.shape == (2, 2048, 4096)
        assert cyx.dtype == np.uint16
        assert len(hists) == 2
        assert hists[0].min() == 107
        assert hists[1].min() == 126
        assert ch_meta[0] == {
            "Exposure Time": "20 ms",
            "Lumencor Cyan Intensity": 0.0,
            "Lumencor Green Intensity": 0.0,
            "Lumencor Red Intensity": 0.0,
            "Lumencor Violet Intensity": 100.0,
            "Lumencor Yellow Intensity": 0.0,
            "ShadingCorrection": "Off",
            "_IllumSetting_": "DAPI",
            "_MagNA_": 0.75,
            "_MagSetting_": "20X Plan Apo Lambda",
            "wavelength": 447.0,
        }
        assert ch_meta[0] == {
            "Exposure Time": "20 ms",
            "Lumencor Cyan Intensity": 0.0,
            "Lumencor Green Intensity": 0.0,
            "Lumencor Red Intensity": 0.0,
            "Lumencor Violet Intensity": 100.0,
            "Lumencor Yellow Intensity": 0.0,
            "ShadingCorrection": "Off",
            "_IllumSetting_": "DAPI",
            "_MagNA_": 0.75,
            "_MagSetting_": "20X Plan Apo Lambda",
            "wavelength": 447.0,
        }
        assert metadata == {
            "PixelType": "uint16",
            "spatial-calibration-units": "um",
            "spatial-calibration-x": 0.3417,
            "spatial-calibration-y": 0.3417,
        }


if __name__ == "__main__":
    unittest.main()
