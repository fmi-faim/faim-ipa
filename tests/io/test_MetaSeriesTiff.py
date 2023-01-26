import unittest
from os.path import join
from pathlib import Path

import numpy as np

from faim_hcs.io.MetaSeriesTiff import load_metaseries_tiff

ROOT_DIR = Path(__file__).parent.parent


class TestMetaSeriesTiff(unittest.TestCase):
    def test_load_metaseries_tiff(self):
        path = join(
            ROOT_DIR.parent,
            "resources",
            "MIP-2P-2sub",
            "2022-07-05",
            "1075",
            "MIP-4P-4sub_C06_s1_w152C23B9A-EB4C-4DF6-8A7F-F4147A9E7DDE.tif",
        )

        img, metadata = load_metaseries_tiff(path=path)

        assert img.shape == (2048, 2048)
        assert img.dtype == np.uint16

        assert metadata["_IllumSetting_"] == "DAPI"
        assert metadata["spatial-calibration-x"] == 0.3417
        assert metadata["spatial-calibration-y"] == 0.3417
        assert metadata["spatial-calibration-units"] == "um"
        assert metadata["stage-position-x"] == 68918.7
        assert metadata["stage-position-y"] == 23890.1
        assert metadata["PixelType"] == "uint16"
        assert metadata["_MagNA_"] == 0.75
        assert metadata["_MagSetting_"] == "20X Plan Apo Lambda"
        assert metadata["Exposure Time"] == "20 ms"
        assert metadata["Lumencor Cyan Intensity"] == 0
        assert metadata["Lumencor Green Intensity"] == 0
        assert metadata["Lumencor Red Intensity"] == 0
        assert metadata["Lumencor Violet Intensity"] == 100.0
        assert metadata["Lumencor Yellow Intensity"] == 0
        assert metadata["ShadingCorrection"] == "Off"
        assert metadata["stage-label"] == "C06 : Site 1"
        assert metadata["SiteX"] == 1
        assert metadata["SiteY"] == 1
        assert metadata["wavelength"] == 447
