# SPDX-FileCopyrightText: 2023 Friedrich Miescher Institute for Biomedical Research (FMI), Basel (Switzerland)
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from os.path import join
from pathlib import Path

import numpy as np

from faim_ipa.io.MetaSeriesTiff import load_metaseries_tiff

ROOT_DIR = Path(__file__).parent.parent


class TestMetaSeriesTiff(unittest.TestCase):
    def test_load_metaseries_tiff(self):
        path = join(
            ROOT_DIR.parent,
            "resources",
            "Projection-Mix",
            "2023-02-21",
            "1334",
            "Projection-Mix_E07_s1_w1E94C24BD-45E4-450A-9919-257C714278F7.tif",
        )

        img, metadata = load_metaseries_tiff(path=path)

        assert img.shape == (512, 512)
        assert img.dtype == np.uint16

        assert metadata["_IllumSetting_"] == "FITC_05"
        assert metadata["spatial-calibration-x"] == 1.3668
        assert metadata["spatial-calibration-y"] == 1.3668
        assert metadata["spatial-calibration-units"] == "um"
        assert metadata["stage-position-x"] == 79813.4
        assert metadata["stage-position-y"] == 41385.4
        assert metadata["stage-position-z"] == 9318.24
        assert metadata["PixelType"] == "uint16"
        assert metadata["_MagNA_"] == 0.75
        assert metadata["_MagSetting_"] == "20X Plan Apo Lambda"
        assert metadata["Exposure Time"] == "15 ms"
        assert metadata["Lumencor Cyan Intensity"] == 5.09804
        assert metadata["Lumencor Green Intensity"] == 0
        assert metadata["Lumencor Red Intensity"] == 0
        assert metadata["Lumencor Violet Intensity"] == 0
        assert metadata["Lumencor Yellow Intensity"] == 0
        assert metadata["ShadingCorrection"] == "Off"
        assert metadata["stage-label"] == "E07 : Site 1"
        assert metadata["SiteX"] == 1
        assert metadata["SiteY"] == 1
        assert metadata["wavelength"] == 536
        assert metadata["Z Projection Method"] == "Maximum"
        assert metadata["Z Projection Step Size"] == 5
