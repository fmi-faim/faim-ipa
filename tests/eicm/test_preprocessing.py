from pathlib import Path
from unittest import TestCase

import numpy as np

from faim_ipa.eicm.preprocessing.multiple_tiles import (
    average_of_mips,
    average_of_thresholded_mips,
)
from faim_ipa.eicm.preprocessing.cellvoyager import get_metadata, parse_filename

RESOURCE_DIR = Path(__file__).parent.parent.parent / "resources" / "eicm"


class CellVoyager(TestCase):
    def test_get_metadata(self):
        acq_date, px_size, px_size_unit, channels = get_metadata(RESOURCE_DIR)

        assert acq_date == "2022-12-21"
        assert px_size == 0.10833333333333334
        assert px_size_unit == "micron"
        assert channels == {
            "1": {
                "pixel_size": "0.10833333333333334",
                "cam_index": "2",
                "objective": "60x-W",
                "filter": "BP445-45",
                "laser": "405nm",
            },
            "2": {
                "pixel_size": "0.10833333333333334",
                "cam_index": "2",
                "objective": "60x-W",
                "filter": "BP525-50",
                "laser": "488nm",
            },
            "3": {
                "pixel_size": "0.10833333333333334",
                "cam_index": "1",
                "objective": "60x-W",
                "filter": "BP600-37",
                "laser": "561nm",
            },
            "4": {
                "pixel_size": "0.10833333333333334",
                "cam_index": "1",
                "objective": "60x-W",
                "filter": "BP676-29",
                "laser": "640nm",
            },
            "5": {
                "pixel_size": "0.10833333333333334",
                "cam_index": "2",
                "objective": "60x-W",
                "filter": "BP525-50",
                "laser": "Lamp",
            },
            "6": {
                "pixel_size": "0.10833333333333334",
                "cam_index": "2",
                "objective": "60x-W",
                "filter": "BP525-50",
                "laser": "488nm",
            },
        }

    def test_parse_file_name(self):
        plate_name = "Test-plate"
        well = "A01"
        time_point = "T0001"
        field = "F010"
        l01 = "L01"
        action = "A01"
        z = "Z32"
        channel = "C01"
        file_name = (
            f"{plate_name}_{well}_{time_point}{field}{l01}{action}{z}{channel}.tif"
        )

        p, w, tp, f, l_, a, z_, ch = parse_filename(file_name, plate_name)

        assert p == plate_name
        assert w == well
        assert tp == time_point
        assert f == 10
        assert l_ == l01
        assert a == action
        assert z_ == 32
        assert ch == channel


class MultipleTiles(TestCase):
    def test_average_of_thresholded_mips_lower(self):
        images = [
            "gradient-left-right.tif",
            "gradient-top-bottom.tif",
        ]
        reference = average_of_thresholded_mips(
            [RESOURCE_DIR / file for file in images],
            lower_threshold=128,
        )
        assert reference.shape == (256, 256)
        assert np.isnan(reference[0, 0])
        assert np.isnan(reference[127, 127])
        assert reference[128, 128] == 128
        assert reference[255, 255] == 255

        assert reference[0, 255] == 255
        assert reference[255, 0] == 255

    def test_average_of_thresholded_mips(self):
        images = [
            "gradient-left-right.tif",
            "gradient-top-bottom.tif",
        ]
        reference = average_of_thresholded_mips(
            [RESOURCE_DIR / file for file in images],
            lower_threshold=0,
            upper_threshold=127,
        )
        assert reference.shape == (256, 256)
        assert reference[0, 0] == 0
        assert reference[127, 127] == 127
        assert np.isnan(reference[128, 128])
        assert np.isnan(reference[255, 255])

        assert reference[0, 255] == 0
        assert reference[255, 0] == 0

    def test_average_of_mips(self):
        images = [
            "gradient-left-right.tif",
            "gradient-top-bottom.tif",
        ]
        reference = average_of_mips(
            [RESOURCE_DIR / file for file in images],
        )
        assert reference.shape == (256, 256)
        assert reference[0, 0] == 0
        assert reference[127, 127] == 127
        assert reference[128, 128] == 128
        assert reference[255, 255] == 255

        assert reference[0, 255] == 127.5
        assert reference[255, 0] == 127.5
