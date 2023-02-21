import os
import shutil
import tempfile
import unittest
from os.path import exists, join
from pathlib import Path

import numpy as np

from faim_hcs.io.MolecularDevicesImageXpress import parse_single_plane_multi_fields
from faim_hcs.MetaSeriesUtils import get_well_image_CYX
from faim_hcs.Zarr import build_zarr_scaffold, write_cyx_image_to_well

ROOT_DIR = Path(__file__).parent


class TestZarr(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.files = parse_single_plane_multi_fields(
            join(ROOT_DIR.parent, "resources", "MIP-2P-2sub")
        )

        self.zarr_root = join(self.tmp_dir, "zarr-files")
        os.mkdir(self.zarr_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_plate_scaffold_96(self):
        plate = build_zarr_scaffold(
            root_dir=self.zarr_root,
            files=self.files,
            layout="96",
            order_name="test-order",
            barcode="test-barcode",
        )

        assert exists(join(self.zarr_root, "MIP-2P-2sub.zarr", "C", "5", "0"))
        assert exists(join(self.zarr_root, "MIP-2P-2sub.zarr", "C", "6", "0"))

        attrs = plate.attrs.asdict()
        assert attrs["order_name"] == "test-order"
        assert attrs["barcode"] == "test-barcode"
        assert len(attrs["plate"]["columns"]) * len(attrs["plate"]["rows"]) == 96

    def test_plate_scaffold_384(self):
        plate = build_zarr_scaffold(
            root_dir=self.zarr_root,
            files=self.files,
            layout="384",
            order_name="test-order",
            barcode="test-barcode",
        )

        assert exists(join(self.zarr_root, "MIP-2P-2sub.zarr", "C", "5", "0"))
        assert exists(join(self.zarr_root, "MIP-2P-2sub.zarr", "C", "6", "0"))

        attrs = plate.attrs.asdict()
        assert attrs["order_name"] == "test-order"
        assert attrs["barcode"] == "test-barcode"
        assert len(attrs["plate"]["columns"]) * len(attrs["plate"]["rows"]) == 384

    def test_plate_scaffold_24(self):
        self.assertRaises(
            NotImplementedError,
            build_zarr_scaffold,
            root_dir=self.zarr_root,
            files=self.files,
            layout="24",
            order_name="test-order",
            barcode="test-barcode",
        )

    def test_write_cyx_image_to_well(self):
        plate = build_zarr_scaffold(
            root_dir=self.zarr_root,
            files=self.files,
            layout="96",
            order_name="test-order",
            barcode="test-barcode",
        )

        for well in self.files["well"].unique():
            well_files = self.files[self.files["well"] == well]
            img, hists, ch_metadata, metadta = get_well_image_CYX(well_files=well_files)

            well_group = plate[well[0]][str(int(well[1:]))][0]
            write_cyx_image_to_well(img, hists, ch_metadata, metadta, well_group)

        c05 = plate["C"]["5"]["0"].attrs.asdict()
        assert exists(
            join(
                self.zarr_root, "MIP-2P-2sub.zarr", "C", "5", "0", "DAPI_histogram.npz"
            )
        )
        assert exists(
            join(
                self.zarr_root, "MIP-2P-2sub.zarr", "C", "5", "0", "FITC_histogram.npz"
            )
        )
        assert "histograms" in c05.keys()
        assert "acquisition_metadata" in c05.keys()
        assert c05["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ] == [1.0, 0.3417, 0.3417]

        c06 = plate["C"]["6"]["0"].attrs.asdict()
        assert exists(
            join(
                self.zarr_root, "MIP-2P-2sub.zarr", "C", "6", "0", "DAPI_histogram.npz"
            )
        )
        assert exists(
            join(
                self.zarr_root, "MIP-2P-2sub.zarr", "C", "6", "0", "FITC_histogram.npz"
            )
        )
        assert "histograms" in c06.keys()
        assert "acquisition_metadata" in c06.keys()
        assert c06["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ] == [1.0, 0.3417, 0.3417]

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
            "channel-name": "DAPI",
            "display-color": "0034ff",
            "exposure-time": 20.0,
            "exposure-time-unit": "ms",
            "objective": "20X Plan Apo Lambda",
            "objective-NA": 0.75,
            "power": 100.0,
            "shading-correction": False,
            "wavelength": "violet",
        }
        assert ch_meta[1] == {
            "channel-name": "FITC",
            "display-color": "73ff00",
            "exposure-time": 250.0,
            "exposure-time-unit": "ms",
            "objective": "20X Plan Apo Lambda",
            "objective-NA": 0.75,
            "power": 100.0,
            "shading-correction": False,
            "wavelength": "cyan",
        }
        assert metadata == {
            "pixel-type": "uint16",
            "spatial-calibration-units": "um",
            "spatial-calibration-x": 0.3417,
            "spatial-calibration-y": 0.3417,
        }


if __name__ == "__main__":
    unittest.main()
