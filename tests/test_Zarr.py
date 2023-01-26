import os
import shutil
import tempfile
import unittest
from os.path import exists, join
from pathlib import Path

from faim_hcs.io.MolecularDevicesImageXpress import parse_single_plane_multi_fields
from faim_hcs.Zarr import build_zarr_scaffold

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


if __name__ == "__main__":
    unittest.main()
