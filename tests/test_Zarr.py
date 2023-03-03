# SPDX-FileCopyrightText: 2023 Friedrich Miescher Institute for Biomedical Research (FMI), Basel (Switzerland)
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
import tempfile
import unittest
from os.path import exists, join
from pathlib import Path

from faim_hcs.io.MolecularDevicesImageXpress import parse_single_plane_multi_fields
from faim_hcs.MetaSeriesUtils import get_well_image_CYX
from faim_hcs.Zarr import PlateLayout, build_zarr_scaffold, write_cyx_image_to_well

ROOT_DIR = Path(__file__).parent


class TestZarr(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.files = parse_single_plane_multi_fields(
            join(ROOT_DIR.parent, "resources", "Projection-Mix")
        )

        self.zarr_root = join(self.tmp_dir, "zarr-files")
        os.mkdir(self.zarr_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_plate_scaffold_96(self):
        plate = build_zarr_scaffold(
            root_dir=self.zarr_root,
            files=self.files,
            layout=PlateLayout.I96,
            order_name="test-order",
            barcode="test-barcode",
        )

        assert exists(join(self.zarr_root, "Projection-Mix.zarr", "E", "7", "0"))
        assert exists(join(self.zarr_root, "Projection-Mix.zarr", "E", "8", "0"))

        attrs = plate.attrs.asdict()
        assert attrs["order_name"] == "test-order"
        assert attrs["barcode"] == "test-barcode"
        assert len(attrs["plate"]["columns"]) * len(attrs["plate"]["rows"]) == 96

    def test_plate_scaffold_384(self):
        plate = build_zarr_scaffold(
            root_dir=self.zarr_root,
            files=self.files,
            layout=PlateLayout.I384,
            order_name="test-order",
            barcode="test-barcode",
        )

        assert exists(join(self.zarr_root, "Projection-Mix.zarr", "E", "7", "0"))
        assert exists(join(self.zarr_root, "Projection-Mix.zarr", "E", "8", "0"))

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
            layout=24,
            order_name="test-order",
            barcode="test-barcode",
        )

    def test_write_cyx_image_to_well(self):
        plate = build_zarr_scaffold(
            root_dir=self.zarr_root,
            files=self.files,
            layout=96,
            order_name="test-order",
            barcode="test-barcode",
        )

        for well in self.files["well"].unique():
            well_files = self.files[self.files["well"] == well]
            img, hists, ch_metadata, metadata = get_well_image_CYX(
                well_files=well_files, channels=["w1", "w2", "w3"]
            )

            well_group = plate[well[0]][str(int(well[1:]))][0]
            write_cyx_image_to_well(img, hists, ch_metadata, metadata, well_group)

        e07 = plate["E"]["7"]["0"].attrs.asdict()
        assert exists(
            join(
                self.zarr_root,
                "Projection-Mix.zarr",
                "E",
                "7",
                "0",
                "C00_FITC_05_histogram.npz",
            )
        )
        assert exists(
            join(
                self.zarr_root,
                "Projection-Mix.zarr",
                "E",
                "7",
                "0",
                "C01_FITC_05_histogram.npz",
            )
        )
        assert exists(
            join(
                self.zarr_root,
                "Projection-Mix.zarr",
                "E",
                "7",
                "0",
                "C02_FITC_05_histogram.npz",
            )
        )
        assert "histograms" in e07.keys()
        assert "acquisition_metadata" in e07.keys()
        assert e07["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ] == [1.0, 1.3668, 1.3668]

        e08 = plate["E"]["8"]["0"].attrs.asdict()
        assert exists(
            join(
                self.zarr_root,
                "Projection-Mix.zarr",
                "E",
                "8",
                "0",
                "C00_FITC_05_histogram.npz",
            )
        )
        assert exists(
            join(
                self.zarr_root,
                "Projection-Mix.zarr",
                "E",
                "8",
                "0",
                "C01_FITC_05_histogram.npz",
            )
        )
        assert exists(
            join(
                self.zarr_root,
                "Projection-Mix.zarr",
                "E",
                "8",
                "0",
                "C02_FITC_05_histogram.npz",
            )
        )
        assert "histograms" in e08.keys()
        assert "acquisition_metadata" in e08.keys()
        assert e08["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ] == [1.0, 1.3668, 1.3668]


if __name__ == "__main__":
    unittest.main()
