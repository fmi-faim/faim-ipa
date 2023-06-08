# SPDX-FileCopyrightText: 2023 Friedrich Miescher Institute for Biomedical Research (FMI), Basel (Switzerland)
#
# SPDX-License-Identifier: BSD-3-Clause

import shutil
import tempfile
import unittest
from os.path import exists, join
from pathlib import Path

import anndata as ad

from faim_hcs.io.MolecularDevicesImageXpress import (
    parse_multi_field_stacks,
    parse_single_plane_multi_fields,
)
from faim_hcs.MetaSeriesUtils import get_well_image_CYX, get_well_image_CZYX
from faim_hcs.Zarr import (
    PlateLayout,
    build_zarr_scaffold,
    write_cyx_image_to_well,
    write_czyx_image_to_well,
    write_labels_to_group,
    write_roi_table,
)

ROOT_DIR = Path(__file__).parent


class TestZarr(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        self.files = parse_single_plane_multi_fields(
            join(ROOT_DIR.parent, "resources", "Projection-Mix")
        )

        self.files3d = parse_multi_field_stacks(
            join(ROOT_DIR.parent, "resources", "Projection-Mix")
        )

        self.zarr_root = Path(self.tmp_dir, "zarr-files")
        self.zarr_root.mkdir()

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
            img, hists, ch_metadata, metadata, roi_tables = get_well_image_CYX(
                well_files=well_files, channels=["w1", "w2", "w3", "w4"]
            )

            field = plate[well[0]][str(int(well[1:]))][0]
            write_cyx_image_to_well(img, hists, ch_metadata, metadata, field)

            # Write all ROI tables
            for roi_table in roi_tables:
                write_roi_table(roi_tables[roi_table], roi_table, field)

        e07 = plate["E"]["7"]["0"].attrs.asdict()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "C00_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "C01_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "C02_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "C03_empty_histogram.npz"
        ).exists()
        assert "histograms" in e07.keys()
        assert "acquisition_metadata" in e07.keys()
        assert e07["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ] == [1.0, 1.3668, 1.3668]

        e08 = plate["E"]["8"]["0"].attrs.asdict()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "8"
            / "0"
            / "C00_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "8"
            / "0"
            / "C01_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "8"
            / "0"
            / "C02_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "8"
            / "0"
            / "C03_empty_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "tables"
            / "well_ROI_table"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "tables"
            / "FOV_ROI_table"
        ).exists()
        assert "histograms" in e08.keys()
        assert "acquisition_metadata" in e08.keys()
        assert e08["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ] == [1.0, 1.3668, 1.3668]

        # Check ROI table content
        table = ad.read_zarr(
            self.zarr_root 
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "tables"
            / "well_ROI_table"
        )
        df_well = table.to_df()
        roi_columns = [
            "x_micrometer", 
            "y_micrometer", 
            "z_micrometer", 
            "len_x_micrometer", 
            "len_y_micrometer", 
            "len_z_micrometer"
        ]
        assert list(df_well.columns) == roi_columns
        assert len(df_well) == 1
        target_values = [0.0, 0.0, 0.0, 1399.6031494140625, 699.8015747070312, 1.0]
        assert df_well.loc["well_1"].values.flatten().tolist() == target_values

        table = ad.read_zarr(
            self.zarr_root 
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "tables"
            / "FOV_ROI_table"
        )
        df_fov = table.to_df()
        assert list(df_fov.columns) == roi_columns
        assert len(df_fov) == 2
        target_values = [0.0, 699.8015747070312, 0.0, 699.8015747070312, 699.8015747070312, 1.0]
        assert df_fov.loc["Site 2"].values.flatten().tolist() == target_values

    def test_write_czyx_image_to_well(self):
        plate = build_zarr_scaffold(
            root_dir=self.zarr_root,
            files=self.files3d,
            layout=96,
            order_name="test-order",
            barcode="test-barcode",
        )

        for well in self.files3d["well"].unique():
            well_files = self.files3d[self.files3d["well"] == well]
            img, hists, ch_metadata, metadata, roi_tables = get_well_image_CZYX(
                well_files=well_files, channels=["w1", "w2", "w3", "w4"]
            )

            field = plate[well[0]][str(int(well[1:]))][0]
            write_czyx_image_to_well(img, hists, ch_metadata, metadata, field)

            # Write all ROI tables
            for roi_table in roi_tables:
                write_roi_table(roi_tables[roi_table], roi_table, field)

        e07 = plate["E"]["7"]["0"].attrs.asdict()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "C00_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "C01_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "C02_empty_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "C03_FITC_05_histogram.npz"
        ).exists()
        assert "histograms" in e07.keys()
        assert "acquisition_metadata" in e07.keys()
        assert e07["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ] == [1.0, 5.02, 1.3668, 1.3668]

        e08 = plate["E"]["8"]["0"].attrs.asdict()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "8"
            / "0"
            / "C00_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "8"
            / "0"
            / "C01_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "8"
            / "0"
            / "C02_empty_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "8"
            / "0"
            / "C03_FITC_05_histogram.npz"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "tables"
            / "well_ROI_table"
        ).exists()
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "tables"
            / "FOV_ROI_table"
        ).exists()
        assert "histograms" in e08.keys()
        assert "acquisition_metadata" in e08.keys()
        assert e08["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ] == [1.0, 5.0, 1.3668, 1.3668]

        # Check ROI table content
        table = ad.read_zarr(
            self.zarr_root 
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "tables"
            / "well_ROI_table"
        )
        df_well = table.to_df()
        roi_columns = [
            "x_micrometer", 
            "y_micrometer", 
            "z_micrometer", 
            "len_x_micrometer", 
            "len_y_micrometer", 
            "len_z_micrometer"
        ]
        assert list(df_well.columns) == roi_columns
        assert len(df_well) == 1
        target_values = [
            0.0, 
            0.0, 
            0.0, 
            1399.6031494140625, 
            699.8015747070312, 
            45.290000915527344
        ]
        assert df_well.loc["well_1"].values.flatten().tolist() == target_values

        table = ad.read_zarr(
            self.zarr_root 
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "tables"
            / "FOV_ROI_table"
        )
        df_fov = table.to_df()
        assert list(df_fov.columns) == roi_columns
        assert len(df_fov) == 2
        target_values = [
            0.0, 
            699.8015747070312, 
            0.0, 
            699.8015747070312, 
            699.8015747070312, 
            45.290000915527344
        ]
        assert df_fov.loc["Site 2"].values.flatten().tolist() == target_values

    def test_write_labels(self):
        plate = build_zarr_scaffold(
            root_dir=self.zarr_root,
            files=self.files3d,
            layout=96,
            order_name="test-order",
            barcode="test-barcode",
        )
        well_files = self.files3d[self.files3d["well"] == "E07"]
        img, hists, ch_metadata, metadata, roi_tables = get_well_image_CZYX(
            well_files=well_files, channels=["w1", "w2", "w3", "w4"]
        )
        field = plate["E"]["7"][0]
        write_czyx_image_to_well(img, hists, ch_metadata, metadata, field)
        threshold = 100
        labels = img > threshold
        labels_name = "my_segmentation"
        write_labels_to_group(
            labels=labels, labels_name=labels_name, parent_group=field
        )
        original_multiscales = plate["E/7/0"].attrs.asdict()["multiscales"]
        labels_multiscales = plate["E/7/0/labels/my_segmentation/"].attrs.asdict()[
            "multiscales"
        ]
        assert (
            self.zarr_root
            / "Projection-Mix.zarr"
            / "E"
            / "7"
            / "0"
            / "labels"
            / "my_segmentation"
        ).exists()
        assert len(original_multiscales) == len(labels_multiscales)
        assert original_multiscales[0]["axes"] == labels_multiscales[0]["axes"]
        assert original_multiscales[0]["datasets"] == labels_multiscales[0]["datasets"]
        assert (
            plate["E/7/0/0"][0, :, :, :].shape
            == plate["E/7/0/labels/my_segmentation/0"][0, :, :, :].shape
        )


if __name__ == "__main__":
    unittest.main()
