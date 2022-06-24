import shutil
import tempfile
from os.path import exists, join
from unittest import TestCase

from pandas import DataFrame

from faim_hcs.records.DefaultRecord import DefaultRecord


class DefaultRecordTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_create(self):
        dr = DefaultRecord("record_id")

        assert dr.record_id == "record_id"
        assert len(dr.raw_files.items()) == 0
        assert len(dr.spacings.items()) == 0
        assert len(dr.segmentations.items()) == 0
        assert len(dr.measurements.items()) == 0

    def test_add_raw_data(self):
        dr = DefaultRecord("record_id")

        self.assertRaises(NotImplementedError, dr.add_raw_data, "name", "path", (2, 2))

    def test_get_raw_data(self):
        dr = DefaultRecord("record_id")

        # Put key into dict to get NotImplementedError
        dr.raw_files["name"] = None
        self.assertRaises(NotImplementedError, dr.get_raw_data, "name")

    def test_add_segmentation(self):
        dr = DefaultRecord("record_id")

        self.assertRaises(NotImplementedError, dr.add_segmentation, "name", "path")

    def test_get_segmentation(self):
        dr = DefaultRecord("record_id")

        # Put key into dict to get NotImplementedError
        dr.segmentations["name"] = None
        self.assertRaises(NotImplementedError, dr.get_segmentation, "name")

    def test_add_measurement(self):
        dr = DefaultRecord("record_id")

        self.assertRaises(NotImplementedError, dr.add_measurement, "name", "path")

    def test_get_measurement(self):
        dr = DefaultRecord("record_id")

        # Put key into dict to get NotImplementedError
        dr.measurements["name"] = None
        self.assertRaises(NotImplementedError, dr.get_measurement, "name")

    def test_save(self):
        dr = DefaultRecord(record_id="record_id")

        dr.save(self.tmp_dir, "DefaultRecord")

        assert exists(join(self.tmp_dir, "DefaultRecord.json"))

    def test_load(self):
        dr = DefaultRecord(record_id="record_id")
        dr.raw_files["raw"] = "/test/raw.tif"
        dr.segmentations["seg"] = "/test/seg.tif"
        dr.measurements["meas"] = "/test/meas.csv"
        dr.spacings["raw"] = [2, 0.161, 0.161]

        dr.save(self.tmp_dir, "DefaultRecord")

        df = DataFrame({"path": [join(self.tmp_dir, "DefaultRecord.json")]})

        dr.load(df, "path")

        assert dr.record_id == "record_id"
        assert dr.raw_files == {"raw": "/test/raw.tif"}
        assert dr.segmentations == {"seg": "/test/seg.tif"}
        assert dr.measurements == {"meas": "/test/meas.csv"}
        assert dr.spacings == {"raw": [2, 0.161, 0.161]}
