import os
import shutil
import tempfile
from os.path import exists, join
from unittest import TestCase

from pandas import DataFrame

from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.OrganoidRecord import OrganoidRecord
from faim_hcs.records.PlateRecord import PlateRecord
from faim_hcs.records.WellRecord import WellRecord


class OrganoidRecordTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        root_dir = join(self.tmp_dir, "root_dir")
        exp_dir = join(self.tmp_dir, "exp_dir")
        os.mkdir(root_dir)
        os.mkdir(exp_dir)
        self.exp = Experiment("Experiment", root_dir=root_dir, save_dir=exp_dir)
        self.plate = PlateRecord(
            experiment=self.exp,
            plate_id="plate",
            save_dir=join(self.exp.get_experiment_dir()),
        )
        self.well = WellRecord(
            plate=self.plate, well_id="well", save_dir=self.plate.plate_dir
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_create(self):
        org = OrganoidRecord(
            well=self.well, organoid_id="org_0", save_dir=self.well.well_dir
        )

        assert org.organoid_id == "org_0"
        assert org.well == self.well
        assert org.organoid_dir == join(self.well.well_dir, "org_0")
        assert self.well.organoids["org_0"] == org

        assert exists(join(self.well.well_dir, "org_0"))

    def test_build_overview(self):
        org = OrganoidRecord(
            well=self.well, organoid_id="org_1", save_dir=self.well.well_dir
        )

        org.raw_files["raw"] = "/test/raw.tif"
        org.raw_files["raw1"] = "/test/raw1.tif"
        org.segmentations["seg"] = "/test/seg.tif"
        org.measurements["meas"] = "/test/meas.csv"
        org.spacings["raw"] = [2, 0.161, 0.161]
        org.spacings["raw1"] = [1, 0.1, 0.1]

        overview = org.build_overview()

        assert len(overview) == 1
        assert overview["organoid_id"].iloc[0] == "org_1"
        assert overview["raw"].iloc[0] == "/test/raw.tif"
        assert overview["raw1"].iloc[0] == "/test/raw1.tif"
        assert overview["seg"].iloc[0] == "/test/seg.tif"
        assert overview["meas"].iloc[0] == "/test/meas.csv"
        assert len(overview.columns) == 5

    def test_save_and_load(self):
        org = OrganoidRecord(
            well=self.well, organoid_id="org_1", save_dir=self.well.well_dir
        )

        org.raw_files["raw"] = "/test/raw.tif"
        org.segmentations["seg"] = "/test/seg.tif"
        org.measurements["meas"] = "/test/meas.csv"
        org.spacings["raw"] = [2, 0.161, 0.161]

        org.save()
        assert exists(join(self.well.well_dir, "org_1", "organoid_summary.json"))

        org.save(path=self.tmp_dir, name="organoid")
        assert exists(join(self.tmp_dir, "organoid.json"))

        df = DataFrame({"path": [join(self.tmp_dir, "organoid.json")]})
        org_loaded = OrganoidRecord(
            well=self.well, organoid_id="org_1", save_dir=self.well.well_dir
        )
        org_loaded.load(df, "path")
        assert org_loaded.organoid_id == "org_1"
        assert org_loaded.raw_files["raw"] == "/test/raw.tif"
        assert org_loaded.segmentations["seg"] == "/test/seg.tif"
        assert org_loaded.measurements["meas"] == "/test/meas.csv"
        assert org_loaded.spacings["raw"] == [2, 0.161, 0.161]

        org_loaded.save(path=self.tmp_dir, name="organoid")
