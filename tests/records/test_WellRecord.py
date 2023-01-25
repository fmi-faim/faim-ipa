import os
import shutil
import tempfile
from os.path import exists, join
from unittest import TestCase

from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.OrganoidRecord import OrganoidRecord
from faim_hcs.records.PlateRecord import PlateRecord
from faim_hcs.records.WellRecord import WellRecord


class WellRecordTest(TestCase):
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

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_create(self):
        well = WellRecord(
            plate=self.plate, well_id="well", save_dir=self.plate.plate_dir
        )

        assert self.plate.wells["well"] == well
        assert well.well_id == "well"
        assert well.plate == self.plate

    def test_save_and_load(self):
        well = WellRecord(
            plate=self.plate, well_id="E03", save_dir=self.plate.plate_dir
        )

        org = OrganoidRecord(well=well, organoid_id="org_1", save_dir=well.well_dir)

        org.raw_files["raw"] = "/test/raw.tif"
        org.segmentations["seg"] = "/test/seg.tif"
        org.measurements["meas"] = "/test/meas.csv"
        org.spacings["raw"] = [2, 0.161, 0.161]

        df = well.save()
        assert exists(
            join(self.plate.plate_dir, "E03", "org_1", "organoid_summary.json")
        )
        assert exists(join(self.plate.plate_dir, "E03", "well_summary.json"))

        well_loaded = WellRecord(
            plate=self.plate, well_id="E03", save_dir=self.plate.plate_dir
        )
        well_loaded.load(df, "well_summary")
        assert well_loaded.well_id == "E03"
        assert well_loaded.organoids["org_1"].organoid_id == "org_1"
