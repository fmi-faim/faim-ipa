import os
import shutil
import tempfile
from os.path import join
from unittest import TestCase

from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.OrganoidRecord import OrganoidRecord
from faim_hcs.records.PlateRecord import PlateRecord
from faim_hcs.records.WellRecord import WellRecord


class ExperimentTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.root_dir = join(self.tmp_dir, "root_dir")
        self.exp_dir = join(self.tmp_dir, "exp_dir")
        os.mkdir(self.root_dir)
        os.mkdir(self.exp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_create(self):
        exp = Experiment("Experiment", root_dir=self.root_dir, save_dir=self.exp_dir)
        plate = PlateRecord(
            experiment=exp,
            plate_id="plate",
            save_dir=join(exp.get_experiment_dir()),
        )

        well = WellRecord(plate=plate, well_id="well", save_dir=plate.plate_dir)

        well1 = WellRecord(plate=plate, well_id="well1", save_dir=plate.plate_dir)

        org = OrganoidRecord(well=well, organoid_id="org_0", save_dir=well.well_dir)

        org.raw_files["raw"] = "/test/raw.tif"
        org.raw_files["raw1"] = "/test/raw1.tif"
        org.segmentations["seg"] = "/test/seg.tif"
        org.measurements["meas"] = "/test/meas.csv"
        org.spacings["raw"] = [2, 0.161, 0.161]
        org.spacings["raw1"] = [1, 0.1, 0.1]

        org1 = OrganoidRecord(well=well, organoid_id="org_1", save_dir=well.well_dir)

        org1.raw_files["raw"] = "/test1/raw.tif"
        org1.raw_files["raw1"] = "/test1/raw1.tif"
        org1.segmentations["seg"] = "/test1/seg.tif"
        org1.measurements["meas"] = "/test1/meas.csv"
        org1.spacings["raw"] = [2, 0.161, 0.161]
        org1.spacings["raw1"] = [1, 0.1, 0.1]

        org2 = OrganoidRecord(well=well1, organoid_id="org_2", save_dir=well1.well_dir)

        org2.raw_files["raw"] = "/test2/raw.tif"
        org2.raw_files["raw1"] = "/test2/raw1.tif"
        org2.segmentations["seg"] = "/test2/seg.tif"
        org2.measurements["meas"] = "/test2/meas.csv"
        org2.spacings["raw"] = [2, 0.161, 0.161]
        org2.spacings["raw1"] = [1, 0.1, 0.1]

        overview = exp.build_overview()

        assert len(overview) == 3
        assert overview["organoid_id"][0] == "org_0"
        assert overview["organoid_id"][1] == "org_1"
        assert overview["organoid_id"][2] == "org_2"
        assert overview["well"][0] == "well"
        assert overview["well"][1] == "well"
        assert overview["well"][2] == "well1"

        exp.reset_iterator()
        exp.only_iterate_over_plates(False)
        exp.only_iterate_over_wells(False)
        assert next(exp).organoid_id == "org_0"
        assert next(exp).organoid_id == "org_1"
        assert next(exp).organoid_id == "org_2"

        exp.reset_iterator()
        exp.only_iterate_over_plates(False)
        exp.only_iterate_over_wells(True)
        assert next(exp).well_id == "well"
        assert next(exp).well_id == "well1"

        exp.reset_iterator()
        exp.only_iterate_over_plates(True)
        exp.only_iterate_over_wells(True)
        assert next(exp).plate_id == "plate"

        exp.save()

        exp_loaded = Experiment()
        exp_loaded.load(join(self.exp_dir, "Experiment", "summary.csv"))

        organoids = exp_loaded.get_organoid_raw_files("raw")
        assert len(organoids) == 3
        assert all(
            organoids.columns
            == [
                "hcs_experiment",
                "root_dir",
                "save_dir",
                "plate",
                "well",
                "organoid_id",
                "raw",
            ]
        )

        organoids = exp_loaded.get_organoid_segmentation_files("seg")
        assert len(organoids) == 3
        assert all(
            organoids.columns
            == [
                "hcs_experiment",
                "root_dir",
                "save_dir",
                "plate",
                "well",
                "organoid_id",
                "seg",
            ]
        )

        organoids = exp_loaded.get_organoid_raw_segmentation_files("raw", "seg")
        assert len(organoids) == 3
        assert all(
            organoids.columns
            == [
                "hcs_experiment",
                "root_dir",
                "save_dir",
                "plate",
                "well",
                "organoid_id",
                "raw",
                "seg",
            ]
        )
