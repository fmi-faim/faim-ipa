from __future__ import annotations

from logging import Logger
from os import mkdir
from os.path import exists, join
from typing import TYPE_CHECKING

import pandas as pd

from ..records.WellRecord import WellRecord

if TYPE_CHECKING:
    from src.faim_hcs.hcs.Experiment import Experiment


class PlateRecord:
    def __init__(self, experiment: Experiment, plate_id: str, save_dir: str = "."):
        self.logger = Logger(f"Plate {plate_id}")
        self.experiment = experiment
        self.plate_id = plate_id

        self.plate_dir = join(save_dir, self.plate_id)
        if not exists(self.plate_dir):
            mkdir(self.plate_dir)

        self.iter_wells_only = False

        self.wells = {}
        self.current_well = None
        self.well_iter = iter(self.wells.values())

        if self.experiment is not None:
            self.experiment.register_plate(self)

    def register_well(self, well: WellRecord):
        self.wells[well.record_id] = well
        self.reset_iterator()

    def get_dataframe(self):
        return pd.DataFrame(
            {"plate": self.plate_id, "well": [w.record_id for w in self.wells.values()]}
        )

    def build_overview(self):
        df = self.get_dataframe()
        well_overviews = []
        for well in self.wells.values():
            well_overviews.append(well.build_overview())

        return df.merge(pd.concat(well_overviews), on="well", how="outer")

    def get_organoid_raw_files(self, name: str):
        df = self.get_dataframe()
        well_raw_files = []
        for well in self.wells.values():
            well_raw_files.append(well.get_organoid_raw_files(name))

        return df.merge(pd.concat(well_raw_files), on="well", how="outer")

    def get_organoid_segmentation_files(self, name: str):
        df = self.get_dataframe()
        well_seg_files = []
        for well in self.wells.values():
            well_seg_files.append(well.get_organoid_segmentation_files(name))

        return df.merge(pd.concat(well_seg_files), on="well", how="outer")

    def get_organoid_raw_segmentation_files(
        self, raw_name: str, segmentation_name: str
    ):
        df = self.get_dataframe()
        well_raw_seg_files = []
        for well in self.wells.values():
            well_raw_seg_files.append(
                well.get_organoid_raw_segmentation_files(raw_name, segmentation_name)
            )

        return df.merge(pd.concat(well_raw_seg_files), on="well", how="outer")

    def save(self):
        df = self.get_dataframe()

        wells = []
        for well in self.wells.values():
            wells.append(well.save(name="well_summary"))

        if len(wells) > 0:
            df = df.merge(pd.concat(wells), on="well", how="outer")
        return df

    def load(self, df, column):
        for well_id in df.well.unique():
            if str(well_id) != "nan":
                well_id = str(well_id)
                wr = WellRecord(self, well_id, self.plate_dir)
                wr.load(df.query(f"well == '{well_id}'"), "well_summary")
                self.wells[well_id] = wr

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_well is None:
            if self.iter_wells_only:
                return next(self.well_iter)
            else:
                self.current_well = next(self.well_iter)
        try:
            return next(self.current_well)
        except StopIteration:
            self.current_well = None
            return next(self)

    def reset_iterator(self):
        self.current_well = None
        self.well_iter = iter(self.wells.values())
        for well in self.wells.values():
            well.reset_iterator()

    def only_iterate_over_wells(self, b: bool = False):
        self.iter_wells_only = b
