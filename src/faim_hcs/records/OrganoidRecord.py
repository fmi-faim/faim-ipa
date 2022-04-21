from logging import Logger
from os.path import exists, join

import pandas as pd
from records import WellRecord
from records.DefaultRecord import DefaultRecord


class OrganoidRecord(DefaultRecord):
    def __init__(self, well: WellRecord, organoid_id: str):
        super().__init__(organoid_id)
        self.logger = Logger(f"Organoid {organoid_id}")
        self.organoid_id = self.record_id
        self.well = well

        if self.well is not None:
            self.well.register_organoid(self)

    def _get_relative_location(self, path):
        if path.startswith(self.well.plate.experiment.root_dir):
            assert exists(path), f"File {path} does not exist."
            return path.replace(self.well.plate.experiment.root_dir, "")
        else:
            path_ = join(self.well.plate.experiment.root_dir, path)
            assert exists(path_), f"File {path_} does not exist."
            return path

    def _get_source_file_location(self, path):
        return join(self.well.plate.experiment.root_dir, path)

    def get_organoid_dir(self):
        return join(self.well.get_well_dir(), self.record_id)

    def build_overview(self):
        summary = {"organoid_id": [self.record_id]}
        for k in self.raw_files.keys():
            summary[k] = self.raw_files[k]

        for k in self.segmentations.keys():
            summary[k] = self.segmentations[k]

        for k in self.measurements.keys():
            summary[k] = self.measurements[k]

        return pd.DataFrame(summary)
