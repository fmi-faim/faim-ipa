from __future__ import annotations

from logging import Logger
from os import mkdir
from os.path import exists, join
from typing import TYPE_CHECKING

import pandas as pd

from ..records.DefaultRecord import DefaultRecord

if TYPE_CHECKING:
    from ..records.WellRecord import WellRecord


class OrganoidRecord(DefaultRecord):
    def __init__(self, well: WellRecord, organoid_id: str, save_dir: str = "."):
        super().__init__(organoid_id)
        self.logger = Logger(f"Organoid {organoid_id}")
        self.organoid_id = self.record_id
        self.well = well

        self.organoid_dir = join(save_dir, self.organoid_id)
        if not exists(self.organoid_dir):
            mkdir(self.organoid_dir)

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

    def build_overview(self):
        summary = {"organoid_id": [self.record_id]}
        for k in self.raw_files.keys():
            summary[k] = self.raw_files[k]

        for k in self.segmentations.keys():
            summary[k] = self.segmentations[k]

        for k in self.measurements.keys():
            summary[k] = self.measurements[k]

        return pd.DataFrame(summary)

    def save(self, path: str = None, name="organoid_summary"):
        if path is None:
            return super().save(self.organoid_dir, name=name)
        else:
            return super().save(path, name=name)
