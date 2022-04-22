from __future__ import annotations

from logging import Logger
from os import mkdir
from os.path import exists, join
from typing import TYPE_CHECKING

import pandas as pd

from ..records.DefaultRecord import DefaultRecord
from ..records.OrganoidRecord import OrganoidRecord

if TYPE_CHECKING:
    from ..records.PlateRecord import PlateRecord


class WellRecord(DefaultRecord):
    def __init__(self, plate: PlateRecord, well_id: str, save_dir: str = "."):
        super().__init__(well_id)
        self.logger = Logger(f"Well {well_id}")
        self.well_id = self.record_id
        self.plate = plate

        self.well_dir = join(save_dir, self.well_id)
        if not exists(self.well_dir):
            mkdir(self.well_dir)

        self.organoids = {}
        self.organoid_iter = iter(self.organoids.values())

        if self.plate is not None:
            self.plate.register_well(self)

    def register_organoid(self, organoid: OrganoidRecord):
        self.organoids[organoid.record_id] = organoid
        self.reset_iterator()

    def get_dataframe(self):
        organoid_id = [organoid.record_id for organoid in self.organoids.values()]
        if len(organoid_id) == 0:
            organoid_id = [None]
        return pd.DataFrame(
            {
                "well": self.record_id,
                "organoid_id": organoid_id,
            }
        )

    def build_overview(self):
        df = self.get_dataframe()
        organoid_summaries = []
        for organoid in self.organoids.values():
            organoid_summaries.append(organoid.build_overview())

        if len(organoid_summaries) > 0:
            return df.merge(
                pd.concat(organoid_summaries), on="organoid_id", how="outer"
            )
        else:
            return df

    def get_organoid_raw_files(self, name: str):
        df = self.get_dataframe()
        organoid_raw_files = []
        for organoid in self.organoids.values():
            if name in organoid.raw_files.keys():
                organoid_raw_files.append(
                    pd.DataFrame(
                        {
                            "organoid_id": [organoid.record_id],
                            f"{name}": organoid.raw_files[name],
                        }
                    )
                )

        if len(organoid_raw_files) > 0:
            return df.merge(
                pd.concat(organoid_raw_files), on="organoid_id", how="outer"
            )
        else:
            return df

    def get_organoid_segmentation_files(self, name: str):
        df = self.get_dataframe()
        organoid_seg_files = []
        for organoid in self.organoids.values():
            if name in organoid.segmentations.keys():
                organoid_seg_files.append(
                    pd.DataFrame(
                        {
                            "organoid_id": [organoid.record_id],
                            f"{name}": organoid.segmentations[name],
                        }
                    )
                )

        if len(organoid_seg_files) > 0:
            return df.merge(
                pd.concat(organoid_seg_files), on="organoid_id", how="outer"
            )
        else:
            return df

    def get_organoid_raw_segmentation_files(
        self, raw_name: str, segmentation_name: str
    ):
        raw_files = self.get_organoid_raw_files(raw_name)
        seg_files = self.get_organoid_segmentation_files(segmentation_name)

        return raw_files.merge(seg_files)

    def _get_relative_location(self, path: str):
        if path.startswith(self.plate.experiment.root_dir):
            assert exists(path), f"File {path} does not exist."
            return path.replace(self.plate.experiment.root_dir, "")
        else:
            path_ = join(self.plate.experiment.root_dir, path)
            assert exists(path_), f"File {path_} does not exist."
            return path

    def _get_source_file_location(self, path):
        return join(self.plate.experiment.root_dir, path)

    def save(self, path: str = None, name: str = "well_summary"):
        if path is None:
            well_summary_location = super().save(self.well_dir, name)
        else:
            well_summary_location = super().save(path, name)
        organoid_id = []
        organoid_summary = []
        for obj in self.organoids.values():
            organoid_id.append(obj.record_id)
            organoid_summary.append(obj.save(name="organoid_summary"))

        df = self.get_dataframe()
        df["well_summary"] = [
            well_summary_location,
        ] * max(1, len(df))
        return df.merge(
            pd.DataFrame(
                {"organoid_id": organoid_id, "organoid_summary": organoid_summary}
            ),
            on="organoid_id",
            how="left",
        )

    def load(self, df, column):
        super().load(df, column)
        self.well_id = self.record_id
        for organoid_id in df.organoid_id.unique():
            if str(organoid_id) != "nan":
                organoid_id = str(organoid_id)
                organoid = OrganoidRecord(self, organoid_id, self.well_dir)
                organoid.load(
                    df.query(f"organoid_id == '{organoid_id}'"), "organoid_summary"
                )
                self.organoids[organoid_id] = organoid

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.organoid_iter)

    def reset_iterator(self):
        self.organoid_iter = iter(self.organoids.values())
