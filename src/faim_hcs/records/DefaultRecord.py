import json
from logging import Logger
from os.path import join

import pandas as pd
from skimage.io import imread


class DefaultRecord:
    def __init__(self, record_id: str):
        self.logger = Logger(f"Record {record_id}")
        self.record_id = record_id

        self.raw_files = {}
        self.spacings = {}
        self.segmentations = {}
        self.measurements = {}

    def add_raw_data(self, name: str, path: str, spacing: tuple):
        location = self._get_relative_location(path)
        self.raw_files[name] = location
        self.spacings[name] = spacing

    def get_raw_data(self, name: str):
        if name in self.raw_files.keys():
            location = self._get_source_file_location(self.raw_files[name])
            return imread(location)
        else:
            self.logger.warning(
                f"Raw file {name} not in {self.record_id}:"
                f"{list(self.raw_files.keys())}"
            )

    def add_segmentation(self, name: str, path: str):
        location = self._get_relative_location(path)
        self.segmentations[name] = location

    def get_segmentation(self, name: str):
        if name in self.segmentations.keys():
            location = self._get_source_file_location(self.segmentations[name])
            return imread(location)
        else:
            self.logger.warning(
                f"Segmentation {name} does not exist in "
                f"{self.record_id}:"
                f"{list(self.segmentations.keys())}."
            )

    def add_measurement(self, name: str, path: str):
        location = self._get_relative_location(path)
        # TODO: validate file structure --> Columns
        self.measurements[name] = location

    def get_measurement(self, name: str):
        if name in self.measurements.keys():
            location = self._get_source_file_location(self.measurements[name])
            return pd.read_csv(location)
        else:
            self.logger.warning(
                f"Measurment {name} does not exist in "
                f"{self.record_id}:"
                f"{list(self.measurements.keys())}."
            )

    def _get_relative_location(self, path: str):
        raise NotImplementedError

    def _get_source_file_location(self, path: str):
        raise NotImplementedError

    def build_overview(self):
        raise NotImplementedError

    def save(self, path, name):

        location = join(path, name + ".json")

        obj_dict = {
            "record_id": self.record_id,
            "raw_files": self.raw_files,
            "segmentations": self.segmentations,
            "measurements": self.measurements,
            "spacings": self.spacings,
        }
        with open(location, "w") as f:
            json.dump(obj_dict, f, indent=4)

        return location

    def load(self, df, column):
        if len(df.index) > 0:
            assert len(
                df[column].unique()
            ), "There should only be 1 record. " "Found {} records.".format(
                len(df[column].unique())
            )
            with open(df[column].iloc[0]) as f:
                obj_dict = json.load(f)

            assert obj_dict["record_id"] == self.record_id
            self.raw_files = obj_dict["raw_files"]
            self.segmentations = obj_dict["segmentations"]
            self.measurements = obj_dict["measurements"]
            self.spacings = obj_dict["spacings"]
