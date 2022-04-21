from logging import Logger
from os import mkdir
from os.path import basename, dirname, exists, join

import pandas as pd
from records import PlateRecord


class Experiment:
    def __init__(self, name: str = None, root_dir: str = None, save_path: str = None):
        self.logger = Logger(f"HCS Experiment {name}")
        self.name = name

        if root_dir is not None:
            assert exists(root_dir), f"{root_dir} does not exist."
        self.root_dir = root_dir
        if self.root_dir[-1] != "/":
            self.root_dir = self.root_dir + "/"

        if save_path is not None:
            assert exists(save_path), f"{save_path} does not exist."
        self.save_path = save_path

        self.plates = {}
        self.current_plate = None
        self.plate_iter = iter(self.plates.values())

        self.iter_plates_only = False
        self.iter_wells_only = False

    def register_plate(self, plate: PlateRecord):
        self.plates[plate.plate_id] = plate
        self.reset_iterator()

    def add_plate(self, path: str):
        plate_id = basename(path)
        PlateRecord(self, plate_id=plate_id)

    def get_dataframe(self):
        return pd.DataFrame(
            {
                "hcs_experiment": self.name,
                "root_dir": self.root_dir,
                "plate": [p.plate_id for p in self.plates.values()],
            }
        )

    def build_overview(self):
        df = self.get_dataframe()
        plate_overviews = []
        for plate in self.plates.values():
            plate_overviews.append(plate.build_overview())

        return df.merge(pd.concat(plate_overviews), on="plate", how="outer")

    def get_organoid_raw_files(self, name: str):
        df = self.get_dataframe()
        plate_raw_files = []
        for plate in self.plates.values():
            plate_raw_files.append(plate.get_organoid_raw_files(name))

        return df.merge(pd.concat(plate_raw_files), on="plate", how="outer")

    def get_organoid_segmentation_files(self, name: str):
        df = self.get_dataframe()
        plate_seg_files = []
        for plate in self.plates.values():
            plate_seg_files.append(plate.get_organoid_segmentation_files(name))

        return df.merge(pd.concat(plate_seg_files), on="plate", how="outer")

    def get_organoid_raw_segmentation_files(
        self, raw_name: str, segmentation_name: str
    ):
        df = self.get_dataframe()
        plate_raw_seg_files = []
        for plate in self.plates.values():
            plate_raw_seg_files.append(
                plate.get_organoid_raw_segmentation_files(raw_name, segmentation_name)
            )

        return df.merge(pd.concat(plate_raw_seg_files), on="plate", how="outer")

    def get_experiment_dir(self):
        return join(self.save_path, self.name)

    def save(self):
        path_ = self.get_experiment_dir()
        if not exists(path_):
            mkdir(path_)

        df = self.get_dataframe()

        saved_plates = []
        for plate in self.plates.values():
            saved_plates.append(plate.save(path_))

        df = df.merge(pd.concat(saved_plates), on="plate", how="outer")

        df.to_csv(join(path_, "summary.csv"))

    def load(self, path):
        self.save_path = dirname(path)
        df = pd.read_csv(path)
        self.name = df.iloc[0]["hcs_experiment"]
        self.root_dir = df.iloc[0]["root_dir"]
        self.plates = {}
        for plate_id in df.plate.unique():
            pr = PlateRecord(self, plate_id)
            pr.load(df.query(f"plate == '{plate_id}'"), None)
            self.plates[plate_id] = pr

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_plate is None:
            if self.iter_plates_only:
                return next(self.plate_iter)
            else:
                self.current_plate = next(self.plate_iter)
                self.current_plate.iter_wells_only = self.iter_wells_only

        try:
            return next(self.current_plate)
        except StopIteration:
            self.current_plate = None
            return next(self)

    def reset_iterator(self):
        self.current_plate = None
        self.plate_iter = iter(self.plates.values())
        for plate in self.plates.values():
            plate.reset_iterator()

    def only_iterate_over_plates(self, b: bool = False):
        self.iter_plates_only = b

    def only_iterate_over_wells(self, b: bool = False):
        self.iter_wells_only = b
