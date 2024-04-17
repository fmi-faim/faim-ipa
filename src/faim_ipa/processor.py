import logging
import os
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from os.path import join

import zarr
from ome_zarr.io import parse_url


class AbstractProcessor(ABC):
    def __init__(self, name: str):
        self._name = name
        self.logger = self._create_logger()
        self.logger.info(f"{self._name} initialized.")
        self._log_environment()

    def _create_logger(self) -> logging.Logger:
        logger = logging.Logger(self._name.capitalize())
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        handler = logging.FileHandler(f"{now}-{self._name}.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _log_environment(self):
        self.logger.info("Logging conda dependencies...")
        conda_result = subprocess.run(["conda", "list"], capture_output=True, text=True)

        self.logger.info("Logging pip dependencies...")
        pip_result = subprocess.run(["pip", "list"], capture_output=True, text=True)

        env_dir = "environment"
        os.makedirs(env_dir, exist_ok=True)

        with open(join(env_dir, "conda-dependencies.txt"), "w") as f:
            f.write(conda_result.stdout)

        with open(join(env_dir, "pip-dependencies.txt"), "w") as f:
            f.write(pip_result.stdout)

    @abstractmethod
    def run(self):
        """
        This method does the heavy lifting. It must be implemented by
        subclasses.
        """
        raise NotImplementedError()


class AbstractHCSPlateProcessor(AbstractProcessor):
    def __init__(self, name: str, plate_reference: str):
        super().__init__(
            name=name,
        )
        self._plate = zarr.group(
            store=(parse_url(path=plate_reference, mode="w").store)
        )

    def run(self):
        # TODO: Parallel processing of all wells.
        pass

    def process_well(self, well: zarr.Group):
        raise NotImplementedError()
