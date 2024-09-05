from abc import ABC
from pathlib import Path

import tifffile
from numpy._typing import NDArray
from tifffile import imread


class Source(ABC):

    def get_image(self, file_name: str) -> NDArray:
        raise NotImplementedError()

    def exists(self, file_name: str) -> bool:
        raise NotImplementedError()


class FileSource(Source):

    def __init__(self, directory: Path, memmap: bool = False):
        self.directory = directory
        self.memmap = memmap

    def get_image(self, file_name: str) -> NDArray:
        if self.memmap:
            return tifffile.memmap(self.directory / file_name, mode="r")
        else:
            return imread(self.directory / file_name)

    def exists(self, file_name: str) -> bool:
        return (self.directory / file_name).exists()
