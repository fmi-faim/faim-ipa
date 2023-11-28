from typing import Optional

from numpy._typing import ArrayLike
from pydantic import NonNegativeInt
from skimage.measure.fit import BaseModel
from tifffile import imread


class TilePosition(BaseModel):
    time: Optional[NonNegativeInt]
    channel: Optional[NonNegativeInt]
    z: int
    y: int
    x: int

    def __init__(
        self,
        time: Optional[NonNegativeInt],
        channel: Optional[NonNegativeInt],
        z: int,
        y: int,
        x: int,
    ):
        super().__init__()
        self.time = time
        self.channel = channel
        self.z = z
        self.y = y
        self.x = x

    def __repr__(self):
        return f"TilePosition(time={self.time}, channel={self.channel}, z={self.z}, y={self.y}, x={self.x})"


class Tile(BaseModel):
    """
    A tile with a path to the image data, shape and position.
    """

    path: str
    shape: tuple[int, int]
    position: TilePosition

    def __init__(self, path: str, shape: tuple[int, int], position: TilePosition):
        super().__init__()
        self.path = path
        self.shape = shape
        self.position = position

    def __repr__(self):
        return (
            f"Tile(path={self.path}, shape={self.shape}, " f"position={self.position})"
        )

    def __str__(self):
        return self.__repr__()

    def get_yx_position(self) -> tuple[int, int]:
        return self.position.y, self.position.x

    def get_zyx_position(self) -> tuple[int, int, int]:
        return self.position.z, self.position.y, self.position.x

    def get_position(self) -> tuple[int, int, int, int, int]:
        return (
            self.position.time,
            self.position.channel,
            self.position.z,
            self.position.y,
            self.position.x,
        )

    def load_data(self) -> ArrayLike:
        """
        Load the image data from the path.

        Returns
        -------
        Image data
        """
        return imread(self.path)
