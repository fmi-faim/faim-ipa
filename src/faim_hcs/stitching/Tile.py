from typing import Optional

from numpy._typing import ArrayLike, NDArray
from pydantic import BaseModel, NonNegativeInt
from tifffile import imread


class TilePosition(BaseModel):
    time: Optional[NonNegativeInt]
    channel: Optional[NonNegativeInt]
    z: int
    y: int
    x: int

    def __repr__(self):
        return f"TilePosition(time={self.time}, channel={self.channel}, z={self.z}, y={self.y}, x={self.x})"


class Tile:
    """
    A tile with a path to the image data, shape and position.
    """

    path: str
    shape: tuple[int, int]
    position: TilePosition
    background_correction_matrix: Optional[NDArray] = None
    illumination_correction_matrix: Optional[NDArray] = None

    def __init__(
        self,
        path: str,
        shape: tuple[int, int],
        position: TilePosition,
        background_correction_matrix: Optional[NDArray] = None,
        illumination_correction_matrix: Optional[NDArray] = None,
    ):
        super().__init__()
        self.path = path
        self.shape = shape
        self.position = position
        if background_correction_matrix is not None:
            assert background_correction_matrix.ndim == 2, "Background must " "be 2D."
            self.background_correction_matrix = background_correction_matrix

        if illumination_correction_matrix is not None:
            assert illumination_correction_matrix.ndim == 2, (
                "Illumination " "correction matrix must be 2D."
            )
            self.illumination_correction_matrix = illumination_correction_matrix

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
        data = imread(self.path)
        dtype = data.dtype
        if self.background_correction_matrix is not None:
            data = data - self.background_correction_matrix

        if self.illumination_correction_matrix is not None:
            data = data / self.illumination_correction_matrix

        return data.astype(dtype=dtype)
