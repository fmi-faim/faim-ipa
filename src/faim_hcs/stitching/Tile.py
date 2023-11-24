from numpy._typing import ArrayLike
from skimage.measure.fit import BaseModel
from tifffile import imread


class Tile(BaseModel):
    """
    A tile with a path to the image data, shape and position.
    """

    path: str
    shape: tuple[int, int]
    position: tuple[int, int, int]

    def __init__(
        self, path: str, shape: tuple[int, int], position: tuple[int, int, int]
    ):
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

    def load_data(self) -> ArrayLike:
        """
        Load the image data from the path.

        Returns
        -------
        Image data
        """
        return imread(self.path)
