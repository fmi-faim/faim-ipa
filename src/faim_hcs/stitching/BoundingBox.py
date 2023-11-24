from skimage.measure.fit import BaseModel


class BoundingBox(BaseModel):
    """
    A 3D bounding box with integer coordinates.
    """

    z_start: int
    z_end: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int

    def __init__(
        self,
        z_start: int,
        z_end: int,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
    ):
        super().__init__()
        self.z_start = z_start
        self.z_end = z_end
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end

    @classmethod
    def from_pos_and_shape(
        cls, position: tuple[int, int, int], shape: tuple[int, int, int]
    ):
        """
        Create a bounding box from a position and shape.

        Parameters
        ----------
        position :
            Location of the bounding box.
        shape :
            Size of the bounding box

        Returns
        -------
        BoundingBox
        """
        return BoundingBox(
            z_start=position[0],
            z_end=position[0] + shape[0],
            y_start=position[1],
            y_end=position[1] + shape[1],
            x_start=position[2],
            x_end=position[2] + shape[2],
        )

    def get_corner_points(self) -> list[tuple[int, int, int]]:
        """
        Get the corner points of this bounding box.

        Returns
        -------
        list of corner points.
        """
        return [
            (self.z_start, self.y_start, self.x_start),
            (self.z_start, self.y_start, self.x_end - 1),
            (self.z_start, self.y_end - 1, self.x_end - 1),
            (self.z_start, self.y_end - 1, self.x_start),
            (self.z_end - 1, self.y_start, self.x_start),
            (self.z_end - 1, self.y_start, self.x_end - 1),
            (self.z_end - 1, self.y_end - 1, self.x_end - 1),
            (self.z_end - 1, self.y_end - 1, self.x_start),
        ]

    def contains(self, point: tuple[int, int, int]) -> bool:
        """
        Check if a point is inside this bounding box.

        Parameters
        ----------
        point :
            Point to check.
        """
        inside_z = point[0] >= self.z_start and point[0] < self.z_end
        inside_y = point[1] >= self.y_start and point[1] < self.y_end
        inside_x = point[2] >= self.x_start and point[2] < self.x_end
        return inside_z and inside_y and inside_x

    def overlaps(self, bbox: "BoundingBox") -> bool:
        """
        Check if this bounding box overlaps with another bounding box.

        Parameters
        ----------
        bbox :
            Bounding box to check.
        """
        inside = False
        for point in self.get_corner_points():
            inside = inside or bbox.contains(point)

        for point in bbox.get_corner_points():
            inside = inside or self.contains(point)

        return inside
