from skimage.measure.fit import BaseModel


class BoundingBox5D(BaseModel):
    """
    A 5D bounding box with integer coordinates.
        * time
        * channel
        * z
        * y
        * x
    """

    time_start: int
    time_end: int
    channel_start: int
    channel_end: int
    z_start: int
    z_end: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int

    def __init__(
        self,
        time_start: int,
        time_end: int,
        channel_start: int,
        channel_end: int,
        z_start: int,
        z_end: int,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
    ):
        super().__init__()
        self.time_start = time_start
        self.time_end = time_end
        self.channel_start = channel_start
        self.channel_end = channel_end
        self.z_start = z_start
        self.z_end = z_end
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end

    @classmethod
    def from_pos_and_shape(
        cls,
        position: tuple[int, int, int, int, int],
        shape: tuple[int, int, int, int, int],
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
        BoundingBox5D
        """
        return BoundingBox5D(
            time_start=position[0],
            time_end=position[0] + shape[0],
            channel_start=position[1],
            channel_end=position[1] + shape[1],
            z_start=position[2],
            z_end=position[2] + shape[2],
            y_start=position[3],
            y_end=position[3] + shape[3],
            x_start=position[4],
            x_end=position[4] + shape[4],
        )

    def get_corner_points(self) -> set[tuple[int, int, int, int, int]]:
        """
        Get the corner points of this bounding box.

        Returns
        -------
        list of corner points.
        """
        points = set()
        for t in [self.time_start, self.time_end - 1]:
            for c in [self.channel_start, self.channel_end - 1]:
                for z in [self.z_start, self.z_end - 1]:
                    for y in [self.y_start, self.y_end - 1]:
                        for x in [self.x_start, self.x_end - 1]:
                            points.add((t, c, z, y, x))
        return points

    def contains(self, point: tuple[int, int, int, int, int]) -> bool:
        """
        Check if a point is inside this bounding box.

        Parameters
        ----------
        point :
            Point to check.
        """
        inside_t = point[0] >= self.time_start and point[0] < self.time_end
        inside_c = point[1] >= self.channel_start and point[1] < self.channel_end
        inside_z = point[2] >= self.z_start and point[2] < self.z_end
        inside_y = point[3] >= self.y_start and point[3] < self.y_end
        inside_x = point[4] >= self.x_start and point[4] < self.x_end
        return inside_t and inside_c and inside_z and inside_y and inside_x

    def overlaps(self, bbox: "BoundingBox5D") -> bool:
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
