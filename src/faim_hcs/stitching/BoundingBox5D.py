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

    def overlaps_in_time(self, bbox: "BoundingBox5D") -> bool:
        return (self.time_start < bbox.time_end) and (bbox.time_start < self.time_end)

    def overlaps_in_channel(self, bbox: "BoundingBox5D") -> bool:
        return (self.channel_start < bbox.channel_end) and (
            bbox.channel_start < self.channel_end
        )

    def overlaps_in_z(self, bbox: "BoundingBox5D") -> bool:
        return (self.z_start < bbox.z_end) and (bbox.z_start < self.z_end)

    def overlaps_in_y(self, bbox: "BoundingBox5D") -> bool:
        return (self.y_start < bbox.y_end) and (bbox.y_start < self.y_end)

    def overlaps_in_x(self, bbox: "BoundingBox5D") -> bool:
        return (self.x_start < bbox.x_end) and (bbox.x_start < self.x_end)

    def overlaps(self, bbox: "BoundingBox5D") -> bool:
        """
        Check if this bounding box overlaps with another bounding box.

        Parameters
        ----------
        bbox :
            Bounding box to check.
        """
        if not self.overlaps_in_time(bbox):
            return False

        if not self.overlaps_in_channel(bbox):
            return False

        if not self.overlaps_in_z(bbox):
            return False

        if not self.overlaps_in_y(bbox):
            return False

        if not self.overlaps_in_x(bbox):
            return False

        return True
