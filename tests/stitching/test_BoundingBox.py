from faim_hcs.stitching import BoundingBox5D


def test_get_corner_points():
    bbox = BoundingBox5D(
        time_start=0,
        time_end=1,
        channel_start=0,
        channel_end=1,
        z_start=0,
        z_end=1,
        y_start=10,
        y_end=21,
        x_start=20,
        x_end=41,
    )
    corners = bbox.get_corner_points()
    expected = {
        (0, 0, 0, 10, 20),
        (0, 0, 0, 10, 40),
        (0, 0, 0, 20, 40),
        (0, 0, 0, 20, 20),
    }
    assert corners == expected


def test_contains_point():
    bbox = BoundingBox5D(
        time_start=0,
        time_end=1,
        channel_start=0,
        channel_end=1,
        z_start=0,
        z_end=2,
        y_start=0,
        y_end=2,
        x_start=0,
        x_end=2,
    )
    assert bbox.contains((0, 0, 1, 1, 1))
    assert bbox.contains((0, 0, 0, 1, 1))
    assert bbox.contains((0, 0, 0, 0, 1))
    assert bbox.contains((0, 0, 0, 0, 0))
    assert not bbox.contains((0, 0, 0, 2, 1))
    assert not bbox.contains((0, 0, 2, 0, 1))
    assert not bbox.contains((0, 0, 1, 0, 2))
    assert not bbox.contains((1, 0, 1, 1, 1))


def test_overlap_exact():
    bbox_a = BoundingBox5D(
        time_start=0,
        time_end=1,
        channel_start=0,
        channel_end=1,
        z_start=0,
        z_end=1,
        y_start=0,
        y_end=10,
        x_start=0,
        x_end=10,
    )
    bbox_b = BoundingBox5D(
        time_start=0,
        time_end=1,
        channel_start=0,
        channel_end=1,
        z_start=0,
        z_end=1,
        y_start=0,
        y_end=10,
        x_start=0,
        x_end=10,
    )
    assert bbox_a.overlaps(bbox_b)
    assert bbox_b.overlaps(bbox_a)


def test_overlap_complete():
    bbox_a = BoundingBox5D(
        time_start=0,
        time_end=3,
        channel_start=0,
        channel_end=3,
        z_start=0,
        z_end=3,
        y_start=0,
        y_end=10,
        x_start=0,
        x_end=10,
    )
    bbox_b = BoundingBox5D(
        time_start=1,
        time_end=2,
        channel_start=1,
        channel_end=2,
        z_start=1,
        z_end=2,
        y_start=2,
        y_end=8,
        x_start=2,
        x_end=8,
    )

    assert bbox_a.overlaps(bbox_b)
    assert bbox_b.overlaps(bbox_a)


def test_overlap_partial():
    bbox_a = BoundingBox5D(
        time_start=0,
        time_end=1,
        channel_start=0,
        channel_end=1,
        z_start=0,
        z_end=1,
        y_start=0,
        y_end=10,
        x_start=0,
        x_end=10,
    )
    bbox_b = BoundingBox5D(
        time_start=0,
        time_end=1,
        channel_start=0,
        channel_end=1,
        z_start=0,
        z_end=1,
        y_start=9,
        y_end=18,
        x_start=9,
        x_end=18,
    )

    assert bbox_a.overlaps(bbox_b)
    assert bbox_b.overlaps(bbox_a)


def test_no_overlap():
    bbox_a = BoundingBox5D(
        time_start=0,
        time_end=1,
        channel_start=0,
        channel_end=1,
        z_start=0,
        z_end=1,
        y_start=0,
        y_end=10,
        x_start=0,
        x_end=10,
    )
    bbox_b = BoundingBox5D(
        time_start=0,
        time_end=1,
        channel_start=0,
        channel_end=1,
        z_start=0,
        z_end=1,
        y_start=10,
        y_end=18,
        x_start=0,
        x_end=18,
    )

    assert not bbox_a.overlaps(bbox_b)
    assert not bbox_b.overlaps(bbox_a)
