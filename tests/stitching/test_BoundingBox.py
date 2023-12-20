from faim_hcs.stitching import BoundingBox5D


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

    bbox_c = BoundingBox5D(
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
    assert not bbox_a.overlaps(bbox_c)
