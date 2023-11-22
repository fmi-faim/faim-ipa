def pad_bbox(
    bbox: tuple[int, ...], padding: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Pad n-dimensional bounding box by padding.

    Parameters
    ----------
    bbox :
        Bounding box to pad.
    padding :
        Padding to apply.
    shape :
        Shape of the image.

    Returns
    -------
    padded bounding box
    """
    ndims = len(bbox) // 2
    padded_starts = []
    padded_ends = []
    for i, (start, stop) in enumerate(zip(bbox[:ndims], bbox[ndims:])):
        padded_starts.append(max(0, start - padding[i]))
        padded_ends.append(min(shape[i], stop + padding[i]))

    return tuple(padded_starts + padded_ends)
