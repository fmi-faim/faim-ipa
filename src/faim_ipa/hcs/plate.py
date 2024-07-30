from enum import IntEnum


class PlateLayout(IntEnum):
    """Plate layout, 18, 24, 96 or 384-well."""

    I18 = 18
    I24 = 24
    I96 = 96
    I384 = 384


def get_rows_and_columns(layout: PlateLayout | int) -> tuple[list[str], list[str]]:
    """Return rows and columns for requested layout."""
    if layout == PlateLayout.I18:
        rows = list("ABC")
        cols = [str(i).zfill(2) for i in range(1, 7)]
    elif layout == PlateLayout.I24:
        rows = list("ABCD")
        cols = [str(i).zfill(2) for i in range(1, 7)]
    elif layout == PlateLayout.I96:
        rows = list("ABCDEFGH")
        cols = [str(i).zfill(2) for i in range(1, 13)]
    elif layout == PlateLayout.I384:
        rows = list("ABCDEFGHIJKLMNOP")
        cols = [str(i).zfill(2) for i in range(1, 25)]
    else:
        msg = f"{layout} layout not supported."
        raise NotImplementedError(msg)

    assert len(rows) * len(cols) == layout
    return rows, cols
