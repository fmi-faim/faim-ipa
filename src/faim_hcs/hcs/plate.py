from enum import IntEnum
from typing import Union


class PlateLayout(IntEnum):
    """Plate layout, 18, 24, 96 or 384-well."""

    I18 = 18
    I24 = 24
    I96 = 96
    I384 = 384


def get_rows_and_columns(
    layout: Union[PlateLayout, int]
) -> tuple[list[str], list[str]]:
    """Return rows and columns for requested layout."""
    if layout == PlateLayout.I18:
        rows = ["A", "B", "C"]
        cols = [str(i).zfill(2) for i in range(1, 7)]
        assert len(rows) * len(cols) == 18
    elif layout == PlateLayout.I24:
        rows = ["A", "B", "C", "D"]
        cols = [str(i).zfill(2) for i in range(1, 7)]
        assert len(rows) * len(cols) == 24
    elif layout == PlateLayout.I96:
        rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
        cols = [str(i).zfill(2) for i in range(1, 13)]
        assert len(rows) * len(cols) == 96
    elif layout == PlateLayout.I384:
        rows = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
        ]
        cols = [str(i).zfill(2) for i in range(1, 25)]
        assert len(rows) * len(cols) == 384
    else:
        raise NotImplementedError(f"{layout} layout not supported.")

    return rows, cols
