import pytest

from faim_ipa.hcs.plate import PlateLayout, get_rows_and_columns


def test_plate_layout():
    assert PlateLayout.I18 == 18
    assert PlateLayout.I24 == 24
    assert PlateLayout.I96 == 96
    assert PlateLayout.I384 == 384


def test_get_rows_and_columns():
    rows, cols = get_rows_and_columns(PlateLayout.I18)
    assert rows == ["A", "B", "C"]
    assert cols == ["01", "02", "03", "04", "05", "06"]

    rows, cols = get_rows_and_columns(PlateLayout.I24)
    assert rows == ["A", "B", "C", "D"]
    assert cols == ["01", "02", "03", "04", "05", "06"]

    rows, cols = get_rows_and_columns(PlateLayout.I96)
    assert rows == ["A", "B", "C", "D", "E", "F", "G", "H"]
    assert cols == [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]

    rows, cols = get_rows_and_columns(PlateLayout.I384)
    assert rows == [
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
    assert cols == [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
    ]

    with pytest.raises(NotImplementedError):
        get_rows_and_columns("I42")
