import os
from os.path import join
from pathlib import Path
from typing import Union

import pandas as pd
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_plate_metadata, write_well_metadata
from zarr import Group


def _get_row_cols(layout: str) -> tuple[list[str], list[str]]:
    """Return rows and columns for requested layout."""
    if layout == "96":
        rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
        cols = [str(i) for i in range(1, 13)]
        assert len(rows) * len(cols) == 96
    elif layout == "384":
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
        cols = [str(i) for i in range(1, 25)]
        assert len(rows) * len(cols) == 384
    else:
        raise NotImplementedError(f"{layout} layout not supported.")

    return rows, cols


def _create_zarr_plate(
    root_dir: Path, layout: str, files: pd.DataFrame, order_name: str, barcode: str
) -> Group:
    """Create plate layout according to ome-zarr NGFF.

    Additionally the `order_name` and `barcode` is added to the plate.attrs.

    :param root_dir: where the zarr is stored
    :param layout: plate layout
    :param files: table of all image files
    :param order_name: plate order name
    :param barcode: plate barcode
    :return: zarr group
    """
    rows, cols = _get_row_cols(layout=layout)

    plate_path = join(root_dir, files["name"].unique()[0] + ".zarr")
    os.makedirs(plate_path, exist_ok=False)

    store = parse_url(plate_path, mode="w").store
    plate = zarr.group(store=store)

    write_plate_metadata(
        plate,
        columns=cols,
        rows=rows,
        wells=[f"{w[0]}/{str(int(w[1:]))}" for w in files["well"].unique()],
        name=files["name"].unique()[0],
        field_count=1,
    )

    attrs = plate.attrs.asdict()
    attrs["order_name"] = order_name
    attrs["barcode"] = barcode
    plate.attrs.put(attrs)

    return plate


def _add_wells_to_plate(plate: Group, files: pd.DataFrame) -> None:
    """Add wells to zarr-plate according to ome-zarr NGFF."""
    for well in files["well"].unique():
        row, col = well[0], str(int(well[1:]))

        if row not in plate:
            plate.create_group(row)

        if col not in plate[row]:
            plate[row].create_group(col).create_group("0")
            write_well_metadata(plate[row][col], [{"path": "0"}])


def build_zarr_scaffold(
    root_dir: Union[str, Path],
    files: pd.DataFrame,
    layout: str = "96",
    order_name: str = "order-name",
    barcode: str = "barcode",
) -> Group:
    """Build empty zarr scaffold of a ome-zarr NGFF conform HCS experiment.

    Additionally `order_name` and `barcode` are added to the plate.attrs.

    :param root_dir: where the zarr is stored
    :param files: table of image files
    :param layout: plate layout
    :param order_name: plate order name
    :param barcode: plate barcode
    :return: zarr plate group
    """
    names = files["name"].unique()
    assert len(names) == 1, "Files do belong to more than one plate."

    plate = _create_zarr_plate(
        root_dir=root_dir,
        layout=layout,
        files=files,
        order_name=order_name,
        barcode=barcode,
    )

    _add_wells_to_plate(plate=plate, files=files)

    return plate
