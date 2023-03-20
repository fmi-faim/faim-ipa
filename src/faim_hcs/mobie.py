import string
from copy import copy
from os.path import join

import pandas as pd
import zarr
from mobie.metadata import (
    add_regions_to_dataset,
    add_source_to_dataset,
    add_view_to_dataset,
    get_default_view,
    get_image_display,
    get_merged_grid_source_transform,
)
from tqdm.auto import tqdm

from faim_hcs.UIntHistogram import UIntHistogram


def hex_to_rgba(h) -> str:
    """Convert a hex-color-code to RGBA values.

    :param h: A hex-color-code
    :return: RGBA color-code
    """
    if not isinstance(h, str):
        return "r=255,g=255,b=255,a=255"
    return f"r={int(h[0:2], 16)},g={int(h[2:4], 16)},b={int(h[4:6], 16)},a=255"


def to_position(well_name):
    r, c = well_name[0], well_name[1:]
    r = string.ascii_uppercase.index(r)
    c = int(c) - 1
    return [c, r]


def add_wells_to_project(
    plate: zarr.Group,
    dataset_folder: str,
    well_group: str = "0",
    view_name: str = "default",
    label_suffix: str = "",
):
    """Add well groups of a ome-zarr plate to project

    :param plate: ome-zarr plate
    :param dataset_folder: mobie dataset folder
    :param well_group: zarr-index to zarr array. Defaults into zeroth field
    :param view_name: mobie view name
    :param label_suffix: suffix to distinguish empty views of the same well.
    :return:
    """

    sources = {}
    plate_hists = {}
    plate_colors = {}

    # Add wells as individual sources
    wells = []
    for i, row in enumerate(tqdm(list(plate.group_keys()))):
        for j, col in enumerate(tqdm(list(plate[row].group_keys()), leave=False)):
            attrs = plate[row][col][well_group].attrs.asdict()
            wells.append(row + col.zfill(2))
            path = join(plate.store.path, row, col, well_group)

            hists = [
                UIntHistogram.load(join(path, h_path)) for h_path in attrs["histograms"]
            ]

            group_name = f"{row}{col.zfill(2)}"

            for k, ch in enumerate(attrs["omero"]["channels"]):
                key = f"{ch['wavelength_id']}_{ch['label']}"
                if ch["label"] == "empty":
                    key = key + label_suffix

                name = f"{group_name}_{key}"
                name = name.replace(" ", "_")
                view = get_default_view(
                    source_type="image",
                    source_name=name,
                    menu_name="Wells",
                    color=hex_to_rgba(ch["color"]),
                    contrastLimits=[hists[k].quantile(0.01), hists[k].quantile(0.99)],
                    opacity=1.0,
                    sources=[
                        name,
                    ],
                )

                add_source_to_dataset(
                    dataset_folder=dataset_folder,
                    source_type="image",
                    source_name=name,
                    image_metadata_path=path,
                    file_format="ome.zarr",
                    channel=k,
                    view=view,
                )

                if key not in sources.keys():
                    sources[key] = [name]
                else:
                    sources[key].append(name)

                if key not in plate_hists.keys():
                    plate_hists[key] = copy(hists[k])
                else:
                    plate_hists[key].combine(hists[k])

                if key not in plate_colors.keys():
                    plate_colors[key] = hex_to_rgba(ch["color"])

    _add_well_regions(
        dataset_folder=dataset_folder,
        wells=wells,
    )

    _add_channel_plate_overviews(
        dataset_folder=dataset_folder,
        plate_hists=plate_hists,
        plate_colors=plate_colors,
        sources=sources,
        view_name=view_name,
    )


def _add_channel_plate_overviews(
    dataset_folder, plate_hists, plate_colors, sources, view_name
):
    default = {
        "isExclusive": True,
        "sourceDisplays": [],
        "sourceTransforms": [],
        "uiSelectionGroup": "bookmarks",
    }
    for i, ch in enumerate(sources.keys()):
        name = ch.replace(" ", "_")
        default["sourceDisplays"].append(
            get_image_display(
                name=f"Plate_{name}",
                sources=[f"merged_view_plate_{name}"],
                color=plate_colors[ch],
                contrastLimits=[
                    plate_hists[ch].quantile(0.01),
                    plate_hists[ch].quantile(0.99),
                ],
                visible=i == 0,
            )
        )
        default["sourceTransforms"].append(
            get_merged_grid_source_transform(
                sources=[src for src in sources[ch]],
                merged_source_name=f"merged_view_plate_{name}",
                positions=[to_position(src[:3]) for src in sources[ch]],
            )
        )

    default["sourceDisplays"].append(
        {
            "regionDisplay": {
                "opacity": 0.5,
                "lut": "glasbey",
                "name": "Wells",
                "sources": _get_well_sources_per_channel(sources),
                "tableSource": "wells",
                "visible": True,
                "showAsBoundaries": False,
            }
        }
    )

    add_view_to_dataset(
        dataset_folder=dataset_folder,
        view_name=view_name,
        view=default,
    )


def _get_well_sources_per_channel(sources):
    wells_per_channel = {}
    for ch in sources.keys():
        for well in sources[ch]:
            if well[:3] not in wells_per_channel.keys():
                wells_per_channel[well[:3]] = [well]
            else:
                wells_per_channel[well[:3]].append(well)

    return wells_per_channel


def _add_well_regions(dataset_folder, wells):
    well_table = pd.DataFrame(
        {
            "region_id": wells,
            "treatment": [
                "Unknown",
            ]
            * len(wells),
        }
    )
    add_regions_to_dataset(dataset_folder, "wells", well_table)
