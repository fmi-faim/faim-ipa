import os
import string
from copy import copy
from os.path import join

import numpy as np
import pandas as pd
import zarr
from mobie.metadata import (
    add_regions_to_dataset,
    add_source_to_dataset,
    add_view_to_dataset,
    get_image_display,
    get_merged_grid_source_transform,
    get_segmentation_display,
    read_dataset_metadata,
)
from skimage.measure import regionprops_table
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

                add_source_to_dataset(
                    dataset_folder=dataset_folder,
                    source_type="image",
                    source_name=name,
                    image_metadata_path=path,
                    file_format="ome.zarr",
                    channel=k,
                    view={},  # do not create default view for source
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


def add_labels_view(
    plate: zarr.Group,
    dataset_folder: str,
    well_group: str = "0",
    channel: int = 0,
    label_name: str = "default",
    view_name: str = "default_labels",
    extra_properties: tuple[str] = ("area",),
):
    """Add merged grid segmentation view for labels of all wells in zarr

    :param plate: Zarr group representing an HCS plate
    :param dataset_folder: Dataset folder of the MoBIE project
    :param well_group: Path to subgroup within each well, e.g. '0' or '0/projections'
    :param channel: Channel in the well image to be added as segmentation view
    :param label_name: Name of the label subgroup in the Zarr file
    :param view_name: View of the MoBIE dataset, will be updated in place
    :param extra_properties: Property names to be added to regionprops measurement table
    """
    # add sources for each label image
    sources = []
    for row in tqdm(list(plate.group_keys())):
        for col in tqdm(list(plate[row].group_keys()), leave=False):
            path = join(plate.store.path, row, col, well_group, "labels", label_name)
            group_name = f"{row}{col.zfill(2)}"
            name = f"{group_name}_{label_name}"
            name = name.replace(" ", "_")

            # measure regionprops
            label_img = plate[row][col][well_group]["labels"][label_name][0][channel]
            datasets = plate[row][col][well_group]["labels"][label_name].attrs.asdict()[
                "multiscales"
            ][0]["datasets"]
            spacing = datasets[0]["coordinateTransformations"][0]["scale"]
            props = regionprops_table(
                label_img[np.newaxis, :],
                properties=("label", "centroid") + extra_properties,
                spacing=spacing,
            )

            # write default.tsv to dataset_folder/tables/name
            # TODO reconcile once saving table data inside zarr is possible
            table_folder = join(dataset_folder, "tables", name)
            os.makedirs(table_folder, exist_ok=True)

            table_path = join(table_folder, "default.tsv")
            table = pd.DataFrame(props)
            # TODO remove this renaming once (and if) MoBIE fully supports other table formats in MoBIE projects
            table = table.rename(
                columns={
                    "label": "label_id",
                    "centroid-0": "anchor_z",
                    "centroid-1": "anchor_y",
                    "centroid-2": "anchor_x",
                }
            )
            table.to_csv(table_path, sep="\t", index=False)

            add_source_to_dataset(
                dataset_folder=dataset_folder,
                source_type="segmentation",
                source_name=name,
                image_metadata_path=path,
                file_format="ome.zarr",
                channel=channel,
                table_folder=table_folder,
                view={},  # do not create default view for source
            )
            sources.append(name)

    # get view 'view_name' from dataset
    dataset_metadata = read_dataset_metadata(dataset_folder=dataset_folder)
    view = dataset_metadata["views"][view_name]

    # get_merged_grid_source_transform for list of sources
    view["sourceTransforms"].append(
        get_merged_grid_source_transform(
            sources=sources,
            merged_source_name=f"merged_segmentation_view_{view_name}",
            positions=[to_position(src[:3]) for src in sources],
        )
    )

    # get_segmentation_display for grid source and append to sourceDisplays
    view["sourceDisplays"].append(
        get_segmentation_display(
            name=f"Segmentation_{view_name}",
            sources=[f"merged_segmentation_view_{view_name}"],
            visible=False,
        )
    )

    # update view in original dataset
    add_view_to_dataset(
        dataset_folder=dataset_folder,
        view_name=view_name,
        view=view,
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
