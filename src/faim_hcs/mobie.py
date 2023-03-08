from copy import copy
from os.path import join

import pandas as pd
import zarr
from mobie import create_view
from mobie.metadata import (
    add_regions_to_dataset,
    add_source_to_dataset,
    get_default_view,
)

from faim_hcs.UIntHistogram import UIntHistogram


def hex_to_rgba(h) -> str:
    """Convert a hex-color-code to RGBA values.

    :param h: A hex-color-code
    :return: RGBA color-code
    """
    if not isinstance(h, str):
        return "r=255,g=255,b=255,a=255"
    return f"r={int(h[0:2], 16)},g={int(h[2:4], 16)},b={int(h[4:6], 16)},a=255"


def get_largest_yx_field_shape(plate: zarr.Group) -> tuple[int, int]:
    """Compute the YX extend of the largest field in the plate.

    :param plate: OME-Zarr plate
    :return: (y, x) shape
    """
    max_y, max_x = 0, 0
    for row in plate.group_keys():
        for col in plate[row].group_keys():
            shape = plate[row][col][0][0].shape
            if max_y < shape[-2]:
                max_y = shape[-2]

            if max_x < shape[-1]:
                max_x = shape[-1]

    return max_y, max_x


def add_group_to_project(
    dataset_folder: str,
    group: zarr.Group,
    group_name: str,
    yx_well_coordinates: tuple[int, int],
    yx_shape: tuple[int, int],
    hists: list[UIntHistogram],
    plate_hists: dict,
    sources: dict,
    source_transforms: dict,
    display_settings: dict,
    gap: int = 50,
):
    """Add a ome-zarr HCS well group to a MoBIE project.

    :param dataset_folder: mobie-dataset folder
    :param group: zarr group
    :param group_name: name of the zarr group
    :param yx_well_coordinates: well position
    :param yx_shape: max shape of the largest field over all wells
    :param hists: list of channel histograms
    :param plate_hists: dict mapping channels to accumulated histograms for
    plate
    :param sources: dict mapping channels to all well-sources
    :param source_transforms: dict mapping channels to source transformations
    :param display_settings: dict mapping channels to display settings
    :param gap: between wells in the viewer
    :return:
    """
    y_wc, x_wc = yx_well_coordinates
    y_max, x_max = yx_shape

    path = join(group.store.path, group.path)

    attrs = group.attrs.asdict()
    scale = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ]
    y_spacing, x_spacing = scale[-2], scale[-1]

    for k, ch in enumerate(attrs["omero"]["channels"]):
        key = f"{ch['wavelength_id']}_{ch['label']}"

        name = f"{group_name}_{key}"
        source_transform = {
            "sources": [name],
            "parameters": [
                1.0,
                0.0,
                0.0,
                (x_max + gap) * x_spacing * x_wc,
                0.0,
                1.0,
                0.0,
                (y_max + gap) * y_spacing * y_wc,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        }
        view = get_default_view(
            source_type="image",
            source_name=name,
            menu_name="Wells",
            source_transform=source_transform,
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

        if key not in plate_hists.keys():
            plate_hists[key] = copy(hists[k])
        else:
            plate_hists[key].combine(hists[k])

        if key not in sources.keys():
            sources[key] = [name]
        else:
            sources[key].append(name)

        if key not in source_transforms.keys():
            source_transforms[key] = [{"affine": source_transform}]
        else:
            source_transforms[key].append({"affine": source_transform})

        if key not in display_settings.keys():
            display_settings[key] = {
                "color": hex_to_rgba(ch["color"]),
                "opacity": 1.0,
                "sources": [
                    name,
                ],
            }
        else:
            display_settings[key]["sources"].append(name)


def add_wells_to_project(
    plate: zarr.Group,
    dataset_folder: str,
    well_group: str = "0",
    view_name: str = "default",
    gap: int = 50,
):
    """Add well groups of a ome-zarr plate to project

    :param plate: ome-zarr plate
    :param dataset_folder: mobie dataset folder
    :param well_group: zarr-index to zarr array. Defaults into zeroth field
    :param view_name: mobie view name
    :param gap: between wells in viewer
    :return:
    """
    y_max, x_max = get_largest_yx_field_shape(plate)

    sources = {}
    source_transforms = {}
    display_settings = {}
    plate_hists = {}

    # Add wells as individual sources
    wells = []
    for i, row in enumerate(plate.group_keys()):
        for j, col in enumerate(plate[row].group_keys()):
            attrs = plate[row][col][well_group].attrs.asdict()
            wells.append(row + col.zfill(2))
            path = join(plate.store.path, row, col, well_group)

            hists = [
                UIntHistogram.load(join(path, h_path)) for h_path in attrs["histograms"]
            ]

            add_group_to_project(
                dataset_folder=dataset_folder,
                group=plate[row][col][well_group],
                group_name=f"{row}{col.zfill(2)}",
                yx_well_coordinates=(i, j),
                yx_shape=(y_max, x_max),
                hists=hists,
                plate_hists=plate_hists,
                sources=sources,
                source_transforms=source_transforms,
                display_settings=display_settings,
                gap=gap,
            )

    _add_well_regions(dataset_folder, wells)

    # Creates an overview of all wells for each channel
    disp_settings, group_names, sources_ = _add_channel_plate_overviews(
        dataset_folder, display_settings, plate_hists, source_transforms, sources
    )

    # Create default overview with all channels and all wells
    wells_per_channel = _get_well_sources_per_channel(sources)
    well_source_transformations = _get_well_source_transformations(source_transforms)

    create_view(
        dataset_folder=dataset_folder,
        view_name=view_name,
        display_settings=disp_settings,
        display_group_names=group_names,
        sources=sources_,
        source_transforms=well_source_transformations,
        region_displays={
            "Wells": {
                "opacity": 0.5,
                "lut": "glasbey",
                "sources": wells_per_channel,
                "tableSource": "wells",
                "visible": True,
                "showAsBoundaries": True,
            }
        },
        menu_name="Overview",
    )


def _get_well_source_transformations(source_transforms):
    first_key = list(source_transforms.keys())[0]
    well_source_transformations = source_transforms[first_key]
    for c in filter(lambda k: k != first_key, source_transforms.keys()):
        for i, trafo in enumerate(source_transforms[c]):
            well_source_transformations[i]["affine"]["sources"].extend(
                trafo["affine"]["sources"]
            )
    return well_source_transformations


def _add_channel_plate_overviews(
    dataset_folder, display_settings, plate_hists, source_transforms, sources
):
    disp_settings = []
    group_names = []
    sources_ = []
    for ch in display_settings.keys():
        ds = display_settings[ch]
        ds["contrastLimits"] = [
            plate_hists[ch].quantile(0.01),
            plate_hists[ch].quantile(0.99),
        ]
        disp_settings.append(ds)

        group_names.append(ch)

        sources_.append(sources[ch])

        create_view(
            dataset_folder=dataset_folder,
            view_name=ch,
            display_settings=[ds],
            display_group_names=[ch],
            sources=[sources[ch]],
            source_transforms=source_transforms[ch],
            menu_name="Channels",
            is_exclusive=False,
        )
    return disp_settings, group_names, sources_


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
