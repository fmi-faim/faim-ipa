# OME-Zarr creation from MD Image Express
from typing import Any, Dict, Sequence

from faim_hcs.io.MolecularDevicesImageXpress import parse_files
from faim_hcs.Zarr import build_zarr_scaffold
from os.path import join, exists
import shutil


def create_ome_zarr_md(
    *,
    input_paths: Sequence[str],
    output_path: str,
    metadata: Dict[str, Any],
    zarr_name: str = "Plate",
    mode: str = "all",
    order_name: str = "example-order",
    barcode: str = "example-barcode",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    TBD
    # Mode can be 3 values: "z-steps" (only parse the 3D data), 
    "top-level" (only parse the 2D data), "all" (parse both)

    """
    if len(input_paths) > 1:
        raise NotImplementedError(
            "MD Create OME-Zarr task is not implemented to handle multiple input paths"
        )
    order_name = (order_name,)

    valid_modes = ("z-steps", "top-level", "all")
    if mode not in valid_modes:
        raise NotImplementedError(
            f"Only implemented for modes {valid_modes}, but got mode {mode=}"
        )

    files = parse_files(input_paths[0], mode=mode)

    if overwrite and exists(join(output_path, zarr_name + ".zarr")):
        # Remove zarr if it already exists.
        shutil.rmtree(join(output_path, zarr_name + ".zarr"))

    # Build empty zarr plate scaffold.
    build_zarr_scaffold(
        root_dir=output_path,
        name=zarr_name,
        files=files,
        layout=96,
        order_name=order_name,
        barcode=barcode,
    )

    # Create the metadata dictionary
    plate_name = zarr_name + ".zarr"
    well_paths = []
    image_paths = []
    for well in sorted(files["well"].unique()):
        curr_well = plate_name + "/" + well[0] + "/" + str(int(well[1:])) + "/"
        well_paths.append(curr_well)
        image_paths.append(curr_well + "0/")

    # FIXME: Find a way to figure out here how many levels will be generated
    # (to be able to put it into the num_levels metadata)
    num_levels = 1

    metadata_update = dict(
        plate=[plate_name],
        well=well_paths,
        image=image_paths,
        num_levels=num_levels,
        coarsening_xy=2,
        channels=files["channel"].unique().tolist(),
        mode=mode,
        original_paths=input_paths[:],
    )
    return metadata_update


# TODO: Add main function to run this as a fractal task
