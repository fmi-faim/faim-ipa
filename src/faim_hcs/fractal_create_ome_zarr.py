# OME-Zarr creation from MD Image Express
from faim_hcs.io.MolecularDevicesImageXpress import parse_files, parse_single_plane_multi_fields
from faim_hcs.Zarr import build_zarr_scaffold
from os.path import join, exists
import shutil

input_paths = ["../../resources/Projection-Mix/"]
output_path = "../../examples/zarr-files"
output_name = "Single-Plane"

order_name = ("example-order",)
barcode = "example-barcode"
overwrite = True
# Mode can be 3 values: "z-steps" (only parse the 3D data), "top-level" (only parse the 2D data), "all" (parse both)
mode = "z-steps"
mode = "top-level"
mode = "all"

# output_name = "Projection-Mix"
# is_2D = False

if len(input_paths) > 1:
    raise NotImplementedError(
        "MD Create OME-Zarr task is not implemented to handle multiple input paths"
    )

valid_modes = ("z-steps", "top-level", "all")
if mode not in valid_modes:
    raise NotImplementedError(
        f"Only implemented for modes {valid_modes}, but got mode {mode=}"
    )

files = parse_files(input_paths[0], mode=mode)

if overwrite and exists(join(output_path, output_name + ".zarr")):
    # Remove zarr if it already exists.
    shutil.rmtree(join(output_path, output_name + ".zarr"))

# Build empty zarr plate scaffold.
plate = build_zarr_scaffold(
    root_dir=output_path, 
    name=output_name, 
    files=files, 
    layout=96, 
    order_name=order_name, 
    barcode=barcode
)

# TODO: Create the metadata dictionary
