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

if len(input_paths) > 1:
    raise NotImplementedError(
        "MD Create OME-Zarr task is not implemented to handle multiple input paths"
    )

# files = parse_files(input_paths[0])
files = parse_single_plane_multi_fields(input_paths[0])

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
