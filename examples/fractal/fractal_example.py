# Fractal example scripts

from faim_hcs.fractal.fractal_create_ome_zarr_md import create_ome_zarr_md
from faim_hcs.fractal.fractal_md_to_ome_zarr import md_to_ome_zarr

input_paths = ["../../resources/Projection-Mix"]
output_path = "../zarr-files"

order_name = "example-order"
barcode = "example-barcode"
overwrite = True
# Mode can be 3 values: "z-steps" (only parse the 3D data),
# "top-level" (only parse the 2D data), "all" (parse both)
# mode = "z-steps"
# mode = "top-level"
mode = "all"

output_name = "TaskTest"

metatada_update = create_ome_zarr_md(
    input_paths=input_paths,
    output_path=output_path,
    metadata={},
    zarr_name=output_name,
    mode=mode,
    order_name=order_name,
    barcode=barcode,
    overwrite=overwrite,
)

for component in metatada_update["image"]:
    md_to_ome_zarr(
        input_paths=[output_path],
        output_path=output_path,
        component=component,
        metadata=metatada_update,
    )
