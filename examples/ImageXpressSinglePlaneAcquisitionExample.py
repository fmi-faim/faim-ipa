import shutil
from pathlib import Path

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_hcs.hcs.imagexpress import SinglePlaneAcquisition
from faim_hcs.hcs.plate import PlateLayout
from faim_hcs.stitching import stitching_utils


def main():
    # Remove existing zarr.
    shutil.rmtree("md-single-plane.zarr", ignore_errors=True)

    # Parse MD plate acquisition.
    plate = SinglePlaneAcquisition(
        acquisition_dir=Path(__file__).parent.parent / "resources" / "Projection-Mix",
        alignment=TileAlignmentOptions.GRID,
    )

    # Create converter.
    converter = ConvertToNGFFPlate(
        ngff_plate=NGFFPlate(
            root_dir=".",
            name="md-single-plane",
            layout=PlateLayout.I384,
            order_name="order",
            barcode="barcode",
        ),
        yx_binning=2,
        dask_chunk_size_factor=2,
        warp_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_mean,
    )

    # Run conversion.
    converter.run(
        plate_acquisition=plate,
        well_sub_group="0",
        chunks=(1, 512, 512),
        max_layer=2,
    )


if __name__ == "__main__":
    main()
