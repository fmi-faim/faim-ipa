import shutil
from pathlib import Path

import distributed

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.cellvoyager import StackAcquisition
from faim_hcs.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_hcs.hcs.plate import PlateLayout
from faim_hcs.stitching import stitching_utils


def main():
    # Remove existing zarr.
    shutil.rmtree("cv-stack.zarr", ignore_errors=True)

    # Parse CV plate acquisition.
    plate_acquisition = StackAcquisition(
        acquisition_dir=Path(__file__).parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack",
        alignment=TileAlignmentOptions.GRID,
        n_planes_in_stacked_tile=3,
    )

    # Create converter.
    converter = ConvertToNGFFPlate(
        ngff_plate=NGFFPlate(
            root_dir=".",
            name="cv-stack",
            layout=PlateLayout.I384,
            order_name="order",
            barcode="barcode",
        ),
        yx_binning=2,
        stitching_yx_chunk_size_factor=2,
        warp_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_mean,
        client=distributed.Client(threads_per_worker=1, processes=False, n_workers=1),
    )

    plate = converter.create_zarr_plate(plate_acquisition)

    # Run conversion.
    converter.run(
        plate=plate,
        plate_acquisition=plate_acquisition,
        well_sub_group="0",
        chunks=(3, 1000, 1000),
        max_layer=2,
    )


if __name__ == "__main__":
    main()
