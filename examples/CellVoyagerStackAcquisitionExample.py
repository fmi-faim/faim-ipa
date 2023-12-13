import shutil
from pathlib import Path

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.cellvoyager import StackAcquisition
from faim_hcs.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_hcs.hcs.plate import PlateLayout
from faim_hcs.stitching import stitching_utils


def main():
    # Remove existing zarr.
    shutil.rmtree("cv-stack.zarr", ignore_errors=True)

    # Parse CV plate acquisition.
    plate = StackAcquisition(
        acquisition_dir=Path(__file__).parent.parent
        / "resources"
        / "CV8000"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839"
        / "CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack",
        alignment=TileAlignmentOptions.GRID,
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
        dask_chunk_size_factor=2,
        warp_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_mean,
    )

    # Run conversion.
    converter.run(
        plate_acquisition=plate,
        well_sub_group="0",
        chunks=(2, 1000, 1000),
        max_layer=2,
    )


if __name__ == "__main__":
    main()
