import shutil
from pathlib import Path

import distributed

from faim_ipa.hcs.acquisition import TileAlignmentOptions
from faim_ipa.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_ipa.hcs.imagexpress import SinglePlaneAcquisition
from faim_ipa.hcs.plate import PlateLayout
from faim_ipa.stitching import stitching_utils


def main():
    # Remove existing zarr.
    shutil.rmtree("md-single-plane.zarr", ignore_errors=True)

    # Parse MD plate acquisition.
    plate_acquisition = SinglePlaneAcquisition(
        acquisition_dir=Path(__file__).parent.parent
        / "resources"
        / "ImageXpress"
        / "Projection-Mix",
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
        chunks=(512, 512),
        max_layer=2,
    )


if __name__ == "__main__":
    main()
