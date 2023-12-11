import shutil

from faim_hcs.hcs.acquisition import PlateLayout, TileAlignmentOptions
from faim_hcs.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_hcs.hcs.imagexpress import StackAcquisition
from faim_hcs.stitching import stitching_utils


def main():
    plate = StackAcquisition(
        acquisition_dir="/home/tibuch/Gitrepos/faim-hcs/resources/Projection" "-Mix",
        alignment=TileAlignmentOptions.GRID,
    )
    shutil.rmtree("test-plate.zarr", ignore_errors=True)
    converter = ConvertToNGFFPlate(
        ngff_plate=NGFFPlate(
            root_dir=".",
            name="test-plate",
            layout=PlateLayout.I384,
            order_name="order",
            barcode="barcode",
        ),
        yx_binning=2,
        dask_chunk_size_factor=2,
        warp_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_mean,
    )

    converter.run(
        plate_acquisition=plate,
        well_sub_group="0",
        chunks=(1, 512, 512),
        max_layer=2,
    )

    # mips = SinglePlaneAcquisition(
    #     acquisition_dir="/home/tibuch/Gitrepos/faim-hcs/resources/Projection" "-Mix",
    #     alignment=TileAlignmentOptions.GRID,
    # )
    # converter.run(
    #     plate_acquisition=mips,
    #     well_sub_group="0",
    #     yx_binning=1,
    #     chunks=(512, 512),
    #     max_layer=2,
    # )


if __name__ == "__main__":
    main()
