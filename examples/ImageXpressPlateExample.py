import shutil

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_hcs.hcs.imagexpress import SinglePlaneAcquisition, StackAcquisition
from faim_hcs.Zarr import PlateLayout


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
    )

    converter.run(
        plate_acquisition=plate,
        well_sub_group="0",
        yx_binning=1,
        chunks=(10, 512, 512),
        max_layer=2,
    )

    mips = SinglePlaneAcquisition(
        acquisition_dir="/home/tibuch/Gitrepos/faim-hcs/resources/Projection" "-Mix",
        alignment=TileAlignmentOptions.GRID,
    )
    converter.run(
        plate_acquisition=mips,
        well_sub_group="0/projections",
        yx_binning=1,
        chunks=(512, 512),
        max_layer=2,
    )


if __name__ == "__main__":
    main()
