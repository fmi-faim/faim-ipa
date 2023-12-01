import shutil

from faim_hcs.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_hcs.hcs.imagexpress import MixedAcquisition
from faim_hcs.io.acquisition import TileAlignmentOptions
from faim_hcs.Zarr import PlateLayout


def main():
    plate = MixedAcquisition(
        acquisition_dir="/home/tibuch/Gitrepos/faim-hcs/resources/Projection" "-Mix",
        alignment=TileAlignmentOptions.GRID,
    )
    shutil.rmtree("test-plate.zarr", ignore_errors=True)
    converter = ConvertToNGFFPlate(
        plate_acquisition=plate,
        ngff_plate=NGFFPlate(
            root_dir=".",
            name="test-plate",
            layout=PlateLayout.I384,
            order_name="order",
            barcode="barcode",
        ),
    )

    converter.run(
        yx_binning=2,
        chunks=(10, 512, 512),
        max_layer=2,
    )


if __name__ == "__main__":
    main()
