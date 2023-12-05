import shutil

from faim_hcs.hcs.acquisition import TileAlignmentOptions
from faim_hcs.hcs.cellvoyager import StackAcquisition
from faim_hcs.hcs.converter import ConvertToNGFFPlate, NGFFPlate
from faim_hcs.Zarr import PlateLayout


def main():
    plate = StackAcquisition(
        acquisition_dir="/home/tibuch/Gitrepos/faim-hcs/resources/CV8000"
        "/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack_20230918_135839/CV8000-Minimal-DataSet-2C-3W-4S-FP2-stack",
        alignment=TileAlignmentOptions.GRID,
    )
    shutil.rmtree("cv-test-plate.zarr", ignore_errors=True)
    converter = ConvertToNGFFPlate(
        ngff_plate=NGFFPlate(
            root_dir=".",
            name="cv-test-plate",
            layout=PlateLayout.I384,
            order_name="order",
            barcode="barcode",
        ),
    )

    converter.run(
        plate_acquisition=plate,
        well_sub_group="0",
        yx_binning=2,
        chunks=(2, 1000, 1000),
        max_layer=2,
    )


if __name__ == "__main__":
    main()
