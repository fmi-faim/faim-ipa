import os
from os.path import join

import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
from pydantic import BaseModel

from faim_hcs.io.acquisition import PlateAcquisition
from faim_hcs.stitching import stitching_utils
from faim_hcs.Zarr import PlateLayout, _get_row_cols


class NGFFPlate(BaseModel):
    root_dir: str
    name: str
    layout: PlateLayout
    order_name: str
    barcode: str


class ConvertToNGFFPlate:
    _plate_acquisition: PlateAcquisition
    _ngff_plate: NGFFPlate

    def __init__(
        self,
        plate_acquisition: PlateAcquisition,
        ngff_plate: NGFFPlate,
    ):
        self._plate_acquisition = plate_acquisition
        self._ngff_plate = ngff_plate

    def _create_zarr_plate(self) -> zarr.Group:
        rows, cols = _get_row_cols(layout=self._ngff_plate.layout)

        plate_path = join(self._ngff_plate.root_dir, self._ngff_plate.name + ".zarr")
        os.makedirs(plate_path, exist_ok=False)

        store = parse_url(plate_path, mode="w").store
        plate = zarr.group(store=store)

        write_plate_metadata(
            plate,
            columns=cols,
            rows=rows,
            wells=[f"{w[0]}/{w[1:]}" for w in self._plate_acquisition.get_well_names()],
            name=self._ngff_plate.name,
            field_count=1,
        )

        attrs = plate.attrs.asdict()
        attrs["order_name"] = self._ngff_plate.order_name
        attrs["barcode"] = self._ngff_plate.barcode
        plate.attrs.put(attrs)

        return plate

    def run(self, max_layer: int = 3):
        plate = self._create_zarr_plate()
        for well_acquisition in self._plate_acquisition.get_well_acquisitions():
            row, col = well_acquisition.get_row_col()
            well_group = plate.require_group(row).require_group(col)

            well_group.create_group("0")
            write_well_metadata(well_group, [{"path": "0"}])

            from faim_hcs.stitching import DaskTileStitcher

            stitcher = DaskTileStitcher(
                tiles=well_acquisition.get_tiles(),
                yx_chunk_shape=(2048, 2048),
                dtype=np.uint16,
            )

            image_da = stitcher.get_stitched_dask_array(
                warp_func=stitching_utils.translate_tiles_2d,
                fuse_func=stitching_utils.fuse_mean,
            )

            write_image(
                image=image_da.squeeze(),
                group=well_group["0"],
                axes=well_acquisition.get_axes(),
                storage_options=dict(
                    dimension_separator="/",
                    compressor=Blosc(cname="zstd", clevel=6, shuffle=Blosc.BITSHUFFLE),
                ),
                scaler=Scaler(max_layer=max_layer),
                coordinate_transformations=well_acquisition.get_coordinate_transformations(
                    max_layer=max_layer
                ),
            )
