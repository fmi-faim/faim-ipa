import os
from os.path import join
from typing import Optional

import dask.array as da
import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
from pydantic import BaseModel

from faim_hcs.hcs.acquisition import PlateAcquisition
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

    def run(
        self,
        yx_binning: int = 1,
        chunks: tuple[int, int, Optional[int]] = (2048, 2048),
        max_layer: int = 3,
    ):
        assert (
            isinstance(yx_binning, int) and yx_binning >= 1
        ), "yx_binning must be an integer >= 1."
        assert 2 <= len(chunks) <= 3, "Chunks must be 2D or 3D."
        plate = self._create_zarr_plate()
        for well_acquisition in self._plate_acquisition.get_well_acquisitions():
            well_group = self._create_well_group(plate, well_acquisition)

            stitched_well_da = self._stitch_well_image(chunks, well_acquisition)

            output_da = self._bin_yx(stitched_well_da, yx_binning)

            write_image(
                image=output_da,
                group=well_group["0"],
                axes=well_acquisition.get_axes(),
                chunks=self._out_chunks(output_da.shape, chunks),
                storage_options=dict(
                    dimension_separator="/",
                    compressor=Blosc(cname="zstd", clevel=6, shuffle=Blosc.BITSHUFFLE),
                ),
                scaler=Scaler(max_layer=max_layer),
                coordinate_transformations=well_acquisition.get_coordinate_transformations(
                    max_layer=max_layer,
                    yx_binning=yx_binning,
                ),
            )

    def _bin_yx(self, image_da, yx_binning):
        if yx_binning > 1:
            output_da = da.coarsen(
                reduction=self._mean_cast_to(image_da.dtype),
                x=image_da,
                axes={
                    0: 1,
                    1: 1,
                    2: 1,
                    3: yx_binning,
                    4: yx_binning,
                },
                trim_excess=False,
            ).squeeze()
        else:
            output_da = image_da.squeeze()
        return output_da

    def _stitch_well_image(self, chunks, well_acquisition):
        from faim_hcs.stitching import DaskTileStitcher

        stitcher = DaskTileStitcher(
            tiles=well_acquisition.get_tiles(),
            yx_chunk_shape=(chunks[-2], chunks[-1]),
            dtype=np.uint16,
        )
        image_da = stitcher.get_stitched_dask_array(
            warp_func=stitching_utils.translate_tiles_2d,
            fuse_func=stitching_utils.fuse_mean,
        )
        return image_da

    def _create_well_group(self, plate, well_acquisition):
        row, col = well_acquisition.get_row_col()
        well_group = plate.require_group(row).require_group(col)
        well_group.create_group("0")
        write_well_metadata(well_group, [{"path": "0"}])
        return well_group

    @staticmethod
    def _mean_cast_to(target_dtype):
        def _mean(
            a,
            axis=None,
            dtype=None,
            out=None,
            keepdims=np._NoValue,
            *,
            where=np._NoValue,
        ):
            return np.mean(
                a=a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
            ).astype(target_dtype)

        return _mean

    @staticmethod
    def _out_chunks(shape, chunks):
        if len(shape) == len(chunks):
            return chunks
        else:
            return (1,) * (len(shape) - len(chunks)) + chunks
