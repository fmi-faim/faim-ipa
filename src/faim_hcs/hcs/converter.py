import os
from os.path import join
from typing import Callable, Union

import dask.array as da
import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
from pydantic import BaseModel
from tqdm import tqdm

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
    _ngff_plate: NGFFPlate

    def __init__(
        self,
        ngff_plate: NGFFPlate,
        yx_binning: int = 1,
        dask_chunk_size_factor: int = 2,
        warp_func: Callable = stitching_utils.translate_tiles_2d,
        fuse_func: Callable = stitching_utils.fuse_mean,
    ):
        assert (
            isinstance(yx_binning, int) and yx_binning >= 1
        ), "yx_binning must be an integer >= 1."
        assert (
            isinstance(dask_chunk_size_factor, int) and dask_chunk_size_factor >= 1
        ), "dask_chunk_size_factor must be an integer >= 1."
        self._ngff_plate = ngff_plate
        self._yx_binning = yx_binning
        self._dask_chunk_size_factor = dask_chunk_size_factor
        self._warp_func = warp_func
        self._fuse_func = fuse_func

    def _create_zarr_plate(self, plate_acquisition: PlateAcquisition) -> zarr.Group:
        plate_path = join(self._ngff_plate.root_dir, self._ngff_plate.name + ".zarr")
        if not os.path.exists(plate_path):
            os.makedirs(plate_path, exist_ok=False)
            store = parse_url(plate_path, mode="w").store
            plate = zarr.group(store=store)

            rows, cols = _get_row_cols(layout=self._ngff_plate.layout)

            write_plate_metadata(
                plate,
                columns=cols,
                rows=rows,
                wells=[f"{w[0]}/{w[1:]}" for w in plate_acquisition.get_well_names()],
                name=self._ngff_plate.name,
                field_count=1,
            )

            attrs = plate.attrs.asdict()
            attrs["order_name"] = self._ngff_plate.order_name
            attrs["barcode"] = self._ngff_plate.barcode
            plate.attrs.put(attrs)
            return plate
        else:
            store = parse_url(plate_path, mode="w").store
            return zarr.group(store=store)

    def run(
        self,
        plate_acquisition: PlateAcquisition,
        well_sub_group: str = "0",
        chunks: Union[tuple[int, int], tuple[int, int, int]] = (2048, 2048),
        max_layer: int = 3,
        storage_options: dict = None,
    ):
        assert 2 <= len(chunks) <= 3, "Chunks must be 2D or 3D."
        plate = self._create_zarr_plate(plate_acquisition)
        for well_acquisition in tqdm(plate_acquisition.get_well_acquisitions()):
            well_group = self._create_well_group(
                plate,
                well_acquisition,
                well_sub_group,
            )

            stitched_well_da = self._stitch_well_image(
                chunks,
                well_acquisition,
                output_shape=plate_acquisition.get_common_well_shape(),
            )

            output_da = self._bin_yx(stitched_well_da)

            write_image(
                image=output_da,
                group=well_group[well_sub_group],
                axes=well_acquisition.get_axes(),
                storage_options=self._get_storage_options(
                    storage_options, output_da.shape, chunks
                ),
                scaler=Scaler(max_layer=max_layer),
                coordinate_transformations=well_acquisition.get_coordinate_transformations(
                    max_layer=max_layer,
                    yx_binning=self._yx_binning,
                ),
            )

            well_group[well_sub_group].attrs["omero"] = {
                "channels": plate_acquisition.get_omero_channel_metadata()
            }

            well_group[well_sub_group].attrs["acquisition_metadata"] = {
                "channels": [
                    ch_metadata.dict()
                    for ch_metadata in plate_acquisition.get_channel_metadata().values()
                ]
            }

    def _bin_yx(self, image_da):
        if self._yx_binning > 1:
            output_da = da.coarsen(
                reduction=self._mean_cast_to(image_da.dtype),
                x=image_da,
                axes={
                    0: 1,
                    1: 1,
                    2: 1,
                    3: self._yx_binning,
                    4: self._yx_binning,
                },
                trim_excess=True,
            ).squeeze()
        else:
            output_da = image_da.squeeze()
        return output_da

    def _stitch_well_image(
        self,
        chunks,
        well_acquisition,
        output_shape: tuple[int, int, int, int, int],
    ):
        from faim_hcs.stitching import DaskTileStitcher

        stitcher = DaskTileStitcher(
            tiles=well_acquisition.get_tiles(),
            yx_chunk_shape=(
                chunks[-2] * self._dask_chunk_size_factor,
                chunks[-1] * self._dask_chunk_size_factor,
            ),
            output_shape=output_shape,
            dtype=well_acquisition.get_dtype(),
        )
        image_da = stitcher.get_stitched_dask_array(
            warp_func=self._warp_func,
            fuse_func=self._fuse_func,
        )
        return image_da

    def _create_well_group(self, plate, well_acquisition, well_sub_group):
        row, col = well_acquisition.get_row_col()
        well_group = plate.require_group(row).require_group(col)
        well_group.require_group(well_sub_group)
        write_well_metadata(well_group, [{"path": well_sub_group}])
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

    def _get_storage_options(
        self,
        storage_options: dict,
        output_shape: tuple[int, ...],
        chunks: tuple[int, ...],
    ):
        if storage_options is None:
            return dict(
                dimension_separator="/",
                compressor=Blosc(cname="zstd", clevel=6, shuffle=Blosc.BITSHUFFLE),
                chunks=self._out_chunks(output_shape, chunks),
                write_empty_chunks=False,
            )
        else:
            return storage_options
