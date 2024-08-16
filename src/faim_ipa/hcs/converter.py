from __future__ import annotations

import os
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import zarr
from dask.distributed import Client, wait
from numcodecs import Blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import (
    Axes,
    write_multiscales_metadata,
    write_plate_metadata,
    write_well_metadata,
)
from pint import Unit
from pydantic import BaseModel

from faim_ipa import dask_utils
from faim_ipa.hcs.plate import PlateLayout, get_rows_and_columns
from faim_ipa.stitching import stitching_utils

if TYPE_CHECKING:
    from collections.abc import Callable

    from faim_ipa.hcs.acquisition import PlateAcquisition


class NGFFPlate(BaseModel):
    root_dir: Path | str
    name: str
    layout: PlateLayout
    order_name: str
    barcode: str


class ConvertToNGFFPlate:
    """
    Convert a plate acquisition to an NGFF plate.
    """

    _ngff_plate: NGFFPlate

    def __init__(
        self,
        ngff_plate: NGFFPlate,
        yx_binning: int = 1,
        warp_func: Callable = stitching_utils.translate_tiles_2d,
        fuse_func: Callable = stitching_utils.fuse_mean,
        client: Client = None,
    ):
        """
        Parameters
        ----------
        ngff_plate :
            NGFF plate information.
        yx_binning :
            YX binning factor.
        warp_func :
            Function used to warp tile images.
        fuse_func :
            Function used to fuse tile images.
        client :
            Dask client used for the conversion.
        """
        assert (
            isinstance(yx_binning, int) and yx_binning >= 1
        ), "yx_binning must be an integer >= 1."
        self._ngff_plate = ngff_plate
        self._yx_binning = yx_binning
        self._warp_func = warp_func
        self._fuse_func = fuse_func
        self._client = client

    def create_zarr_plate(
        self, plate_acquisition: PlateAcquisition, wells: list[str] | None = None
    ) -> zarr.Group:
        """
        Create empty NGFF zarr plate.

        Note: Loads the plate from disk if it already exists.

        Parameters
        ----------
        plate_acquisition :
            A single plate acquisition.
        wells :
            List of wells to build. If None, all wells are built.
        """
        plate_path = join(self._ngff_plate.root_dir, self._ngff_plate.name + ".zarr")
        if not os.path.exists(plate_path):
            os.makedirs(plate_path, exist_ok=False)
            store = parse_url(plate_path, mode="w").store
            plate = zarr.group(store=store)

            rows, cols = get_rows_and_columns(layout=self._ngff_plate.layout)

            write_plate_metadata(
                plate,
                columns=cols,
                rows=rows,
                wells=[
                    f"{w[0]}/{w[1:]}" for w in plate_acquisition.get_well_names(wells)
                ],
                name=self._ngff_plate.name,
                field_count=1,
            )

            attrs = plate.attrs.asdict()
            attrs["order_name"] = self._ngff_plate.order_name
            attrs["barcode"] = self._ngff_plate.barcode
            plate.attrs.put(attrs)
            return plate
        store = parse_url(plate_path, mode="w").store
        return zarr.group(store=store)

    def run(
        self,
        plate: zarr.Group,
        plate_acquisition: PlateAcquisition,
        wells: list[str] | None = None,
        well_sub_group: str = "0",
        chunks: tuple[int, int] | tuple[int, int, int] = (2048, 2048),
        max_layer: int = 3,
        storage_options: dict | None = None,
        *,
        build_acquisition_mask: bool = False,
        overwrite: bool = False,
    ):
        """
        Convert a plate acquisition to an NGFF plate.

        Parameters
        ----------
        plate_acquisition :
            A single plate acquisition.
        well_sub_group :
            Name of the well subgroup.
        chunks :
            Chunk size in (Z)YX.
        max_layer :
            Maximum layer of the resolution pyramid layers.
        storage_options :
            Zarr storage options.
        build_acquisition_mask :
            Writes a boolean mask instead of the image data, indicating where the image data is present.
        overwrite :
            Overwrite existing data.

        Returns
        -------
            zarr.Group of the plate.
        """
        assert 2 <= len(chunks) <= 3, "Chunks must be 2D or 3D."
        assert len(chunks) == len(
            plate_acquisition.get_well_acquisitions()[0].get_tiles()[0].shape
        ), "Chunks must have the same number of dimensions as the tile shape."
        well_acquisitions = plate_acquisition.get_well_acquisitions(wells)

        for well_acquisition in well_acquisitions:
            well_group = self._create_well_group(
                plate,
                well_acquisition,
                well_sub_group,
                add_to_well_images=not build_acquisition_mask,
            )
            group = well_group[well_sub_group]
            self._write_stitched_image(
                group,
                chunks,
                plate_acquisition,
                storage_options,
                well_acquisition,
                build_acquisition_mask=build_acquisition_mask,
                overwrite=overwrite,
            )
            shapes, datasets = self._build_pyramid(
                group,
                chunks,
                max_layer,
                storage_options,
                overwrite=overwrite,
            )
            self._write_metadata(
                group, max_layer, shapes, datasets, plate_acquisition, well_acquisition
            )

        return plate

    def _write_metadata(
        self, group, max_layer, shapes, datasets, plate_acquisition, well_acquisition
    ):
        coordinate_transformations = well_acquisition.get_coordinate_transformations(
            max_layer=max_layer,
            yx_binning=self._yx_binning,
            ndim=len(shapes[0]),
        )
        fmt = CurrentFormat()
        dims = len(shapes[0])
        fmt.validate_coordinate_transformations(
            dims, len(datasets), coordinate_transformations
        )
        for dataset, transform in zip(
            datasets, coordinate_transformations, strict=True
        ):
            dataset["coordinateTransformations"] = transform
        axes = Axes(well_acquisition.get_axes(), fmt).to_list()
        spatial_calibration_unit = Unit(
            plate_acquisition.get_channel_metadata()[0].spatial_calibration_units
        )
        for axis in axes:
            if axis["name"] in ["z", "y", "x"]:
                axis["unit"] = str(spatial_calibration_unit)
        write_multiscales_metadata(
            group,
            datasets,
            fmt,
            axes,
        )
        group.attrs["omero"] = {
            "channels": plate_acquisition.get_omero_channel_metadata()
        }
        group.attrs["acquisition_metadata"] = {
            "channels": [
                ch_metadata.model_dump()
                for ch_metadata in plate_acquisition.get_channel_metadata().values()
            ]
        }

    def _write_stitched_image(
        self,
        group,
        chunks,
        plate_acquisition,
        storage_options,
        well_acquisition,
        build_acquisition_mask,
        overwrite,
    ):
        stitched_well_da = self._stitch_well_image(
            well_acquisition,
            output_shape=plate_acquisition.get_common_well_shape(),
            build_acquisition_mask=build_acquisition_mask,
        )
        binned_da = self._drop_missing_axes(stitched_well_da, well_acquisition)
        rechunked_da = binned_da.rechunk(self._out_chunks(binned_da.shape, chunks))
        options = self._get_storage_options(storage_options, rechunked_da.shape, chunks)
        wait(
            self._client.persist(
                da.to_zarr(
                    arr=rechunked_da,
                    url=group.store,
                    compute=False,
                    component=str(Path(group.path, "0")),
                    storage_options=options,
                    compressor=options.get(
                        "compressor", zarr.storage.default_compressor
                    ),
                    dimension_separator=group._store._dimension_separator,
                    overwrite=overwrite,
                ),
            )
        )

    def _drop_missing_axes(self, stitched_well_da, well_acquisition):
        drop_axes = tuple(
            i
            for i, s in enumerate(["t", "c", "z", "y", "x"])
            if s not in well_acquisition.get_axes()
        )
        binned_da = self._bin_yx(stitched_well_da)
        if len(drop_axes) > 0:
            binned_da = binned_da.squeeze(drop_axes)
        return binned_da

    def _build_pyramid(
        self,
        group,
        chunks,
        max_layer,
        storage_options,
        overwrite,
    ):
        image = da.from_zarr(url=group.store, component=str(Path(group.path, "0")))
        datasets = [{"path": "0"}]
        shapes = [image.shape]
        for path in range(1, max_layer + 1):
            image = da.coarsen(
                reduction=dask_utils.mean_cast_to(image.dtype),
                x=image,
                axes={
                    image.ndim - 2: 2,
                    image.ndim - 1: 2,
                },
                trim_excess=True,
            )
            options = self._get_storage_options(storage_options, image.shape, chunks)
            image = image.rechunk(options["chunks"])
            wait(
                self._client.persist(
                    da.to_zarr(
                        arr=image,
                        url=group.store,
                        compute=False,
                        component=str(Path(group.path, str(path))),
                        storage_options=options,
                        compressor=options.get(
                            "compressor", zarr.storage.default_compressor
                        ),
                        dimension_separator=group._store._dimension_separator,
                        overwrite=overwrite,
                    )
                )
            )
            datasets.append({"path": str(path)})
            shapes.append(image.shape)
            image = da.from_zarr(
                url=group.store, component=str(Path(group.path, str(path)))
            )

        return shapes, datasets

    def _bin_yx(self, image_da):
        if self._yx_binning > 1:
            return da.coarsen(
                reduction=dask_utils.mean_cast_to(image_da.dtype),
                x=image_da,
                axes={
                    0: 1,
                    1: 1,
                    2: 1,
                    3: self._yx_binning,
                    4: self._yx_binning,
                },
                trim_excess=True,
            )
        return image_da

    def _stitch_well_image(
        self,
        well_acquisition,
        output_shape: tuple[int, int, int, int, int],
        *,
        build_acquisition_mask: bool,
    ):
        from faim_ipa.stitching import DaskTileStitcher

        chunk_shape = well_acquisition.get_tiles()[0].shape

        stitcher = DaskTileStitcher(
            tiles=well_acquisition.get_tiles(),
            chunk_shape=chunk_shape,
            output_shape=output_shape,
            dtype=bool if build_acquisition_mask else well_acquisition.get_dtype(),
        )
        return stitcher.get_stitched_dask_array(
            warp_func=self._warp_func,
            fuse_func=(
                stitching_utils.fuse_sum if build_acquisition_mask else self._fuse_func
            ),
            build_acquisition_mask=build_acquisition_mask,
        )

    def _create_well_group(
        self, plate, well_acquisition, well_sub_group, *, add_to_well_images=True
    ):
        row, col = well_acquisition.get_row_col()
        well_group = plate.require_group(row).require_group(col)
        well_group.require_group(well_sub_group)
        if add_to_well_images:
            zattrs = well_group.attrs.asdict()
            if "well" in zattrs and "images" in zattrs["well"]:
                existing_images = well_group.attrs.asdict()["well"]["images"]
            else:
                existing_images = []
            write_well_metadata(
                well_group, [*existing_images, {"path": well_sub_group}]
            )
        return well_group

    @staticmethod
    def _get_storage_options(
        storage_options: dict,
        output_shape: tuple[int, ...],
        chunks: tuple[int, ...],
    ):
        if storage_options is None:
            return {
                "dimension_separator": "/",
                "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
                "chunks": ConvertToNGFFPlate._out_chunks(output_shape, chunks),
                "write_empty_chunks": False,
            }
        return storage_options

    @staticmethod
    def _out_chunks(shape, chunks):
        if len(shape) == len(chunks):
            return chunks
        return (1,) * (len(shape) - len(chunks)) + chunks
