import datetime
import shutil

from faim_ipa.hcs.cellvoyager.source import CVSourceZIP
from faim_ipa.hcs.cellvoyager.acquisition import (
    ZAdjustedStackAcquisition,
    TileAlignmentOptions,
)
from distributed import Client
import dask.array as da
from faim_ipa.stitching import DaskTileStitcher, stitching_utils
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscales_metadata

from numcodecs import Blosc
import dask


def init():
    import numcodecs

    numcodecs.blosc.use_threads = True
    numcodecs.blosc.set_nthreads(4)


def main():
    print("Remove output directory.")
    shutil.rmtree("./C07-ZIP.zarr", ignore_errors=True)
    dask.config.set({"distributed.scheduler.work-stealing": False})
    client = Client(
        n_workers=3,
        threads_per_worker=1,
        memory_limit="5GB",
        local_directory="./dask_tmp/",
    )
    client.run(init)

    print(client.dashboard_link)

    start = datetime.datetime.now()
    h5_src = CVSourceZIP(
        "/tungstenfs/scratch/gscicomp_share/gmicro/test-data-with-logs/jetraw_v3.zip",
        "jetraw/C07-data",
    )

    plate = ZAdjustedStackAcquisition(
        source=h5_src,
        alignment=TileAlignmentOptions.STAGE_POSITION,
        n_planes_in_stacked_tile=16,
    )

    c07 = plate.get_well_acquisitions(["C07"])[0]

    tiles = c07.get_tiles()

    stitcher = DaskTileStitcher(
        tiles=tiles,
        chunk_shape=(16, 2000, 2000),
        output_shape=c07.get_shape(),
        dtype=c07.get_dtype(),
    )

    image_da = stitcher.get_stitched_dask_array(
        warp_func=stitching_utils.translate_tiles_2d,
        fuse_func=stitching_utils.fuse_linear,
    ).squeeze()

    storage_options = dict(
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        chunks=(16, 2000, 2000),
        write_empty_chunks=False,
    )

    group = zarr.group(parse_url("./C07-ZIP.zarr", mode="w").store, overwrite=True)

    write_multiscales_metadata(
        group=group,
        datasets=[
            {
                "path": "0",
                "coordinateTransformations": c07.get_coordinate_transformations(
                    max_layer=0,
                    yx_binning=1,
                    ndim=4,
                )[0],
            }
        ],
        axes=["c", "z", "y", "x"],
    )

    da.to_zarr(
        arr=image_da,
        url=group.store,
        component="0",
        storage_options=storage_options,
        compressor=storage_options["compressor"],
        dimension_separator="/",
    )

    end = datetime.datetime.now()
    print(f"Convert from ZIP source took {end - start}.")
    print("Done.")
    client.shutdown()


if __name__ == "__main__":
    main()
