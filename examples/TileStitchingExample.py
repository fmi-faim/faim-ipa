import zarr
from numcodecs import Blosc
from numpy._typing import ArrayLike
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image

from faim_hcs.stitching import DaskTileStitcher, Tile, stitching_utils
from faim_hcs.stitching.Tile import TilePosition


def no_fuse(warped_tiles: ArrayLike, warped_masks: ArrayLike) -> ArrayLike:
    return warped_tiles[0]


def main():
    tiles = [
        Tile(
            path="/home/tibuch/Data/ctc/Fluo-N2DH-GOWT1/01/t000.tif",
            shape=(1024, 1024),
            position=TilePosition(time=0, channel=0, z=0, y=0, x=0),
        ),
        Tile(
            path="/home/tibuch/Data/ctc/Fluo-N2DH-GOWT1/01/t000.tif",
            shape=(1024, 1024),
            position=TilePosition(time=0, channel=1, z=0, y=0, x=0),
        ),
        Tile(
            path="/home/tibuch/Data/ctc/Fluo-N2DH-GOWT1/01/t000.tif",
            shape=(1024, 1024),
            position=TilePosition(time=1, channel=0, z=0, y=0, x=0),
        ),
        Tile(
            path="/home/tibuch/Data/ctc/Fluo-N2DH-GOWT1/01/t000.tif",
            shape=(1024, 1024),
            position=TilePosition(time=1, channel=1, z=0, y=0, x=0),
        ),
    ]

    stitcher = DaskTileStitcher(
        tiles=tiles,
        yx_chunk_shape=(1024, 1024),
    )

    stitched_img_da = stitcher.get_stitched_dask_array(
        warp_func=stitching_utils.translate_tiles_2d,
        fuse_func=no_fuse,
    )

    store = parse_url("stitched.zarr", mode="w").store
    zarr_file = zarr.group(store=store, overwrite=True)
    write_image(
        image=stitched_img_da,
        group=zarr_file,
        axes=["t", "c", "z", "y", "x"],
        storage_options=dict(
            dimension_separator="/",
            compressor=Blosc(cname="zstd", clevel=6, shuffle=Blosc.BITSHUFFLE),
        ),
        scaler=Scaler(
            max_layer=0,
        ),
    )


if __name__ == "__main__":
    main()
