from typing import Callable

import dask.array as da
import numpy as np
import pandas as pd
import tifffile
from numpy.typing import ArrayLike
from scipy.ndimage import distance_transform_edt


def fuse_mean(tiles: ArrayLike, positions: ArrayLike) -> ArrayLike:
    """
    Fuses tiles according to positions.
    Where tiles overlap, it writes the mean of all tiles.
    tiles: ArrayLike, should have shape (tiles, ny, nx)
    positions: ArrayLike, should have shape (tiles, 2)
    """
    ny_tot, nx_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((ny_tot, nx_tot), dtype=tiles.dtype)
    im_count = np.zeros_like(im_fused, dtype="uint8")
    tile_count = np.ones_like(tiles[0], dtype="uint8")
    ny_tile, nx_tile = tiles.shape[-2:]

    for tile, pos in zip(tiles, positions):
        im_fused[pos[0] : pos[0] + ny_tile, pos[1] : pos[1] + nx_tile] += tile
        im_count[pos[0] : pos[0] + ny_tile, pos[1] : pos[1] + nx_tile] += tile_count

    with np.errstate(divide="ignore"):
        im_fused = im_fused // im_count
    return im_fused


def fuse_mean_gradient(tiles: ArrayLike, positions: ArrayLike) -> ArrayLike:
    """
    Fuses tiles according to positions.
    Where tiles overlap, it writes the mean with a linear gradient.
    tiles: ArrayLike, should have shape (tiles, ny, nx)
    positions: ArrayLike, should have shape (tiles, 2)
    """
    ny_tot, nx_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((ny_tot, nx_tot), dtype="uint32")
    im_weight = np.zeros_like(im_fused, dtype="uint32")
    ny_tile, nx_tile = tiles.shape[-2:]

    # distance map to border of image
    mask = np.ones((ny_tile, nx_tile))
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0
    dist_map = distance_transform_edt(mask).astype("uint32") + 1

    for tile, pos in zip(tiles, positions):
        im_fused[pos[0] : pos[0] + ny_tile, pos[1] : pos[1] + nx_tile] += (
            tile * dist_map
        )
        im_weight[pos[0] : pos[0] + ny_tile, pos[1] : pos[1] + nx_tile] += dist_map
    with np.errstate(divide="ignore"):
        im_fused = im_fused // im_weight
    return im_fused.astype(tiles.dtype)


def fuse_random_gradient(
    tiles: ArrayLike, positions: ArrayLike, random_seed=0
) -> ArrayLike:
    """
    Fuses tiles according to positions.
    Where tiles overlap, it chooses a random pixel, weighted according to  a
    linear gradient.
    tiles: ArrayLike, should have shape (tiles, ny, nx)
    positions: ArrayLike, should have shape (tiles, 2)
    """
    np.random.seed(random_seed)
    ny_tot, nx_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((len(tiles), ny_tot, nx_tot), dtype=tiles.dtype)
    im_weight = np.zeros_like(im_fused, dtype="uint16")
    ny_tile, nx_tile = tiles.shape[-2:]

    # distance map to border of image
    mask = np.ones((ny_tile, nx_tile))
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0
    dist_map = distance_transform_edt(mask).astype("uint32") + 1

    for i in range(len(tiles)):
        tile = tiles[i]
        pos = positions[i]
        im_fused[i, pos[0] : pos[0] + ny_tile, pos[1] : pos[1] + nx_tile] += tile
        im_weight[i, pos[0] : pos[0] + ny_tile, pos[1] : pos[1] + nx_tile] += dist_map
    im_weight[:, im_weight.max(axis=0) == 0] = 1
    im_weight = im_weight / im_weight.sum(axis=0)[np.newaxis, :, :]
    im_weight = np.cumsum(im_weight, axis=0)
    im_weight = np.insert(im_weight, 0, np.zeros_like(im_weight[0]), axis=0)
    im_rand = np.random.rand(*im_fused[0].shape)
    for i in range(len(im_fused)):
        im_fused[i, (im_rand < im_weight[i]) | (im_weight[i + 1] < im_rand)] = 0
    return im_fused.sum(axis=0)


def fuse_fw(tiles: ArrayLike, positions: ArrayLike) -> ArrayLike:
    """
    Fuses tiles according to positions.
    Where tiles overlap, it overwrites with the tile that comes later in the
    sequence.
    tiles: ArrayLike, should have shape (tiles, ny, nx)
    positions: ArrayLike, should have shape (tiles, 2)
    """
    ny_tot, nx_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((ny_tot, nx_tot), dtype=tiles.dtype)
    ny_tile, nx_tile = tiles.shape[-2:]

    for tile, pos in zip(tiles, positions):
        im_fused[pos[0] : pos[0] + ny_tile, pos[1] : pos[1] + nx_tile] = tile

    return im_fused


def fuse_rev(tiles: ArrayLike, positions: ArrayLike) -> ArrayLike:
    """
    Fuses tiles according to positions.
    Where tiles overlap, it overwrites with the tile that comes earlier in the
    sequence.
    tiles: ArrayLike, should have shape (tiles, ny, nx)
    positions: ArrayLike, should have shape (tiles, 2)
    """
    ny_tot, nx_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((ny_tot, nx_tot), dtype=tiles.dtype)
    ny_tile, nx_tile = tiles.shape[-2:]

    for tile, pos in reversed(list(zip(tiles, positions))):
        im_fused[pos[0] : pos[0] + ny_tile, pos[1] : pos[1] + nx_tile] = tile

    return im_fused


def _fuse_xy(
    x: ArrayLike, assemble_fun: Callable, positions: ArrayLike, ny_tot: int, nx_tot: int
) -> ArrayLike:
    """
    Load images and fuse them (used with dask.array.map_blocks())
    x: block of a dask array. axis[0]=fields & axis[-2:]=(y,x). The other axes
    can be anything.
    assemble_fun: function used to assemble tiles
    positions: Array of tile-positions in pixels
    ny_tot, nx_tot: yx-dimensions of output array
    """
    ims_fused = np.empty(x.shape[1:-2] + (ny_tot, nx_tot), dtype=x.dtype)
    for i in np.ndindex(x.shape[1:-2]):
        slice_tuple = (slice(None),) + i  # workaround for slicing like [:,*i]
        ims_fused[i] = assemble_fun(x[slice_tuple], positions)

    return ims_fused


def fuse_dask(data: da.Array, positions: ArrayLike, assemble_fun: Callable) -> da.Array:
    """
    Function to lazily fuse tiles of a dask-array
    data:
        dask array of shape (fields,channels,planes,ny,nx)
        should have chunks (fields,1,1,ny,nx) or (1,1,1,ny,nx)
    positions:
        numpy-array of tile (y,x) positions in pixels
    """
    # calculate tile-shape (ny,nx) and fused-shape (ny_tot,nx_tot)
    ny, nx = data.shape[-2:]
    ny_tot, nx_tot = positions.max(axis=0) + (ny, nx)

    # determine chunks of output array
    out_chunks = da.core.normalize_chunks(
        chunks=(1,) * (len(data.shape) - 3) + (ny_tot, nx_tot),
        shape=data.shape[1:-2] + (ny_tot, nx_tot),
    )

    imgs_fused_da = da.map_blocks(
        _fuse_xy,
        data,
        chunks=out_chunks,
        drop_axis=0,
        meta=np.array((), dtype=data.dtype),
        # parameters for _fuse_da:
        assemble_fun=assemble_fun,
        positions=positions,
        ny_tot=ny_tot,
        nx_tot=nx_tot,
    )

    return imgs_fused_da


def create_filename_structure_FCZ(
    well_files: pd.DataFrame,
    channels: list[str],
) -> ArrayLike:
    """
    Assemble filenames in a numpy-array with ordering (field,channel,plane).
    This allows us to later easily map over the filenames to create a
    dask-array of the images.
    """
    planes = sorted(well_files["z"].unique(), key=int)
    fields = sorted(well_files["field"].unique(), key=int)
    # legacy: key=lambda s: int(re.findall(r"(\d+)", s)[0])

    # Create an empty np array to store the filenames in the correct structure
    fn_dtype = f"<U{max([len(fn) for fn in well_files['path']])}"
    fns_np = np.zeros(
        (len(fields), len(channels), len(planes)),
        dtype=fn_dtype,
    )

    # Store fns in correct position
    for s, field in enumerate(fields):
        field_files = well_files[well_files["field"] == field]
        for c, channel in enumerate(channels):
            channel_files = field_files[field_files["channel"] == channel]
            for z, plane in enumerate(planes):
                plane_files = channel_files[channel_files["z"] == plane]
                if len(plane_files) == 1:
                    fns_np[s, c, z] = list(plane_files["path"])[0]
                elif len(plane_files) > 1:
                    raise RuntimeError("Multiple files found for one FCZ")

    return fns_np


def create_filename_structure_FC(
    well_files: pd.DataFrame,
    channels: list[str],
) -> ArrayLike:
    """
    Assemble filenames in a numpy-array with ordering (field,channel).
    This allows us to later easily map over the filenames to create a
    dask-array of the images.
    """
    fields = sorted(well_files["field"].unique(), key=int)
    # legacy: key=lambda s: int(re.findall(r"(\d+)", s)[0])

    # Create an empty np array to store the filenames in the correct structure
    fn_dtype = f"<U{max([len(fn) for fn in well_files['path']])}"
    fns_np = np.zeros(
        (len(fields), len(channels)),
        dtype=fn_dtype,
    )

    # Store fns in correct position
    for s, field in enumerate(fields):
        field_files = well_files[well_files["field"] == field]
        for c, channel in enumerate(channels):
            channel_files = field_files[field_files["channel"] == channel]
            if len(channel_files) == 1:
                fns_np[s, c] = list(channel_files["path"])[0]
            elif len(channel_files) > 1:
                raise RuntimeError("Multiple files found for one FC")

    return fns_np


def _read_images(x: ArrayLike, ny: int, nx: int, im_dtype: type) -> ArrayLike:
    """
    read images from filenames in an array
    x: Array with one or more filenames, in any shape
    ny, nx: shape of one tile in pixesl
    dtype: dtype of image
    returns: numpy-array of images, arranged in same shape as input array
    """
    images = np.zeros(x.shape + (ny, nx), dtype=im_dtype)
    for i in np.ndindex(x.shape):
        filename = x[i]
        if filename != "":
            images[i] = tifffile.imread(filename)
    return images


def read_FCZYX(
    well_files: pd.DataFrame, channels: list[str], ny: int, nx: int, dtype: type
) -> da.Array:
    """
    reads images from tiff-files into a dask array of shape (F,C,Z,Y,X)
    """

    # load filenames into array, so we can easily map over it
    fns_np = create_filename_structure_FCZ(well_files, channels)
    fns_da = da.from_array(fns_np, chunks=(1,) * len(fns_np.shape))

    # create dask array of images
    images_da = da.map_blocks(
        _read_images,
        fns_da,
        chunks=da.core.normalize_chunks(
            (1,) * len(fns_da.shape) + (ny, nx), fns_da.shape + (ny, nx)
        ),
        new_axis=list(range(len(fns_da.shape), len(fns_da.shape) + 2)),
        meta=np.array((), dtype=dtype),
        # parameters for _read_images:
        ny=ny,
        nx=nx,
        im_dtype=dtype,
    )
    return images_da


def read_FCYX(
    well_files: pd.DataFrame, channels: list[str], ny: int, nx: int, dtype: type
) -> da.Array:
    """
    reads images from tiff-files into a dask array of shape (F,C,Z,Y,X)
    """
    # TODO: maye look into merging read_FCZYX and read FCYX

    # load filenames into array, so we can easily map over it
    fns_np = create_filename_structure_FC(well_files, channels)
    fns_da = da.from_array(fns_np, chunks=(1,) * len(fns_np.shape))

    # create dask array of images
    images_da = da.map_blocks(
        _read_images,
        fns_da,
        chunks=da.core.normalize_chunks(
            (1,) * len(fns_da.shape) + (ny, nx), fns_da.shape + (ny, nx)
        ),
        new_axis=list(range(len(fns_da.shape), len(fns_da.shape) + 2)),
        meta=np.array((), dtype=dtype),
        # parameters for _read_images:
        ny=ny,
        nx=nx,
        im_dtype=dtype,
    )
    return images_da
