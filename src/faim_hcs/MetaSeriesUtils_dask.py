from typing import Callable

import numpy as np
import dask.array as da
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
    im_count = np.zeros_like(im_fused, dtype='uint8')
    tile_count = np.ones_like(tiles[0], dtype='uint8')
    ny_tile, nx_tile = tiles.shape[-2:]
    
    for tile, pos in zip(tiles, positions):
        im_fused[pos[0]:pos[0]+ny_tile, pos[1]:pos[1]+nx_tile] += tile
        im_count[pos[0]:pos[0]+ny_tile, pos[1]:pos[1]+nx_tile] += tile_count
    
    with np.errstate(divide='ignore'):
        im_fused = im_fused//im_count
    return im_fused

def fuse_mean_gradient(tiles: ArrayLike, positions: ArrayLike) -> ArrayLike:
    """
    Fuses tiles according to positions.
    Where tiles overlap, it writes the mean with a linear gradient.
    tiles: ArrayLike, should have shape (tiles, ny, nx)
    positions: ArrayLike, should have shape (tiles, 2)
    """
    ny_tot, nx_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((ny_tot, nx_tot), dtype='uint32')
    im_weight = np.zeros_like(im_fused, dtype='uint32')
    ny_tile, nx_tile = tiles.shape[-2:]
    
    # distance map to border of image
    mask = np.ones((ny_tile, nx_tile))
    mask[[0,-1],:] = 0
    mask[:,[0,-1]] = 0
    dist_map = distance_transform_edt(mask).astype('uint32') + 1
    
    for tile, pos in zip(tiles, positions):
        im_fused[pos[0]:pos[0]+ny_tile, pos[1]:pos[1]+nx_tile] += tile*dist_map
        im_weight[pos[0]:pos[0]+ny_tile, pos[1]:pos[1]+nx_tile] += dist_map
    with np.errstate(divide='ignore'):
        im_fused = im_fused//im_weight
    return im_fused.astype(tiles.dtype)

def fuse_random_gradient(tiles: ArrayLike, positions: ArrayLike) -> ArrayLike:
    """
    Fuses tiles according to positions.
    Where tiles overlap, it chooses a random pixel, weighted according to  a
    linear gradient.
    tiles: ArrayLike, should have shape (tiles, ny, nx)
    positions: ArrayLike, should have shape (tiles, 2)
    """
    ny_tot, nx_tot = positions.max(axis=0) + tiles.shape[-2:]
    im_fused = np.zeros((len(tiles), ny_tot, nx_tot), dtype=tiles.dtype)
    im_weight = np.zeros_like(im_fused, dtype='uint16')
    ny_tile, nx_tile = tiles.shape[-2:]
    
    # distance map to border of image
    mask = np.ones((ny_tile, nx_tile))
    mask[[0,-1],:] = 0
    mask[:,[0,-1]] = 0
    dist_map = distance_transform_edt(mask).astype('uint32') + 1
    
    for i in range(len(tiles)):
        tile = tiles[i]
        pos = positions[i]
        im_fused[i, pos[0]:pos[0]+ny_tile, pos[1]:pos[1]+nx_tile] += tile
        im_weight[i, pos[0]:pos[0]+ny_tile, pos[1]:pos[1]+nx_tile] += dist_map
    im_weight[:,im_weight.max(axis=0) == 0] = 1
    im_weight = im_weight / im_weight.sum(axis=0)[np.newaxis,:,:]
    im_weight = np.cumsum(im_weight, axis=0)
    im_weight = np.insert(im_weight, 0, np.zeros_like(im_weight[0]), axis=0)
    im_rand = np.random.rand(*im_fused[0].shape)
    for i in range(len(im_fused)):
        im_fused[i,(im_rand < im_weight[i]) | (im_weight[i+1] < im_rand)] = 0
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
        im_fused[pos[0]:pos[0]+ny_tile, pos[1]:pos[1]+nx_tile] = tile
        
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
        im_fused[pos[0]:pos[0]+ny_tile, pos[1]:pos[1]+nx_tile] = tile
        
    return im_fused

def _fuse_xy(
    x: ArrayLike,
    assemble_fun: Callable,
    positions: ArrayLike,
) -> ArrayLike:
    """
    Load images and fuse them (used with dask.array.map_blocks())
    x: block of a dask array. axis[0]=fields & axis[-2:]=(y,x). The other axes
    can be anything.
    assemble_fun: function used to assemble tiles
    positions: Array of tile-positions in pixels
    ny_tot, nx_tot: yx-dimensions of output array
    """
    # x can have any number of additional axes between fields & yx
    # -> flatten those axes and loop over them
    # (ideally, to optimize memory-efficiency, the additional axes should only
    # have one element per axis)
    x_flat = np.reshape(x,
        newshape = (x.shape[0],) + (np.product(x.shape[1:-2]),) + x.shape[-2:]
    )
    ims_fused = []
    for i in range(x_flat.shape[1]):
        # assemble tiles into one image
        im_fused = assemble_fun(x_flat[:,i], positions)
        ims_fused.append(im_fused)

    ims_fused = np.array(ims_fused, dtype=x.dtype)
    # put fused images back in correct shape, according to input
    out_shape  = x.shape[1:-2] + ims_fused.shape[-2:]
    ims_fused = np.reshape(ims_fused, out_shape)
    return ims_fused

def fuse_dask(
    data: da.Array,
    positions: ArrayLike,
    assemble_fun: Callable
) -> da.Array:
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
            chunks=(1,)*(len(data.shape)-3) + (ny_tot, nx_tot),
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
        positions=positions
    )

    return imgs_fused_da