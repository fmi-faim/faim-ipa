import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import distance_transform_edt


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

    for tile, pos in zip(tiles, positions, strict=True):
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

    for tile, pos in reversed(list(zip(tiles, positions, strict=True))):
        im_fused[pos[0] : pos[0] + ny_tile, pos[1] : pos[1] + nx_tile] = tile

    return im_fused
