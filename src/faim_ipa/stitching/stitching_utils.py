from copy import copy

import numpy as np
from numpy._typing import NDArray
from scipy.ndimage import distance_transform_cdt

from faim_ipa.stitching.tile import Tile, TilePosition


def fuse_linear(warped_tiles: NDArray, warped_distance_masks: NDArray) -> NDArray:
    """
    Fuse transformed tiles using a linear gradient to compute the weighted
    average where tiles are overlapping.

    Parameters
    ----------
    warped_tiles :
        Tile images transformed to the final image space.
    warped_distance_masks :
        Distance masks for the transformed tiles. They are non-zero for
        foreground pixels, and the value is the distance to the closest edge of
        the tile.

    Returns
    -------
    Fused image.
    """
    dtype = warped_tiles.dtype
    fused_image = np.zeros_like(warped_tiles[0], dtype=np.float32)

    if warped_tiles.shape[0] > 1:
        denominator = np.sum(warped_distance_masks, axis=0)

        for tile, mask in zip(warped_tiles, warped_distance_masks, strict=True):
            weight = np.divide(mask, denominator, where=denominator > 0)
            np.clip(weight, 0, 1, out=weight)
            np.add(
                fused_image,
                tile.astype(np.float32) * weight,
                out=fused_image,
                where=weight > 0,
            )
    else:
        np.add(
            fused_image,
            warped_tiles[0],
            out=fused_image,
            where=warped_distance_masks[0] > 0,
        )

    return np.clip(fused_image, 0, np.iinfo(dtype).max).astype(dtype)


def fuse_linear_random(
    warped_tiles: NDArray, warped_distance_masks: NDArray
) -> NDArray:
    """
    Fuse transformed tiles by sampling random pixels where tiles are
    overlapping, using a linear gradient to compute the random weights.

    Parameters
    ----------
    warped_tiles :
        Tile images transformed to the final image space.
    warped_distance_masks :
        Distance masks for the transformed tiles. They are non-zero for
        foreground pixels, and the value is the distance to the closest edge of
        the tile.

    Returns
    -------
    Fused image.
    """
    np.random.seed(0)
    dtype = warped_tiles.dtype
    if warped_tiles.shape[0] > 1:
        denominator = warped_distance_masks.sum(axis=0)
        weights = np.true_divide(
            warped_distance_masks, denominator, where=denominator > 0
        )
        weights = np.clip(np.nan_to_num(weights, nan=0, posinf=1, neginf=0), 0, 1)
        weights = np.cumsum(weights, axis=0)
        weights = np.insert(weights, 0, np.zeros_like(weights[0]), axis=0)
        rand_tile = np.random.rand(*warped_tiles.shape[1:])
        for i in range(len(warped_tiles)):
            warped_tiles[i, (rand_tile < weights[i]) | (weights[i + 1] < rand_tile)] = 0

    return np.sum(warped_tiles, axis=0).astype(dtype)


def fuse_mean(warped_tiles: NDArray, warped_distance_masks: NDArray) -> NDArray:
    """
    Fuse transformed tiles and compute the mean of the overlapping pixels.

    Parameters
    ----------
    warped_tiles :
        Tile images transformed to the final image space.
    warped_distance_masks :
        Distance masks for the transformed tiles. They are non-zero for
        foreground pixels, and the value is the distance to the closest edge of
        the tile.

    Returns
    -------
    Fused image.
    """
    denominator = np.sum(warped_distance_masks > 0, axis=0)
    fused_image = np.zeros_like(warped_tiles[0], dtype=np.float32)

    for tile, mask in zip(warped_tiles, warped_distance_masks, strict=True):
        weight = np.divide(mask > 0, denominator, where=denominator > 0)
        np.add(
            fused_image,
            tile.astype(np.float32) * weight,
            out=fused_image,
            where=weight > 0,
        )

    return np.clip(fused_image, 0, np.iinfo(warped_tiles.dtype).max).astype(
        warped_tiles.dtype
    )


def fuse_sum(
    warped_tiles: NDArray, warped_distance_masks: NDArray  # noqa: ARG001
) -> NDArray:
    """
    Fuse transformed tiles and compute the sum of the overlapping pixels.

    Parameters
    ----------
    warped_tiles :
        Tile images transformed to the final image space.
    warped_distance_masks :
        Distance masks for the transformed tiles. They are non-zero for
        foreground pixels, and the value is the distance to the closest edge of
        the tile. (Not used in this function)

    Returns
    -------
    Fused image.
    """
    fused_image = np.sum(warped_tiles, axis=0)
    return fused_image.astype(warped_tiles.dtype)


def fuse_overlay_fwd(warped_tiles: NDArray, warped_distance_masks: NDArray) -> NDArray:
    """
    Fuse transformed tiles. Where tiles overlap, the tile later in the sequence
    overwrites the earlier one.

    Parameters
    ----------
    warped_tiles :
        Tile images transformed to the final image space.
    warped_distance_masks :
        Distance masks for the transformed tiles. They are non-zero for
        foreground pixels, and the value is the distance to the closest edge of
        the tile.

    Returns
    -------
    Fused image.
    """

    warped_masks = warped_distance_masks.astype(bool)

    fused_image = np.zeros_like(warped_tiles[0])
    for tile, mask in zip(warped_tiles, warped_masks, strict=True):
        fused_image[mask] = tile[mask]

    return fused_image


def fuse_overlay_bwd(warped_tiles: NDArray, warped_distance_masks: NDArray) -> NDArray:
    """
    Fuse transformed tiles. Where tiles overlap, the tile earlier in the
    sequence overwrites the later one.

    Parameters
    ----------
    warped_tiles :
        Tile images transformed to the final image space.
    warped_distance_masks :
        Distance masks for the transformed tiles. They are non-zero for
        foreground pixels, and the value is the distance to the closest edge of
        the tile.

    Returns
    -------
    Fused image.
    """

    warped_masks = warped_distance_masks.astype(bool)

    fused_image = np.zeros_like(warped_tiles[0])
    for tile, mask in zip(reversed(warped_tiles), reversed(warped_masks), strict=True):
        fused_image[mask] = tile[mask]

    return fused_image


def translate_tiles_2d(
    block_info, chunk_shape, tiles, *, build_acquisition_mask: bool = False
):
    """
    Translate tiles to their relative position inside the given block.

    Parameters
    ----------
    block_info :
        da.map_blocks block_info.
    chunk_shape :
        shape of the chunk in zyx.
    tiles :
        list of tiles.

    Returns
    -------
        translated tiles, translated distance masks
    """
    array_location = block_info[None]["array-location"]
    chunk_zyx_origin = np.array(
        [array_location[2][0], array_location[3][0], array_location[4][0]]
    )

    if not all(tile.shape == tiles[0].shape for tile in tiles):
        msg = "All tiles must have the same shape."
        raise ValueError(msg)
    distance_mask = get_distance_mask(tiles[0].shape)

    warped_tiles = []
    warped_distance_masks = []
    for tile in tiles:
        tile_origin = np.array(tile.get_zyx_position())
        tile_data = (
            tile.load_data_mask() if build_acquisition_mask else tile.load_data()
        )
        if tile_data.ndim == 2:
            tile_data = tile_data[np.newaxis, ...]
        warped_tile = shift_yx(chunk_zyx_origin, tile_data, tile_origin, chunk_shape)
        warped_distance_mask = shift_yx(
            chunk_zyx_origin, distance_mask, tile_origin, chunk_shape
        )

        warped_tiles.append(warped_tile)
        warped_distance_masks.append(warped_distance_mask)

    return np.array(warped_tiles), np.array(warped_distance_masks)


def get_distance_mask(tile_shape):
    mask = np.zeros(tile_shape[-2:], dtype=bool)
    mask[1:-1, 1:-1] = True
    return distance_transform_cdt(mask, metric="taxicab").astype(np.uint16) + 1


def shift_yx(chunk_zyx_origin, tile_data, tile_origin, chunk_shape):
    warped_tile = np.zeros(chunk_shape, dtype=tile_data.dtype)
    yx_shift = (tile_origin - chunk_zyx_origin)[1:]
    if yx_shift[0] < 0:
        tile_start_y = abs(yx_shift[0])
        tile_end_y = min(tile_start_y + chunk_shape[1], tile_data.shape[-2])
    else:
        tile_start_y = 0
        tile_end_y = max(
            0, min(tile_start_y + chunk_shape[1] - yx_shift[0], tile_data.shape[-2])
        )
    if yx_shift[1] < 0:
        tile_start_x = abs(yx_shift[1])
        tile_end_x = min(tile_start_x + chunk_shape[2], tile_data.shape[-1])
    else:
        tile_start_x = 0
        tile_end_x = min(
            tile_start_x + chunk_shape[2] - yx_shift[1], tile_data.shape[-1]
        )
    tile_data = tile_data[..., tile_start_y:tile_end_y, tile_start_x:tile_end_x]
    if tile_data.size > 0:
        start_y = max(0, yx_shift[0])
        end_y = start_y + tile_data.shape[-2]
        start_x = max(0, yx_shift[1])
        end_x = start_x + tile_data.shape[-1]
        warped_tile[: tile_data.shape[0], start_y:end_y, start_x:end_x] = tile_data
    return warped_tile


def assemble_chunk(
    block_info=None,
    tile_map=None,
    warp_func=None,
    fuse_func=None,
    dtype=None,
    *,
    build_acquisition_mask: bool = False,
):
    """
    Assemble a chunk of the stitched image.

    Parameters
    ----------
    block_info :
        da.map_blocks block_info.
    tile_map :
        map of block positions to tiles.
    warp_func :
        function used to warp tiles.
    fuse_func :
        function used to fuse tiles.
    dtype :
        tile data type.

    Returns
    -------
        fused tiles corresponding to this block/chunk
    """
    chunk_location = block_info[None]["chunk-location"]
    chunk_shape = block_info[None]["chunk-shape"]
    tiles = tile_map[chunk_location]

    if len(tiles) > 0:
        warped_tiles, warped_distance_masks = warp_func(
            block_info,
            chunk_shape[-3:],
            tiles,
            build_acquisition_mask=build_acquisition_mask,
        )

        if len(tiles) > 1:
            stitched_img = fuse_func(
                warped_tiles,
                warped_distance_masks,
            )
            stitched_img = stitched_img[np.newaxis, np.newaxis, ...]
        else:
            stitched_img = warped_tiles[np.newaxis, ...]
    else:
        stitched_img = np.zeros(chunk_shape, dtype=dtype)

    return stitched_img


def shift_to_origin(tiles: list[Tile]) -> list[Tile]:
    """
    Shift tile positions such that the minimal position is (0, 0, 0, 0, 0).

    Parameters
    ----------
    tiles :
        List of tiles.

    Returns
    -------
    List of shifted tiles.
    """
    min_tile_origin = np.min([np.array(tile.get_position()) for tile in tiles], axis=0)
    shifted_tiles = copy(tiles)
    for tile in shifted_tiles:
        shifted_pos = np.array(tile.get_position()) - min_tile_origin
        tile.position = TilePosition(
            time=shifted_pos[0],
            channel=shifted_pos[1],
            z=shifted_pos[2],
            y=shifted_pos[3],
            x=shifted_pos[4],
        )
    return shifted_tiles
