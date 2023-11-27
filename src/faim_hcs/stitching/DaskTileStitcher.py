from functools import partial
from typing import Callable

import numpy as np
from dask import array as da
from dask.array.core import normalize_chunks

from faim_hcs.stitching import stitching_utils
from faim_hcs.stitching.BoundingBox import BoundingBox
from faim_hcs.stitching.Tile import Tile


class DaskTileStitcher:
    """
    Stitch tiles into a single image using dask.
    """

    def __init__(
        self,
        tiles: list[Tile],
        yx_chunk_shape: tuple[int, int],
        dtype: np.dtype = np.uint16,
    ):
        self.tiles = stitching_utils.shift_to_origin(tiles)
        self.chunk_shape = (1,) + yx_chunk_shape
        self.dtype = dtype

        self._shape = self._compute_output_shape()
        self._n_chunks = self._compute_number_of_chunks()
        self._block_to_tile_map = self._compute_block_to_tile_map()

    def _compute_number_of_chunks(self):
        """
        Compute the number of chunks (blocks) in the stitched image.
        """
        return tuple(
            np.ceil(np.array(self._shape) / np.array(self.chunk_shape)).astype(int)
        )

    def _compute_block_to_tile_map(self):
        """
        Compute a map from block position to tiles that overlap with the block.
        """
        block_to_tile_map = {}
        for block_position in np.ndindex(self._n_chunks):
            block_to_tile_map[block_position] = []
            block_bbox = BoundingBox.from_pos_and_shape(
                position=tuple(block_position * np.array(self.chunk_shape)),
                shape=self.chunk_shape,
            )
            for tile in self.tiles:
                tile_bbox = BoundingBox.from_pos_and_shape(
                    position=tile.position,
                    shape=(1,) + tile.shape,
                )
                if block_bbox.overlaps(tile_bbox):
                    block_to_tile_map[block_position].append(tile)

        return block_to_tile_map

    def _compute_output_shape(self):
        """
        Compute the shape of the stitched image.
        """
        tile_extents = []
        for tile in self.tiles:
            tile_extents.append(tile.position + np.array((1,) + tile.shape))
        return tuple(np.max(tile_extents, axis=0))

    def get_stitched_dask_array(
        self,
        warp_func: Callable = stitching_utils.translate_tiles_2d,
        fuse_func: Callable = stitching_utils.fuse_mean,
    ) -> da.array:
        """
        Build the dask array for the stitched image.

        Parameters
        ----------
        warp_func :
            Function which warps the tiles into the correct position.
        fuse_func :
            Function which fuses the warped tiles.

        Returns
        -------

        """
        func = partial(
            stitching_utils.assemble_chunk,
            tile_map=self._block_to_tile_map,
            warp_func=warp_func,
            fuse_func=fuse_func,
            dtype=self.dtype,
        )

        return da.map_blocks(
            func=func,
            dtype=self.dtype,
            chunks=normalize_chunks(
                chunks=self.chunk_shape, shape=self._shape, dtype=self.dtype
            ),
        )

    def get_stitched_image(
        self,
        transform_func: Callable = stitching_utils.translate_tiles_2d,
        fuse_func: Callable = stitching_utils.fuse_mean,
    ) -> np.ndarray:
        """
        Stitch the tiles into a single image.

        Note: This runs in 'synchronous' mode i.e. single threaded. If you
        want to use multiple threads, use get_stitched_dask_array() and
        compute the result using a configured dask scheduler.

        Parameters
        ----------
        transform_func :
            Function to transform the tiles into the stitched image.
        fuse_func :
            Function to fuse the transformed tiles.

        Returns
        -------
        Fused image.
        """
        stitched = self.get_stitched_dask_array(
            warp_func=transform_func, fuse_func=fuse_func
        )
        return stitched.compute(scheduler="synchronous")
