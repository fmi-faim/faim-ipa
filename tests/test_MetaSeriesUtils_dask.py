from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from faim_hcs.io.MolecularDevicesImageXpress import parse_files
from faim_hcs.MetaSeriesUtils_dask import (
    _fuse_xy,
    _read_images,
    create_filename_structure_FC,
    create_filename_structure_FCZ,
    fuse_dask,
    fuse_fw,
    fuse_mean,
    fuse_mean_gradient,
    fuse_random_gradient,
    fuse_rev,
    read_FCYX,
    read_FCZYX,
)

ROOT_DIR = Path(__file__).parent.parent


def tiles():
    fake_tiles = [
        np.ones((8, 8), dtype=np.uint8),
        np.ones((8, 8), dtype=np.uint8) + 1,
        np.ones((8, 8), dtype=np.uint8) + 2,
    ]
    return np.stack(fake_tiles, axis=0)


def positions():
    return np.array([[0, 0], [0, 4], [3, 1]])


def block_FYX():
    return tiles()


def block_FCYX():
    return np.reshape(tiles(), (tiles().shape[0],) + (1,) + tiles().shape[-2:])


def block_FCZYX():
    return np.reshape(tiles(), (tiles().shape[0],) + (1, 1) + tiles().shape[-2:])


def block_FTCZYX():
    return np.reshape(tiles(), (tiles().shape[0],) + (1, 1, 1) + tiles().shape[-2:])


def block_FCZYX_2():
    return np.reshape(
        np.stack([tiles(), tiles()]), (tiles().shape[0],) + (1, 2) + tiles().shape[-2:]
    )


def dask_data_FCYX():
    data = np.stack(
        [
            tiles(),
        ]
        * 4,
        axis=1,
    )
    return da.from_array(
        data,
        chunks=(
            data.shape[0],
            1,
        )
        + data.shape[-2:],
    )


def dask_data_FCZYX():
    data = np.stack(
        [
            tiles(),
        ]
        * 4,
        axis=1,
    )
    data = np.stack(
        [
            data,
        ]
        * 5,
        axis=2,
    )
    return da.from_array(
        data,
        chunks=(
            data.shape[0],
            1,
            1,
        )
        + data.shape[-2:],
    )


def dask_data_FTCZYX():
    data = np.stack(
        [
            tiles(),
        ]
        * 4,
        axis=1,
    )
    data = np.stack(
        [
            data,
        ]
        * 5,
        axis=2,
    )
    data = np.stack(
        [
            data,
        ]
        * 6,
        axis=1,
    )
    return da.from_array(
        data,
        chunks=(
            data.shape[0],
            1,
            1,
            1,
        )
        + data.shape[-2:],
    )


def files():
    output = parse_files(ROOT_DIR / "resources" / "Projection-Mix")
    output.field = [el[1:] for el in output.field]
    output.channel = [el[1:] for el in output.channel]
    return output


def files_FCZ():
    files_all = files()
    return files_all[np.logical_and(files_all.z.notnull(), files_all.well == "E07")]


def files_FC():
    files_all = files()
    return files_all[np.logical_and(files_all.z.isnull(), files_all.well == "E07")]


def fns_np_FC():
    return create_filename_structure_FC(files_FC(), ["1", "2", "3", "4"])


def fns_np_FCZ():
    return create_filename_structure_FCZ(files_FCZ(), ["1", "2", "3", "4"])


@pytest.mark.parametrize(
    "tiles,positions",
    [
        (tiles(), positions()),
    ],
)
def test_fuse_rev(tiles, positions):
    fused_result = fuse_rev(tiles=tiles, positions=positions)
    # should be the same for all fuse-functions:
    assert fused_result.shape == (11, 12)
    assert fused_result[2, 3] == 1
    assert fused_result[2, 8] == 2
    assert fused_result[8, 3] == 3
    assert fused_result[8, 9] == 0
    # depends on fuse-functions:
    assert fused_result[3, 4] == 1
    assert fused_result[3, 7] == 1
    assert fused_result[7, 4] == 1
    assert fused_result[7, 7] == 1


@pytest.mark.parametrize(
    "tiles,positions",
    [
        (tiles(), positions()),
    ],
)
def test_fuse_fw(tiles, positions):
    fused_result = fuse_fw(tiles=tiles, positions=positions)
    # should be the same for all fuse-functions:
    assert fused_result.shape == (11, 12)
    assert fused_result[2, 3] == 1
    assert fused_result[2, 8] == 2
    assert fused_result[8, 3] == 3
    assert fused_result[8, 9] == 0
    # depends on fuse-functions:
    assert fused_result[3, 4] == 3
    assert fused_result[3, 7] == 3
    assert fused_result[7, 4] == 3
    assert fused_result[7, 7] == 3


@pytest.mark.parametrize(
    "tiles,positions",
    [
        (tiles(), positions()),
    ],
)
def test_fuse_mean(tiles, positions):
    fused_result = fuse_mean(tiles=tiles, positions=positions)
    # should be the same for all fuse-functions:
    assert fused_result.shape == (11, 12)
    assert fused_result[2, 3] == 1
    assert fused_result[2, 8] == 2
    assert fused_result[8, 3] == 3
    assert fused_result[8, 9] == 0
    # depends on fuse-functions:
    assert fused_result[3, 4] == 2
    assert fused_result[3, 7] == 2
    assert fused_result[7, 4] == 2
    assert fused_result[7, 7] == 2


@pytest.mark.parametrize(
    "tiles,positions",
    [
        (tiles(), positions()),
    ],
)
def test_fuse_mean_gradient(tiles, positions):
    fused_result = fuse_mean_gradient(tiles=tiles, positions=positions)
    # should be the same for all fuse-functions:
    assert fused_result.shape == (11, 12)
    assert fused_result[2, 3] == 1
    assert fused_result[2, 8] == 2
    assert fused_result[8, 3] == 3
    assert fused_result[8, 9] == 0
    # depends on fuse-functions:
    assert fused_result[3, 4] == 1
    assert fused_result[3, 7] == 2
    assert fused_result[7, 4] == 2
    assert fused_result[7, 7] == 2


@pytest.mark.parametrize(
    "tiles,positions",
    [
        (tiles(), positions()),
    ],
)
def test_fuse_random_gradient(tiles, positions):
    fused_result = fuse_random_gradient(tiles=tiles, positions=positions)
    # should be the same for all fuse-functions:
    assert fused_result.shape == (11, 12)
    assert fused_result[2, 3] == 1
    assert fused_result[2, 8] == 2
    assert fused_result[8, 3] == 3
    assert fused_result[8, 9] == 0
    # depends on fuse-functions:
    assert fused_result[3, 4] == 1
    assert fused_result[3, 7] == 1
    assert fused_result[7, 4] == 3
    assert fused_result[7, 7] == 3


@pytest.mark.parametrize(
    "x,assemble_fun,positions,ny_tot,nx_tot,expected",
    [
        (block_FYX(), fuse_rev, positions(), 11, 12, ((11, 12),)),
        (block_FCYX(), fuse_rev, positions(), 11, 12, ((1, 11, 12),)),
        (block_FCZYX(), fuse_rev, positions(), 11, 12, ((1, 1, 11, 12),)),
        (block_FTCZYX(), fuse_rev, positions(), 11, 12, ((1, 1, 1, 11, 12),)),
        (block_FCZYX_2(), fuse_rev, positions(), 11, 12, ((1, 2, 11, 12),)),
    ],
)
def test__fuse_xy(x, assemble_fun, positions, ny_tot, nx_tot, expected):
    ims_fused = _fuse_xy(x, assemble_fun, positions, ny_tot, nx_tot)
    assert ims_fused.shape == expected[0]


@pytest.mark.parametrize(
    "data,positions,assemble_fun,expected",
    [
        (
            dask_data_FCYX(),
            positions(),
            fuse_rev,
            (
                (4, 11, 12),
                (1, 11, 12),
            ),
        ),
        (
            dask_data_FCZYX(),
            positions(),
            fuse_rev,
            (
                (4, 5, 11, 12),
                (1, 1, 11, 12),
            ),
        ),
        (
            dask_data_FTCZYX(),
            positions(),
            fuse_rev,
            (
                (6, 4, 5, 11, 12),
                (1, 1, 1, 11, 12),
            ),
        ),
    ],
)
def test_fuse_dask(data, positions, assemble_fun, expected):
    imgs_fused_da = fuse_dask(data, positions, assemble_fun)
    assert imgs_fused_da.shape == expected[0]
    assert imgs_fused_da.chunksize == expected[1]
    assert imgs_fused_da.compute().shape == expected[0]


@pytest.mark.parametrize(
    "well_files,channels",
    [
        (files_FCZ(), ["1", "2", "3", "4"]),
    ],
)
def test_create_filename_structure_FCZ(well_files, channels):
    fns_np = create_filename_structure_FCZ(well_files, channels)
    assert fns_np.shape == (2, 4, 10)
    assert "" not in fns_np[:, :2]
    assert np.unique(fns_np[:, 2]) == [""]
    assert len(np.unique(fns_np[:, 3])) == 3


@pytest.mark.parametrize(
    "well_files,channels",
    [
        (files_FC(), ["1", "2", "3", "4"]),
    ],
)
def test_create_filename_structure_FC(well_files, channels):
    fns_np = create_filename_structure_FC(well_files, channels)
    assert fns_np.shape == (2, 4)
    assert "" not in fns_np[:, :-1]
    assert np.unique(fns_np[:, 3]) == [""]


@pytest.mark.parametrize(
    "x,ny,nx,dtype",
    [
        (fns_np_FC(), 512, 512, np.uint16),
        (fns_np_FCZ(), 512, 512, np.uint16),
    ],
)
def test__read_images(x, ny, nx, dtype):
    images = _read_images(x, ny, nx, dtype)
    assert images.shape == x.shape + (ny, nx)
    assert np.unique(images[x == ""]) == [
        0,
    ]


@pytest.mark.parametrize(
    "well_files, channels, ny, nx, dtype",
    [
        (files_FCZ(), ["1", "2", "3", "4"], 512, 512, np.uint16),
    ],
)
def test_read_FCZYX(well_files, channels, ny, nx, dtype):
    images_da = read_FCZYX(well_files, channels, ny, nx, dtype)
    images = images_da.compute()
    assert images_da.shape == (2, 4, 10, ny, nx)
    assert np.unique(images[:, 2]) == [
        0,
    ]
    assert np.unique(images[:, 3, 1:]) == [
        0,
    ]
    assert [
        0,
    ] not in [list(np.unique(images[i])) for i in np.ndindex((2, 2, 10))]


@pytest.mark.parametrize(
    "well_files, channels, ny, nx, dtype",
    [
        (files_FC(), ["1", "2", "3", "4"], 512, 512, np.uint16),
    ],
)
def test_read_FCYX(well_files, channels, ny, nx, dtype):
    images_da = read_FCYX(well_files, channels, ny, nx, dtype)
    images = images_da.compute()
    assert images_da.shape == (2, 4, ny, nx)
    assert np.unique(images[:, 3]) == [
        0,
    ]
    assert [
        0,
    ] not in [list(np.unique(images[i])) for i in np.ndindex((2, 3))]
