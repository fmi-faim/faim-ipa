from inspect import cleandoc
from typing import NamedTuple

import pytest

from faim_ipa import pixi
from faim_ipa.pixi.cache_status import min_cache_size_gb
from faim_ipa.pixi.src_status import git_status_clean


def test_src_status(mocker):
    mock_result = mocker.Mock()
    mock_result.stdout = cleandoc(
        """
        A some/file
        M some/other/file
        """
    )
    mock_result.returncode = 1

    mocker.patch("faim_ipa.pixi.src_status.subprocess.run", return_value=mock_result)

    with pytest.raises(
        RuntimeError, match="There are 2 untracked changes in mock_source"
    ):
        git_status_clean(src_dir="mock_source")

    pixi.src_status.subprocess.run.assert_called_once_with(
        ["git", "status", "--porcelain", "mock_source"],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cache_status(mocker):
    mock_result_pixi = mocker.Mock()
    mock_result_pixi.stdout = '{"cache_dir": "/some/folder"}'
    mock_result_pixi.returncode = 0

    class Usage(NamedTuple):
        free: int

    mock_usage = Usage
    mock_disk_usage = mocker.Mock(return_value=mock_usage(free=0))

    mocker.patch(
        "faim_ipa.pixi.cache_status.subprocess.run", return_value=mock_result_pixi
    )
    mocker.patch(
        "faim_ipa.pixi.cache_status.shutil.disk_usage", side_effect=mock_disk_usage
    )

    with pytest.raises(
        RuntimeError, match="Disk space in cache directory is below 1 GB"
    ):
        min_cache_size_gb(gb=1)

    pixi.cache_status.subprocess.run.assert_called_once_with(
        ["pixi", "info", "--json"], text=True, capture_output=True, check=False
    )
    pixi.cache_status.shutil.disk_usage.assert_called_once_with("/some/folder")
