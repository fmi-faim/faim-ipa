import os
import re
from inspect import cleandoc
from typing import NamedTuple

import pytest

from faim_ipa import pixi
from faim_ipa.pixi.cache_status import min_cache_size_gb
from faim_ipa.pixi.log_commit import log_commit
from faim_ipa.pixi.src_status import git_status_clean


def test_src_status_fail(mocker):
    mock_result = mocker.Mock()
    mock_result.stdout = cleandoc("""
        A some/file
        M some/other/file
        """)
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


def test_src_status_succeed(mocker):
    mock_result = mocker.Mock()
    mock_result.stdout = ""
    mock_result.returncode = 0

    mocker.patch("faim_ipa.pixi.src_status.subprocess.run", return_value=mock_result)

    with pytest.raises(SystemExit) as excinfo:
        git_status_clean(src_dir="mock_source")
    assert excinfo.value.code == 0

    pixi.src_status.subprocess.run.assert_called_once_with(
        ["git", "status", "--porcelain", "mock_source"],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cache_status_fail(mocker):
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


def test_cache_status_succeed(mocker):
    mock_result_pixi = mocker.Mock()
    mock_result_pixi.stdout = '{"cache_dir": "/some/folder"}'
    mock_result_pixi.returncode = 0

    class Usage(NamedTuple):
        free: int

    mock_usage = Usage
    mock_disk_usage = mocker.Mock(return_value=mock_usage(free=2 * 1024 * 1024 * 1024))

    mocker.patch(
        "faim_ipa.pixi.cache_status.subprocess.run", return_value=mock_result_pixi
    )
    mocker.patch(
        "faim_ipa.pixi.cache_status.shutil.disk_usage", side_effect=mock_disk_usage
    )

    with pytest.raises(SystemExit) as excinfo:
        min_cache_size_gb()
    assert excinfo.value.code == 0

    pixi.cache_status.subprocess.run.assert_called_once_with(
        ["pixi", "info", "--json"], text=True, capture_output=True, check=False
    )
    pixi.cache_status.shutil.disk_usage.assert_called_once_with("/some/folder")


def test_log_commit(tmp_path):
    os.environ["WD"] = str(tmp_path)
    log_commit(task="TESTING")
    log_path = tmp_path / "githash.log"
    assert log_path.exists()
    pattern = r"^\d{4}-\d{2}-\d{2}.*githash.*INFO.*Executing.*TESTING.*git-commit\] [0-9a-f]*$"
    with open(log_path) as log:
        assert re.match(pattern=pattern, string=log.readline())
        assert not log.readline()
