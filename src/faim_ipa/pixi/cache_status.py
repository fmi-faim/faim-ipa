import json
import shutil
import subprocess
import sys
from inspect import cleandoc

GB = 1024 * 1024 * 1024


def min_cache_size_gb(gb: int = 2):
    info = subprocess.run(
        ["pixi", "info", "--json"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    cache_dir = json.loads(info.stdout)["cache_dir"]

    if shutil.disk_usage(cache_dir).free < gb * GB:
        sys.tracebacklimit = 0
        message = cleandoc(
            f"""
        Disk space in cache directory is below {gb} GB.

        PIXI_CACHE_DIR: {cache_dir}

        Did you initialize your session correctly ('source ./init.sh') ?
        """
        )
        raise RuntimeError(message)
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        min_cache_size_gb(gb=int(sys.argv[1]))
    min_cache_size_gb()
