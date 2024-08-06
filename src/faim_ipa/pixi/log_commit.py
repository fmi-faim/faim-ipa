import os
import subprocess
import sys

from faim_ipa.utils import create_logger


def log_commit(task: str = ""):
    command = ["git", "rev-parse", "--verify", "HEAD"]
    info = subprocess.run(command, capture_output=True, text=True, check=False)
    hash_string = info.stdout.strip()

    os.chdir(path=os.getenv("WD", default="."))
    logger = create_logger(name="githash", include_timestamp=False)
    logger.info("[Executing] %s [git-commit] %s", task, hash_string)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_commit(task=str(sys.argv[1]))
    else:
        log_commit()
