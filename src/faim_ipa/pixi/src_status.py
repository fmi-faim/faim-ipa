import subprocess
import sys
from inspect import cleandoc


def git_status_clean(src_dir: str = "src"):
    command = ["git", "status", "--porcelain", src_dir]
    info = subprocess.run(command, capture_output=True, text=True, check=False)
    changes = len(info.stdout.splitlines())

    if changes > 0:
        sys.tracebacklimit = 0
        message = cleandoc(
            f"""
        There are {changes} untracked changes in {src_dir}.

        Please commit or stash before proceeding.
        """
        )
        raise RuntimeError(message)
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        git_status_clean(src_dir=str(sys.argv[1]))
    git_status_clean()
