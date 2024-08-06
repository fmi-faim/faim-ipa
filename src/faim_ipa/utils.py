import logging
import os
from datetime import datetime
from pathlib import Path

import pydantic
from pydantic import BaseModel


def wavelength_to_rgb(wavelength, gamma=0.8):
    """This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html

    Obtained from https://gist.github.com/error454/65d7f392e1acd4a782fc
    """

    wavelength = float(wavelength)
    if 380 <= wavelength <= 440:  # noqa: PLR2004
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        r = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        g = 0.0
        b = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength <= 490:  # noqa: PLR2004
        r = 0.0
        g = ((wavelength - 440) / (490 - 440)) ** gamma
        b = 1.0
    elif 490 <= wavelength <= 510:  # noqa: PLR2004
        r = 0.0
        g = 1.0
        b = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength <= 580:  # noqa: PLR2004
        r = ((wavelength - 510) / (580 - 510)) ** gamma
        g = 1.0
        b = 0.0
    elif 580 <= wavelength <= 645:  # noqa: PLR2004
        r = 1.0
        g = (-(wavelength - 645) / (645 - 580)) ** gamma
        b = 0.0
    elif 645 <= wavelength <= 750:  # noqa: PLR2004
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        r = (1.0 * attenuation) ** gamma
        g = 0.0
        b = 0.0
    else:
        r = 0.0
        g = 0.0
        b = 0.0
    r *= 255
    g *= 255
    b *= 255
    return int(r), int(g), int(b)


def rgb_to_hex(r, g, b):
    """Convert RGB values into hex number."""
    return f"{r:02x}{g:02x}{b:02x}"


def create_logger(name: str, *, include_timestamp: bool = True) -> logging.Logger:
    """
    Create logger which logs to <timestamp>-<name>.log inside the current
    working directory.

    Parameters
    ----------
    name
        Name of the logger instance.
    """
    logger = logging.getLogger(name=name)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    handler = logging.FileHandler(
        f"{now}-{name}.log" if include_timestamp else f"{name}.log"
    )
    formatter = logging.Formatter(
        fmt="{asctime} - {name} - {levelname} - {message}",
        style="{",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def get_git_root() -> Path:
    """
    Recursively search for the directory containing the .git folder.

    Returns
    -------
    Path
        Path to the root of the git repository.
    """
    parent_dir = Path(__file__).parent
    while not (parent_dir / ".git").exists():
        parent_dir = parent_dir.parent

    return parent_dir


def resolve_with_git_root(relative_path: Path) -> Path:
    """
    Takes a relative path and resolves it relative to the git_root directory.

    Parameters
    ----------
    relative_path
        Path relative to the git root.

    Returns
    -------
    Path
        Absolute path to the file.
    """
    git_root = get_git_root()
    return (git_root / relative_path).resolve()


def make_relative_to_git_root(path: Path) -> Path:
    """
    Convert an absolute path to a path relative to the git_root directory.

    Parameters
    ----------
    path
        Absolute path to a file.

    Returns
    -------
    pathlib.Path
        Path relative to the git root.
    """
    git_root = get_git_root()
    try:
        # requires Python >= 3.12
        return path.relative_to(git_root, walk_up=True)
    except (ValueError, TypeError):
        # fallback for Python < 3.12
        return Path(os.path.relpath(path, git_root))


class IPAConfig(BaseModel):

    def make_paths_absolute(self):
        """
        Convert all `pathlib.Path` fields to absolute paths.

        The paths are assumed to be relative to a git root directory somewhere
        in the parent directories of the class implementing `IPAConfig`.
        """
        fields = (
            self.model_fields_set
            if pydantic.__version__.startswith("2")
            else self.__fields_set__
        )

        for f in fields:
            attr = getattr(self, f)
            if isinstance(attr, Path) and not attr.is_absolute():
                setattr(self, f, resolve_with_git_root(attr))

    def make_paths_relative(self):
        """
        Convert all `pathlib.Path` fields to relative paths.

        The resulting paths will be relative to the git-root directory
        somewhere in the parent directories of the class implementing
        `IPAConfig`.
        """
        fields = (
            self.model_fields_set
            if pydantic.__version__.startswith("2")
            else self.__fields_set__
        )

        for f in fields:
            attr = getattr(self, f)
            if isinstance(attr, Path) and attr.is_absolute():
                setattr(self, f, make_relative_to_git_root(attr))
