import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import questionary
import yaml
from pydantic import (
    BaseModel,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
)
from questionary import ValidationError as QuestionaryValidationError
from questionary import Validator


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
    Create logger which logs to `<timestamp>-<name>.log` inside the current
    working directory.

    Parameters
    ----------
    name
        Name of the logger instance.
    include_timestamp
        Whether to include the timestamp in the log file name.

    Returns
    -------
    logging.Logger
        Logger instance
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
    Recursively search for the directory containing the `.git` folder.

    Returns
    -------
    Path
        Path to the root of the git repository. None if not found.
    """
    parent_dir = Path.cwd()
    while not (parent_dir / ".git").exists():
        if parent_dir == parent_dir.parent:
            return None  # reached root directory
        parent_dir = parent_dir.parent

    return parent_dir


def resolve(relative_path: Path, reference: Path) -> Path:
    """
    Takes a relative path and resolves it relative to the reference directory.

    Parameters
    ----------
    relative_path
        Path relative to the reference.
    reference
        Reference path.

    Returns
    -------
    Path
        Absolute path to the file.
    """
    if relative_path.is_absolute():
        return relative_path
    return (reference / relative_path).absolute()


def make_relative(path: Path, reference: Path):
    """
    Convert an absolute path to a path relative to the reference directory.

    Parameters
    ----------
    path
        Absolute path to a file.
    reference
        Path to the reference directory.

    Returns
    -------
    pathlib.Path
        Path relative to the reference.
    """
    try:
        # requires Python >= 3.12
        return path.relative_to(reference, walk_up=True)
    except (ValueError, TypeError):
        # fallback for Python < 3.12
        return Path(os.path.relpath(path, reference))


def prompt_with_questionary(
    model: "IPAConfig",
    defaults: dict | None = None,
):
    schema = model.model_json_schema()
    defaults = defaults or {}
    responses = {}

    for field_name, field_info in schema["properties"].items():
        description = field_info.get("description", field_name)
        field_type = field_info["type"]
        default_value = defaults.get(field_name, "")

        if field_type == "string":
            if field_info.get("format") == "path":
                default_path = str(resolve(Path(default_value), model.reference_dir()))
                responses[field_name] = Path(
                    questionary.path(
                        f"Enter {description} [{default_value}]",
                        validate=PathValidator(
                            model=model,
                            field_name=field_name,
                        ),
                        default=default_path,
                    ).ask()
                ).absolute()
            elif field_info.get("format") == "directory-path":
                default_path = str(resolve(Path(default_value), model.reference_dir()))
                responses[field_name] = Path(
                    questionary.path(
                        f"Enter {description} (directory) [{default_value}]",
                        validate=PathValidator(
                            model=model,
                            field_name=field_name,
                        ),
                        default=default_path,
                    ).ask()
                ).absolute()
            else:
                responses[field_name] = questionary.text(
                    f"Enter {description} [{default_value}]",
                    validate=QuestionaryPydanticValidator(
                        model=model, field_name=field_name
                    ),
                    default=default_value,
                ).ask()
        elif field_type == "integer":
            min_val = field_info.get("minimum", None)
            max_val = field_info.get("maximum", None)
            prompt_message = f"{description} ({f'minimum: {min_val}' if min_val else ''}-{f'maximum: {max_val}' if max_val else ''}) [{default_value}]"
            responses[field_name] = int(
                questionary.text(
                    prompt_message,
                    validate=QuestionaryPydanticValidator(
                        model=model, field_name=field_name
                    ),
                    default=str(default_value),
                ).ask()
            )
        elif field_type == "number":
            min_val = field_info.get("minimum", None)
            max_val = field_info.get("maximum", None)
            prompt_message = f"{description} ({f'minimum: {min_val}' if min_val else ''}-{f'maximum: {max_val}' if max_val else ''}) [{default_value}]"
            responses[field_name] = float(
                questionary.text(
                    prompt_message,
                    validate=QuestionaryPydanticValidator(
                        model=model, field_name=field_name
                    ),
                    default=str(default_value),
                ).ask()
            )
        elif field_type == "boolean":
            responses[field_name] = questionary.confirm(
                prompt_message,
                default=default_value,
            ).ask()
        else:
            msg = f"Unknown field type: {field_type}"
            raise ValueError(msg)

    return model(**responses)


class QuestionaryPydanticValidator(Validator):
    def __init__(self, model: BaseModel, field_name: str):
        self.field_name = field_name
        self.model = model
        self.field_info = model.model_fields[field_name]
        self.type_adapter = TypeAdapter(self.field_info.annotation)

    def _preprocess(self, value):
        return value

    def validate(self, document):
        try:
            value = self._preprocess(self.type_adapter.validate_python(document.text))
            self.model.__pydantic_validator__.validate_assignment(
                self.model.model_construct(), self.field_name, value
            )
        except ValidationError as e:
            raise QuestionaryValidationError(
                message=f"Invalid value for field: {e.errors()[0]['msg']}"
            ) from e


class PathValidator(QuestionaryPydanticValidator):
    def __init__(self, model: BaseModel, field_name: str):
        super().__init__(model, field_name)

    def _preprocess(self, value):
        return Path(value).absolute()


T = TypeVar("T", bound="IPAConfig")


class IPAConfig(BaseModel):

    @field_serializer("*")
    @classmethod
    def path_relative_to_git(cls, value):
        if isinstance(value, Path):
            try:
                return make_relative(value, cls.reference_dir()).as_posix()
            except ValueError:
                return str(value)
        return value

    @field_validator("*", mode="before")
    @classmethod
    def git_relative_path_to_absolute(cls, value, info):
        field_name = info.field_name
        field_type = cls.__annotations__[field_name]
        if isinstance(field_type, type) and issubclass(field_type, Path):
            return resolve(Path(value), cls.reference_dir())
        if hasattr(field_type, "__metadata__") and issubclass(
            field_type.__origin__, Path
        ):
            return resolve(Path(value), cls.reference_dir())
        return value

    @staticmethod
    def reference_dir():
        return get_git_root()

    @staticmethod
    def config_name():
        return "config.yml"

    def save(self, config_file=None):
        config_file = config_file or Path.cwd() / self.config_name()
        with open(config_file, "w") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)

    @classmethod
    def load(cls: type[T], config_file=None) -> T:
        config_file = config_file or Path.cwd() / cls.config_name()
        with open(config_file) as f:
            return cls(**yaml.safe_load(f))
