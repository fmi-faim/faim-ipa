import os
from os.path import basename
from pathlib import Path

import pytest
import questionary
from pydantic import DirectoryPath, Field, create_model
from questionary import ValidationError

from faim_ipa.utils import (
    IPAConfig,
    QuestionaryPydanticValidator,
    create_logger,
    prompt_with_questionary,
    wavelength_to_rgb,
)


@pytest.fixture
def dummy_file(tmp_path: Path):
    tmp_file = tmp_path / "dummy.txt"
    with open(tmp_file, "w"):
        pass
    return tmp_file


@pytest.fixture
def model():
    return create_model(
        "TestModel",
        path=(Path, Field(..., description="File path")),
        directory=(DirectoryPath, Field(..., description="Folder path")),
        string=(str, Field(..., description="Some text")),
        ge0=(int, Field(..., ge=0)),
        number=(float, Field(..., gt=0.0)),
        boolean=(bool, Field(..., description="Checkbox")),
    )


def test_wavelength_to_rgb():
    assert wavelength_to_rgb(370) == (0, 0, 0)
    assert wavelength_to_rgb(380) == (97, 0, 97)
    assert wavelength_to_rgb(440) == (0, 0, 255)
    assert wavelength_to_rgb(490) == (0, 255, 255)
    assert wavelength_to_rgb(510) == (0, 255, 0)
    assert wavelength_to_rgb(580) == (255, 255, 0)
    assert wavelength_to_rgb(645) == (255, 0, 0)
    assert wavelength_to_rgb(750) == (97, 0, 0)
    assert wavelength_to_rgb(751) == (0, 0, 0)


def test_create_logger(tmp_path_factory):
    os.chdir(tmp_path_factory.mktemp("logs"))
    logger = create_logger("test")
    logger.info("Test")
    assert logger.name == "test"
    assert basename(logger.handlers[0].baseFilename).endswith("-test.log")

    with open(logger.handlers[0].baseFilename) as f:
        assert f.read().strip()[-11:] == "INFO - Test"


def test_validator(dummy_file: Path, model):
    class Document:
        text: str

        def __init__(self, text):
            self.text = text

    path_validator = QuestionaryPydanticValidator(model=model, field_name="path")
    assert (
        path_validator.validate(document=Document(str(dummy_file.absolute()))) is None
    )

    string_validator = QuestionaryPydanticValidator(model=model, field_name="string")
    assert string_validator.validate(document=Document("some text")) is None

    directory_validator = QuestionaryPydanticValidator(
        model=model, field_name="directory"
    )
    with pytest.raises(ValidationError):
        directory_validator.validate(document=Document(str(dummy_file.absolute())))

    ge0_validator = QuestionaryPydanticValidator(model=model, field_name="ge0")
    with pytest.raises(ValidationError):
        ge0_validator.validate(document=Document("-1"))


def test_prompt_with_questionary(model, mocker, dummy_file):
    class Question:
        def __init__(self, answers):
            self.answers = answers
            self._answerer = self._answer()

        def _answer(self):
            for answer in self.answers:
                yield answer

        def ask(self):
            return next(self._answerer)

    text_patch = mocker.patch(
        "questionary.text", return_value=Question(["text", "10", "0.01"])
    )
    path_patch = mocker.patch(
        "questionary.path",
        return_value=Question(
            [str(dummy_file.absolute()), str(dummy_file.parent.absolute())]
        ),
    )
    confirm_patch = mocker.patch("questionary.confirm", return_value=Question([True]))
    response = prompt_with_questionary(model=model)
    questionary.text.assert_called()
    questionary.path.assert_called()
    assert text_patch.call_count == 3
    assert path_patch.call_count == 2
    assert confirm_patch.call_count == 1
    assert response.model_dump() == {
        "boolean": True,
        "directory": dummy_file.parent.absolute(),
        "ge0": 10,
        "number": 0.01,
        "path": dummy_file.absolute(),
        "string": "text",
    }


def test_ipa_config(tmp_path):
    class SomeConfig(IPAConfig):
        string: str
        integer: int
        number: float

    config = SomeConfig(string="dummy", integer=42, number=-0.01)
    config_path = tmp_path / "config.yml"
    config.save(config_file=config_path)

    loaded_config = SomeConfig.load(config_file=config_path)

    assert loaded_config.model_dump() == config.model_dump()
