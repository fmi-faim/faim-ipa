import os
from os.path import basename

from faim_ipa.utils import create_logger, wavelength_to_rgb


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
    assert logger.name == "Test"
    assert basename(logger.handlers[0].baseFilename).endswith("-test.log")
