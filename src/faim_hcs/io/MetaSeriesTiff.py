from pathlib import Path

from numpy._typing import ArrayLike
from tifffile import tifffile


def load_metaseries_tiff(path: Path) -> tuple[ArrayLike, dict]:
    """Load metaseries tiff file and parts of its metadata.

    The following metadata is collected:
    * _IllumSetting_
    * spatial-calibration-x
    * spatial-calibration-y
    * spatial-calibration-units
    * stage-position-x
    * stage-position-y
    * PixelType
    * _MagNA_
    * _MagSetting_
    * Exposure Time
    * Lumencor Cyan Intensity
    * Lumencor Green Intensity
    * Lumencor Red Intensity
    * Lumencor Violet Intensity
    * Lumencor Yellow Intensity
    * ShadingCorrection
    * stage-label
    * SiteX
    * SiteY
    * wavelength

    :param path:
    :return:
    image_data, metadata-dict
    """
    with tifffile.TiffFile(path) as tiff:
        assert tiff.is_metaseries, f"{path} is not a metamorph file."
        data = tiff.asarray()
        metadata = {
            "_IllumSetting_": tiff.metaseries_metadata["PlaneInfo"]["_IllumSetting_"],
            "spatial-calibration-x": tiff.metaseries_metadata["PlaneInfo"][
                "spatial-calibration-x"
            ],
            "spatial-calibration-y": tiff.metaseries_metadata["PlaneInfo"][
                "spatial-calibration-y"
            ],
            "spatial-calibration-units": tiff.metaseries_metadata["PlaneInfo"][
                "spatial-calibration-units"
            ],
            "stage-position-x": tiff.metaseries_metadata["PlaneInfo"][
                "stage-position-x"
            ],
            "stage-position-y": tiff.metaseries_metadata["PlaneInfo"][
                "stage-position-y"
            ],
            "z-position": tiff.metaseries_metadata["PlaneInfo"]["z-position"],
            "PixelType": str(data.dtype),
            "_MagNA_": tiff.metaseries_metadata["PlaneInfo"]["_MagNA_"],
            "_MagSetting_": tiff.metaseries_metadata["PlaneInfo"]["_MagSetting_"],
            "Exposure Time": tiff.metaseries_metadata["PlaneInfo"]["Exposure Time"],
            "Lumencor Cyan Intensity": tiff.metaseries_metadata["PlaneInfo"][
                "Lumencor Cyan Intensity"
            ],
            "Lumencor Green Intensity": tiff.metaseries_metadata["PlaneInfo"][
                "Lumencor Green Intensity"
            ],
            "Lumencor Red Intensity": tiff.metaseries_metadata["PlaneInfo"][
                "Lumencor Red Intensity"
            ],
            "Lumencor Violet Intensity": tiff.metaseries_metadata["PlaneInfo"][
                "Lumencor Violet Intensity"
            ],
            "Lumencor Yellow Intensity": tiff.metaseries_metadata["PlaneInfo"][
                "Lumencor Yellow Intensity"
            ],
            "ShadingCorrection": tiff.metaseries_metadata["PlaneInfo"][
                "ShadingCorrection"
            ],
            "stage-label": tiff.metaseries_metadata["PlaneInfo"]["stage-label"],
            "SiteX": tiff.metaseries_metadata["PlaneInfo"]["SiteX"],
            "SiteY": tiff.metaseries_metadata["PlaneInfo"]["SiteY"],
            "wavelength": tiff.metaseries_metadata["PlaneInfo"]["wavelength"],
        }

    return data, metadata
