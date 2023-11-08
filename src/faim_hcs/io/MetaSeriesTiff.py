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
    * z-position
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
    * Z Step (if existent)
    * Z Projection Method (if existent)

    :param path:
    :return:
    image_data, metadata-dict
    """
    with tifffile.TiffFile(path) as tiff:
        assert tiff.is_metaseries, f"{path} is not a metamorph file."
        data = tiff.asarray()
        selected_keys = [
            "_IllumSetting_",
            "spatial-calibration-x",
            "spatial-calibration-y",
            "spatial-calibration-units",
            "stage-position-x",
            "stage-position-y",
            "z-position",
            "_MagNA_",
            "_MagSetting_",
            "Exposure Time",
            "ShadingCorrection",
            "stage-label",
            "SiteX",
            "SiteY",
            "wavelength",
            "Z Step",  # optional
            "Z Projection Method",  # optional
            "Z Projection Step Size",  # optional
        ]
        plane_info = tiff.metaseries_metadata["PlaneInfo"]
        metadata = {
            k: plane_info[k]
            for k in plane_info
            if k in selected_keys or k.endswith("Intensity")
        }
        metadata["PixelType"] = str(data.dtype)

    return data, metadata
