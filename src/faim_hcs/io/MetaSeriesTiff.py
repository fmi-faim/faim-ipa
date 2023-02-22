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
        plane_info = tiff.metaseries_metadata["PlaneInfo"]
        metadata = {
            "_IllumSetting_": plane_info["_IllumSetting_"],
            "spatial-calibration-x": plane_info["spatial-calibration-x"],
            "spatial-calibration-y": plane_info["spatial-calibration-y"],
            "spatial-calibration-units": plane_info["spatial-calibration-units"],
            "stage-position-x": plane_info["stage-position-x"],
            "stage-position-y": plane_info["stage-position-y"],
            "z-position": plane_info["z-position"],
            "PixelType": str(data.dtype),
            "_MagNA_": plane_info["_MagNA_"],
            "_MagSetting_": plane_info["_MagSetting_"],
            "Exposure Time": plane_info["Exposure Time"],
            "Lumencor Cyan Intensity": plane_info["Lumencor Cyan Intensity"],
            "Lumencor Green Intensity": plane_info["Lumencor Green Intensity"],
            "Lumencor Red Intensity": plane_info["Lumencor Red Intensity"],
            "Lumencor Violet Intensity": plane_info["Lumencor Violet Intensity"],
            "Lumencor Yellow Intensity": plane_info["Lumencor Yellow Intensity"],
            "ShadingCorrection": plane_info["ShadingCorrection"],
            "stage-label": plane_info["stage-label"],
            "SiteX": plane_info["SiteX"],
            "SiteY": plane_info["SiteY"],
            "wavelength": plane_info["wavelength"],
        }
        if "Z Step" in plane_info:
            metadata["Z Step"] = plane_info["Z Step"]
        if "Z Projection Method" in plane_info:
            metadata["Z Projection Method"] = plane_info["Z Projection Method"]

    return data, metadata
