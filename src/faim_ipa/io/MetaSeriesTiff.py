from pathlib import Path

from tifffile import tifffile
from numpy._typing import ArrayLike


def load_imagexpress_metadata(path: Path) -> tuple[ArrayLike, dict]:
    """Load parts of the metadata of a imagexpress tiff file.

    The following metadata is collected:
    * pixel-size-x
    * pixel-size-y
    * _IllumSetting_
    * spatial-calibration-x
    * spatial-calibration-y
    * spatial-calibration-units
    * ImageXpress Micro X
    * ImageXpress Micro Y
    * ImageXpress Micro Z
    * z-position
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
        assert (
            tiff.is_metaseries or tiff.is_stk or tiff.is_shaped
        ), f"{path} is not a metamorph, STK file, or shaped file"

        if tiff.is_metaseries:
            selected_keys = [
                "pixel-size-x",
                "pixel-size-y",
                "_IllumSetting_",
                "spatial-calibration-x",
                "spatial-calibration-y",
                "spatial-calibration-units",
                "ImageXpress Micro X",
                "ImageXpress Micro Y",
                "ImageXpress Micro Z",
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
            metadata["stage-position-x"] = metadata["ImageXpress Micro X"]
            metadata["stage-position-y"] = metadata["ImageXpress Micro Y"]
            metadata["stage-position-z"] = metadata["ImageXpress Micro Z"]

            metadata.pop("ImageXpress Micro X")
            metadata.pop("ImageXpress Micro Y")
            metadata.pop("ImageXpress Micro Z")
        else:
            metadata = load_alternative_imagexpress_metadata(path)

    return metadata


def parse_stk_plane_description(tiff: tifffile.TiffFile) -> dict:
    """Parse plane description of legacy STK files, which are newline-delimted key-value pairs."""
    assert "PlaneDescriptions" in tiff.stk_metadata, "No PlaneDescriptions found."
    parsed_metadata = {}
    plane_descriptions = tiff.stk_metadata["PlaneDescriptions"]
    if isinstance(plane_descriptions, list):
        plane_descriptions = plane_descriptions[0].splitlines()
    for line in plane_descriptions:
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":")
            value = value.strip()
            parsed_metadata[key.strip()] = value

    return parsed_metadata


def load_alternative_imagexpress_metadata(path: Path) -> tuple[ArrayLike, dict]:
    """Load parts of the metadata of a non-metaseries tiff file.

    The following metadata is collected:
    * pixel-size-x
    * pixel-size-y
    * _IllumSetting_
    * spatial-calibration-x
    * spatial-calibration-y
    * spatial-calibration-units
    * ImageXpress Micro X
    * ImageXpress Micro Y
    * ImageXpress Micro Z
    * z-position
    * _MagNA_
    * _MagSetting_
    * Exposure Time
    * ShadingCorrection
    * stage-label

    :param path:
    :return:
    image_data, metadata-dict
    """
    with tifffile.TiffFile(path) as tiff:
        assert tiff.is_stk or tiff.is_shaped, f"{path} is not an STK or shaped file."

        selected_keys = [
            "Name",  # remap to _IllumSetting_
            "XCalibration",  # remap tp spatial-calibration-x
            "YCalibration",  # remap to spatial-calibration-y
            "CalibrationUnits",  # remap to spatial-calibration-units
            "_MagNA_",
            "_MagSetting_",
            "Exposure Time",
            "ShadingCorrection",
            "stage-label",
            "wavelength",
        ]
        convert_key_map = {
            "Name": "_IllumSetting_",
            "XCalibration": "spatial-calibration-x",
            "YCalibration": "spatial-calibration-y",
            "CalibrationUnits": "spatial-calibration-units",
            "Exposure": "Exposure Time",
            "Shading": "ShadingCorrection",
        }
        plane_info = tiff.stk_metadata if tiff.is_stk else tiff.shaped_metadata

        metadata = {
            k: plane_info[k]
            for k in plane_info
            if k in selected_keys or k.endswith("Intensity")
        }

        if plane_info["StagePosition"].shape == (1, 2):
            (
                metadata["stage-position-x"],
                metadata["stage-position-y"],
            ) = plane_info[
                "StagePosition"
            ][0]
        else:
            raise NotImplementedError("Only non-zstack STK files are supported.")

        pixel_size_x, pixel_size_y = tiff.pages[0].shape
        metadata["pixel-size-x"] = pixel_size_x
        metadata["pixel-size-y"] = pixel_size_y
        metadata["stage-position-z"] = 1

        plate_description = parse_stk_plane_description(tiff)
        metadata.update(plate_description)

        for k, v in convert_key_map.items():
            metadata[v] = metadata.pop(k)

        if "_MagSetting_" not in metadata:
            metadata["_MagSetting_"] = "N/A"
    return metadata


def load_imagexpress_tiff(path: Path) -> tuple[ArrayLike, dict]:
    with tifffile.TiffFile(path) as tiff:
        data = tiff.asarray()
    metadata = load_imagexpress_metadata(path=path)
    metadata["PixelType"] = str(data.dtype)
    return data, metadata


# Avoid breaking changes
load_metaseries_tiff = load_imagexpress_tiff
