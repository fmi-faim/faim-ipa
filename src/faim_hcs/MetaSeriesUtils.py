import numpy as np

from faim_hcs.UIntHistogram import UIntHistogram
from faim_hcs.utils import rgb_to_hex, wavelength_to_rgb


def build_omero_channel_metadata(
    metaseries_ch_metadata: dict, dtype: type, histograms: list[UIntHistogram]
):
    """Build omero conform channel metadata to be stored in zarr attributes.

    * Color is computed from the metaseries wavelength metadata.
    * Label is the set to the metaseries _IllumSetting_ metadata.
    * Intensity scaling is obtained from the data histogram [0.01,
    0.99] quantiles.

    :param metaseries_ch_metadata: channel metadata from tiff-tags
    :param dtype: data type
    :param histograms: histograms of channels
    :return: omero metadata dictionary
    """
    channels = []
    for i, (ch, hist) in enumerate(zip(metaseries_ch_metadata, histograms)):
        channels.append(
            {
                "active": True,
                "coefficient": 1,
                "color": rgb_to_hex(*wavelength_to_rgb(ch["wavelength"])),
                "family": "linear",
                "inverted": False,
                "label": ch["_IllumSetting_"],
                "wavelength_id": f"C{str(i).zfill(2)}",
                "window": {
                    "min": np.iinfo(dtype).min,
                    "max": np.iinfo(dtype).max,
                    "start": hist.quantile(0.01),
                    "end": hist.quantile(0.99),
                },
            }
        )

    return {"channels": channels}
