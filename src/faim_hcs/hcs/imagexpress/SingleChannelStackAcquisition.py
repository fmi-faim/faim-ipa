import re

import pandas as pd

from faim_hcs.hcs.imagexpress import ImageXpressPlateAcquisition, StackAcquisition


class SingleChannelStackAcquisition(StackAcquisition):
    """Single channel image stack acquisition with a Molecular Devices
    ImageXpress Micro Confocal system.

    MIP-2P-2sub-Stack --> {name} [Optional]
    └── 2023-02-21 --> {date}
        └── 1334 --> {acquisition id}
            ├── ZStep_1
            │   ├── 240124d3_B05_s7ABE2661D-258C-46DB-B4EF-2F7F18F288E1.tif
            │   ├── 240124d3_B05_s7_thumb521FDCDC-3EA9-468F-9E74-772F496E33F7.tif
            │   ├── ...
            │   └── 240124d3_D05_s22129693C3-8269-49C0-A4AC-40459ADE6D2D.tif
            ├── ...
            └── ZStep_9
                ├── 240124d3_B05_s1C3DE8036-1874-4A98-9AC0-9A99D9C0D55B.tif
                ├── ...
                └── 240124d3_D05_s2265838367-D078-42FB-B050-F3A28BC4AA1E.tif

    Image data is stored in {name}_{well}_{field}_{md_id}.tif.
    The *_thumb*.tif files, used by Molecular Devices as preview, are ignored.
    """

    def _get_filename_re(self) -> re.Pattern:
        return re.compile(
            r"(?P<name>.*)_(?P<well>[A-Z]+\d{2})_(?P<field>s\d+)(?!_thumb)(?P<md_id>.*)(?P<ext>.tif)"
        )

    def _parse_files(self) -> pd.DataFrame:
        files = pd.DataFrame(
            ImageXpressPlateAcquisition._list_and_match_files(
                root_dir=self._acquisition_dir,
                root_re=self._get_root_re(),
                filename_re=self._get_filename_re(),
            )
        )
        files["channel"] = "w1"
        self._z_spacing = self._compute_z_spacing(files)
        return files
