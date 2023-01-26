import unittest
from os.path import exists, join
from pathlib import Path

from faim_hcs.io.MolecularDevicesImageXpress import parse_single_plane_multi_fields

ROOT_DIR = Path(__file__).parent.parent


class TestMolecularDevicesImageXpress(unittest.TestCase):
    def test_parse_single_plane_multi_fields(self):
        acquisition_dir = join(ROOT_DIR.parent, "resources", "MIP-2P-2sub")

        files = parse_single_plane_multi_fields(acquisition_dir=acquisition_dir)

        assert len(files) == 8
        assert files["name"].unique() == ["MIP-2P-2sub"]
        assert all(files["well"].unique() == ["C05", "C06"])
        assert all(files["field"].unique() == ["s1", "s2"])
        channels = files["channel"].unique()
        assert len(channels) == 2
        assert "w1" in channels
        assert "w2" in channels
        for item in files["path"]:
            assert exists(item)
            assert "thumb" not in item
