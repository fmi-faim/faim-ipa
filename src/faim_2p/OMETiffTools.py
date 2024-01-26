#! /usr/bin/env python

import tifffile
import lxml.etree as le
from os.path import isfile, dirname, basename, join
from tqdm import tqdm

def get_datafile_uuids(inxml: str) -> dict:
    """
    Input: XML string from OME-Tiff header (master file).
    Extract UUIDs of the master file and the parameters
    (uuid, filename, IFD) of each BinData file from an OME-Tiff XML
    header (masterfile of multi-file OME-Tiff record).

    The output is a dictionary with the keys "master", which contains
    the uuid of the master file, and "bindata", which contains a list
    of dictionaries with the bindata file parameters.
    """
    root = le.fromstring(inxml.encode('utf-8'))

    out = {}

    assert 'UUID' in root.attrib, "no UUID found in XML root"

    out["master"] = {"uuid": root.attrib["UUID"]}

    out["bindata"] = []

    for elem in root.iter():
        if elem.tag.endswith('UUID'):
            parent = elem.getparent()
            assert elem.text.startswith('urn:uuid'), "no uuid in " + str(elem.text)
            assert "FileName" in elem.attrib.keys(), " no FileName in" + str(elem.attrib)
            assert "IFD" in parent.attrib.keys(), "no IFD in " + str(parent.attrib)

            out["bindata"].append({"uuid": elem.text,
                                   "filename": elem.attrib["FileName"],
                                   "IFD": parent.attrib["IFD"]})

    return out

# update image file directory (ifd) of each entry, and drop UUID entries with filenames
def update_xml_header(inxml: str) -> str:
    """
    Input: XML header from the OME-Tiff master file of a record stored across multiple  OME-Tiff files.
    Output: XML header for a single OME-Tiff file that contains all the data.
    In the output XML string, the IFD (image file directory) values in <TiffData> elements are updated,
    and the <UUID> subelements with the binary data filename are dropped.
    """

    root = le.fromstring(inxml.encode('utf-8'))

    ifd_counter = 0

    for elem in root.iter():
        if elem.tag.endswith('TiffData'):
            assert "IFD" in elem.attrib.keys(), "no IFD entry found in " + elem.tag
            elem.attrib["IFD"] = str(ifd_counter)
            ifd_counter += 1

        if elem.tag.endswith('UUID'):
            parent = elem.getparent()
            parent.remove(elem)

    outxml = le.tostring(root, encoding='utf-8').decode('utf-8')
    return outxml


def get_bindata_uuid(inxml: str) -> dict:
    """
    Input: XML string from OME-Tiff file with partial metadata (of multi-file sequence).
    Output: UUID of the binary only file, and the UUID and filename of the master file
    (in dictionary).
    """

    root = le.fromstring(inxml.encode('utf-8'))

    assert 'UUID' in root.attrib, "no UUID found in XML root"

    out = {"uuid": root.attrib["UUID"]}

    for elem in root.iter():
        if elem.tag.endswith("BinaryOnly"):
            assert "UUID" in elem.attrib.keys(), "no UUID in " + str(elem.attrib)
            assert "MetadataFile" in elem.attrib.keys(), "no Metadatafile in " + str(elem.attrib)
            out["MetadataFile_uuid"] = elem.attrib["UUID"]
            out["MetadataFile"] = elem.attrib["MetadataFile"]

    return out

class OMETiffConcat:
    """
    Creates a single, OME-BigTiff file from a multi-file OME-Tiff record.
    The masterfile is the name of the OME-Tiff file that contains the full XML header.
    """

    def __init__(self, masterfile: str):

        assert isfile(masterfile), f"{masterfile} does not exist or is not a file."

        self.filepath = dirname(masterfile)
        self.masterfile = basename(masterfile)

        with tifffile.TiffFile(masterfile) as tiffmaster:
            self.omexml = tiffmaster.pages[0].description

        self.datafiles = get_datafile_uuids(self.omexml)

        self.nfiles = len(self.datafiles["bindata"])

        assert self.nfiles > 1, "input file must be multi-file OME-Tiff."

    def get_abspath(self, filename):
        return join(self.filepath, filename)

    def write_bigtiff(self, outputfile: str = None, compression: str = "zstd", progress: str = "tqdm"):
        """
        Writes single OME BigTiff file.
        Outputfile is the absolute path of the BigTiff file
        Compression default is `zstd`.  Use None for no compression.
        Progress: tqdm (
        """

        assert not isfile(outputfile), f"{outputfile} already exists."
        # XML header for BigTiff file
        btxml = update_xml_header(self.omexml)

        with tifffile.TiffWriter(outputfile, bigtiff=True, ome=False) as tiff_writer:

            tqdm_fileinfo = tqdm(self.datafiles["bindata"]) if progress == "tqdm" else self.datafiles["bindata"]

            # for n, fileinfo in enumerate(self.datafiles["bindata"]):
            for n, fileinfo in enumerate(tqdm_fileinfo):

                secfile = self.get_abspath(fileinfo["filename"])

                if progress == "tqdm" and (n % 100) == 0:
                    tqdm_fileinfo.set_postfix({"file": fileinfo["filename"]})
                elif progress == "plain":
                    print(f"\rReading {secfile} ...", end="", flush=True)

                assert isfile(secfile), f"{secfile} does not exist or is not a file"

                with tifffile.TiffFile(secfile) as tiffx:
                    secinfo = get_bindata_uuid(tiffx.pages[0].description)

                    assert secinfo["uuid"] == fileinfo["uuid"], "inconsistent UUID of BinaryOnly file"

                    # if this condition is not met, the file being read is the master file
                    if secinfo["uuid"] != self.datafiles["master"]["uuid"]:
                        assert secinfo["MetadataFile_uuid"] == self.datafiles["master"]["uuid"]
                        assert secinfo["MetadataFile"] == self.masterfile

                    ifd = int(fileinfo["IFD"])

                    tiff_writer.save(
                        tiffx.pages[ifd].asarray(),
                        # photometric='minisblack' # this could be parsed from the image tags
                        compression=compression,
                        description=btxml if n == 0 else None,
                        contiguous=False
                    )

            if progress == "plain":
                print("  done.")

def main():
    inputfile = "/tungstenfs/scratch/gluthi/odstiris/data/2P/A_1172118/Session1/Imaging/TSeries-09112023-1821-000/TSeries-09112023-1821-000_Cycle00001_Ch2_000001.ome.tif"
    outputfile = "/tungstenfs/scratch/gscicomp_share/gluthi/merged_timeseries_v3.tf8"

    otc=OMETiffConcat(inputfile)

    print(f"Total number of OME-Tiff files: {otc.nfiles}")

    otc.write_bigtiff(outputfile, progress="plain")

if __name__ == "__main__":
    main()
