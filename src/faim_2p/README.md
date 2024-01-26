## OME-Tiff Tools

A collection of tools to handle OME-Tiff files.
Implemented so far:

####  OMETiffConcat ([OMETiffTools.py](OmeTiffTools.py)):

Concatenates a multi-file OME-Tiff into a single OME-Tiff file in BigTiff format.
The multi-file OME-Tiff is expected to follow the 
[OME specifications](https://ome-model.readthedocs.io/en/stable/ome-tiff/specification.html).
The `FileName` attribute of `UUID` elements is mandatory.
Multi-file OME-Tiffs consist of one file with a complete OME_XML header (the masterfile)
and secondary files with partial OME-XML metadata (called binary only files) .
The only input argument is the path to the masterfile; 
the binary only files must reside in the same directory as the master file.

Usage:

    import OMETiffTools
    otc = OMETiffTools.OMETiffConcat('/path/to/masterfile.tif')
    otc.writebigtiff('/path/to/other_directory/bigtiff.tf8', progress="tqdm")