# FAIM-HCS
A collection of functions we use at [FAIM](https://www.fmi.ch/platforms/platform.html?plt=110) to handle HCS (high-content screening) data, which is stored according to the [NGFF](https://ngff.openmicroscopy.org/latest/) spec in OME-Zarr with additional metadata fields.

## Functionality
* Convert Molecular Devices ImageXpress acquisitions to ome-zarr.
* Create [MoBIE - Projects](https://mobie.github.io/) for created ome-zarr plates.

## Extra Metadata in OME-Zarr
`{plate_name}/.zattrs`:
* `barcode`: The barcode of the imaged plate
* `order_name`: Name of the plate order

`{plate_name}/{row}/{col}/0/.zattrs`:
* `acquisition_metadata`: A dictionary with key channels.
    * `channels`: A list of dicitionaries for each acquired channel, with the following keys:
        * `channel-name`: Name of the channel during acquisition
        * `display-color`: RGB hex-code of the display color
        * `exposure-time`
        * `exposure-time-unit`
        * `objective`: Objective description
        * `objective-numerical-aperture`
        * `power`: Illumination power used for this channel
        * `shading-correction`: Set to On if a shading correction was applied automatically.
        * `wavelength`: Name of the wavelength as provided by the microscope.
    * `histograms`: A list of relative paths to the histograms of each channel.

## Histograms
We use a custom [`UIntHistogram`](src/UIntHistogram.py) class to generate and save histograms of the individual wells.
The histograms can be aggregated with the `combine()` method, which allows us to quickly compute `mean()`, `std()`, `quantile()`, `min()` and `max()` over the whole plate or any subset of wells.

## Installation
Create a new conda environment from the `environment.yaml` with `conda env create -f envrionment.yaml`.

If you want to run the examples you must install `jupyter` into the activated environment.

## Examples
* [Create ome-zarr from Single-Plane Multi-Field Acquisition](examples/Create%20ome-zarr%20from%20Single-Plane%20Multi-Field%20Acquisition.ipynb)
* [Create ome-zarr from Z-Stack Multi-Field Acquisition](examples/Create%20ome-zarr%20from%20Z-Stack%20Multi-Field%20Acquisition.ipynb)
* [Create MoBIE Project](examples/Create%20MoBIE%20Project.ipynb)
* [Inspect Zarr](examples/Inspect%20Zarr.ipynb)
