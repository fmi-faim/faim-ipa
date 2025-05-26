# INSTALLATATION

The hdf5-to-zarr branch can be installed directly using [Pixi](https://pixi.sh/latest/):

    pixi init faimipa
    cd faimipa
    pixi add python=3.12 pip
    pixi run pip install "git+https://github.com/fmi-faim/faim-ipa@hdf5-to-zarr"