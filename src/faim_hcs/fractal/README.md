# Using Fractal task

Fractal comes with extra dependencies. We're currently working on making the fractal-tasks-core package much more lightweight by default, so that one does need to depend on all its task dependencies. All Fractal dependencies are anyway optional. Thus, to install them, one needs to specify them:
```
cd faim-hcs
pip install ".[fractal-tasks]"
```

Or, once this package is on pypi:
```
pip install "faim-hcs[fractal-tasks]"
```

To collect the task in Fractal, one can load the Python wheel (see instructions below):
```
fractal task collect /path/to/faim-hcs-version-details.whl --package-extras fractal-tasks
```

After that, the two tasks become available in Fractal.

### Developer info
To create a new Fractal task, one needs to create a linux executable (e.g. a Python file with a `if __name__ == "__main__":` section) and this executable needs to follow the Fractal standards on how to read in inputs & store outputs ([see details here](https://fractal-analytics-platform.github.io/fractal-tasks-core/task_howto.html)). A typical Fractal task uses pydantic for input validation.

To make the task installable by a Fractal server, there needs to be a `__FRACTAL_MANIFEST__.json` file in the src/faim_hcs folder. This file contains a list of all available tasks, their default parameters as well as the relative path to the executable.

The manifest needs to be included when a package is built.

For local creation of a Python whl, it means that the setup.cfg contains the following:
```
[options.package_data]
faim_hcs = __FRACTAL_MANIFEST__.json
```
Afterwards, the wheel can be built using `python -m build` and collected by Fractal using the command line client:
(you may need to `pip install build` to run the wheel creation)

```
fractal task collect /path/to/faim-hcs-version-details.whl --package-extras fractal-tasks
```

If the fractal tasks are in their own module (i.e their own folder), this folder needs to contain an (empty) `__init__.py` file.


### Working with a Fractal task in development
The above instructions work well to install the Fractal task as it is available in the package. If you want to run a task through Fractal server that you keep changing, it's not advisable to use the fractal task collection, but instead manually register your task.

For that purpose, create a Python environment that the task runs in (with all dependencies installed) and then use manual task registration pointing to the task Python file that you're working with. [See here for an example](https://github.com/fractal-analytics-platform/fractal-demos/tree/d241c7e29e5016bca6e0fd7647f44947e1501509/examples/08_scMultipleX_task).

For example, add this via the web interface:
```
command: /path/to/python-env/faim-hcs-dev/bin/python /path/to/faim-hcs/src/faim_hcs/fractal/fractal_create_ome_zarr_md.py
source: joel:create-ome-zarr-md:0.0.1
image
zarr
```

Then set the default args correctly for the task:
```
fractal task edit 9 --meta-file /path/to/faim-hcs/examples/fractal/meta_create_ome_zarr_md.json
```
