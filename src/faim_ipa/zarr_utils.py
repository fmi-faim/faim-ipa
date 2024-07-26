from typing import Any

from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_multiscales_metadata
from zarr import Group


def _copy_multiscales_metadata(parent_group, subgroup):
    datasets = parent_group.attrs.asdict()["multiscales"][0]["datasets"]
    axes = parent_group.attrs.asdict()["multiscales"][0]["axes"]
    write_multiscales_metadata(subgroup, datasets=datasets, axes=axes)


def write_labels_to_group(
    labels,
    labels_name,
    parent_group: Group,
    *,
    storage_options: dict[str, Any],
    max_layer: int = 0,
    overwrite: bool = False,
):
    """
    Write labels to zarr group and copy multiscales metadata from parent group.
    """
    subgroup = parent_group.require_group(
        f"labels/{labels_name}",
        overwrite=overwrite,
    )

    axes = parent_group.attrs.asdict()["multiscales"][0]["axes"]
    if len(axes) == len(labels.shape):
        message = f"Group axes don't match label image dimensions: {len(axes)} <> {len(labels.shape)}."
        raise ValueError(message)

    write_image(
        image=labels,
        group=subgroup,
        axes=axes,
        storage_options=storage_options,
        scaler=Scaler(max_layer=max_layer),
    )

    _copy_multiscales_metadata(parent_group, subgroup)
