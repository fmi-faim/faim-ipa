# Convert a well of MD Image Xpress data to OME-Zarr
from faim_hcs.io.MolecularDevicesImageXpress import parse_files, parse_single_plane_multi_fields
from faim_hcs.Zarr import build_zarr_scaffold, write_czyx_image_to_well, write_cyx_image_to_well, write_roi_table
from faim_hcs.MetaSeriesUtils import get_well_image_CYX, get_well_image_CZYX, montage_grid_image_YX
from faim_hcs.UIntHistogram import UIntHistogram
from tqdm.notebook import tqdm
from os.path import join, exists

import zarr

is_2D = True
# FIXME: Is the zarr name part of the path in Fractal? No => get it correctly from metadata
input_paths = ['/Users/joel/Dropbox/Joel/FMI/Code/faim-hcs/examples/zarr-files/Single-Plane.zarr']

is_2D = False
input_paths = ['/Users/joel/Dropbox/Joel/FMI/Code/faim-hcs/examples/zarr-files/Projection-Mix.zarr']

# Mode can be 3 values: "z-steps" (only parse the 3D data), "top-level" (only parse the 2D data), "all" (parse both)
mode = "z-steps"
mode = "top-level"
mode = "all"

# Parameters
# TODO: Get this from prior task => from metadata
images_path = '/Users/joel/Dropbox/Joel/FMI/Code/faim-hcs/resources/Projection-Mix'

# TODO: How to get the wells? Normally passed as `component`. 
# But here should be actual well name, as well as the plate object

# TODO: How do we get the channel names?
channels = ['w1', 'w2', 'w3', 'w4']
# Add image data to wells
well = 'E07'

valid_modes = ("z-steps", "top-level", "all")
if mode not in valid_modes:
    raise NotImplementedError(
        f"Only implemented for modes {valid_modes}, but got mode {mode=}"
    )

# TODO: Only open the image in the well, not the whole plate to avoid concurrency issues?
# Can probably build this from input_path + component?
plate = zarr.open(input_paths[0], mode='r+')
if is_2D:
    # TODO: Can I also go via parse_files for this? Does it work for real 2D?
    files = parse_single_plane_multi_fields(images_path)
    well_files = files[files['well'] == well]

    img, hists, ch_metadata, metadata, roi_tables = get_well_image_CYX(
        well_files=well_files,
        channels=channels,
        assemble_fn=montage_grid_image_YX,
    )
    # well_group = plate[well[0]][str(int(well[1:]))][0] # This is basically the component, but in 2 parts
    # Get the zeroth field of the well.
    field: zarr.Group = plate[well[0]][str(int(well[1:]))][0]
    write_cyx_image_to_well(img, hists, ch_metadata, metadata, field)

    # Write all ROI tables
    for roi_table in roi_tables:
        write_roi_table(roi_tables[roi_table], roi_table, field)

else:
    files = parse_files(images_path)
    well_files = files[files['well'] == well]
    
    projection_files = well_files[well_files['z'].isnull()]
    
    # This also contains the single z-plane for channel 3
    stack_files = well_files[~well_files['z'].isnull()]
    
    projection, proj_hists, proj_ch_metadata, proj_metadata, proj_roi_tables = get_well_image_CYX(
        well_files=projection_files,
        channels=channels,
        assemble_fn=montage_grid_image_YX,
    )
    # Should we also write proj_roi_tables? Unclear whether we should have 
    # separate ROI tables for projections
    
    stack, stack_hist, stack_ch_metadata, stack_metadata, roi_tables = get_well_image_CZYX(
        well_files=stack_files,
        channels=channels,
        assemble_fn=montage_grid_image_YX,
    )
    
    # Get the zeroth field of the well.
    field: zarr.Group = plate[well[0]][str(int(well[1:]))][0]
        
    # Write CZYX stack into zeroth field.
    write_czyx_image_to_well(stack, stack_hist, stack_ch_metadata, stack_metadata, field, True)
        
    # Create projections group 
    projections = field.create_group("projections")
    
    # Write projections
    write_cyx_image_to_well(projection, proj_hists, proj_ch_metadata, proj_metadata, projections, True)

    # Write all ROI tables for the 3D stack
    for roi_table in roi_tables:
        write_roi_table(roi_tables[roi_table], roi_table, field)