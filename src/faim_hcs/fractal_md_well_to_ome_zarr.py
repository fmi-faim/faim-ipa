# Convert a well of MD Image Xpress data to OME-Zarr
from faim_hcs.io.MolecularDevicesImageXpress import parse_files, parse_single_plane_multi_fields
from faim_hcs.Zarr import build_zarr_scaffold, write_czyx_image_to_well, write_cyx_image_to_well, write_roi_table
from faim_hcs.MetaSeriesUtils import get_well_image_CYX, get_well_image_CZYX, montage_grid_image_YX
from faim_hcs.UIntHistogram import UIntHistogram
from tqdm.notebook import tqdm
from os.path import join, exists

import zarr

is_2D = True
# TODO: Is the zarr name part of the path in Fractal? Likely not
input_paths = ['/Users/joel/Dropbox/Joel/FMI/Code/faim-hcs/examples/zarr-files/Single-Plane.zarr']

# Parameters
# TODO: Get this from prior task => from metadata
images_path = '/Users/joel/Dropbox/Joel/FMI/Code/faim-hcs/resources/Projection-Mix'

# TODO: How to get the wells? Normally passed as `component`. 
# But here should be actual well name, as well as the plate object

# TODO: How do we get the channel names?
channels = ['w1', 'w2', 'w3', 'w4']
# Add image data to wells
well = 'E07'

# 2D flow
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
    well_group = plate[well[0]][str(int(well[1:]))][0] # This is basically the component, but in 2 parts
    write_cyx_image_to_well(img, hists, ch_metadata, metadata, well_group)

    # Write all ROI tables
    for roi_table in roi_tables:
        write_roi_table(roi_tables[roi_table], roi_table, well_group)

else:
    raise NotImplementedError("3D flow not implemented yet")
    # files = parse_files(images_path)
    # well_files = files[files['well'] == well]
#     # TODO: What is the plate object and do I need it? It's the Zarr group for the plate?
#     well_group = plate[well[0]][str(int(well[1:]))][0]
#     write_cyx_image_to_well(img, hists, ch_metadata, metadta, well_group)

    
#     projection_files = well_files[well_files['z'].isnull()]
    
#     # This also contains the single z-plane for channel 3
#     stack_files = well_files[~well_files['z'].isnull()]
    
    
#     projection, proj_hists, proj_ch_metadata, proj_metadata = get_well_image_CYX(
#         well_files=projection_files,
#         channels=channels,
#         assemble_fn=montage_grid_image_YX,
#     )
    
#     stack, stack_hist, stack_ch_metadata, stack_metadata = get_well_image_CZYX(
#         well_files=stack_files,
#         channels=channels,
#         assemble_fn=montage_grid_image_YX,
#     )
    
#     # Get the zeroth field of the well.
#     field: zarr.Group = plate[well[0]][str(int(well[1:]))][0]
        
#     # Write CZYX stack into zeroth field.
#     write_czyx_image_to_well(stack, stack_hist, stack_ch_metadata, stack_metadata, field, True)
        
#     # Create projections group 
#     projections = field.create_group("projections")
    
#     # Write projections
#     write_cyx_image_to_well(projection, proj_hists, proj_ch_metadata, proj_metadata, projections, True)