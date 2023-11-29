from typing import Optional, Union

from pydantic import BaseModel, NonNegativeInt, PositiveFloat, PositiveInt


class TileMetadata(BaseModel):
    tile_size_x: PositiveInt
    tile_size_y: PositiveInt


class ChannelMetadata(BaseModel):
    channel_index: NonNegativeInt
    channel_name: str
    display_color: str
    spatial_calibration_x: float
    spatial_calibration_y: float
    spatial_calibration_units: str
    z_scaling: Optional[PositiveFloat]
    wavelength: Union[PositiveInt, str]
    exposure_time: PositiveFloat
    exposure_time_unit: str
    objective: str
