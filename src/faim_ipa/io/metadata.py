from pydantic import BaseModel, NonNegativeInt, PositiveFloat


class ChannelMetadata(BaseModel):
    channel_index: NonNegativeInt
    channel_name: str
    display_color: str
    spatial_calibration_x: PositiveFloat
    spatial_calibration_y: PositiveFloat
    spatial_calibration_units: str
    z_spacing: PositiveFloat | None = None
    wavelength: NonNegativeInt | None = None
    exposure_time: PositiveFloat | None = None
    exposure_time_unit: str | None = None
    objective: str
