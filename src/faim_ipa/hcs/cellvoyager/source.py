from io import BytesIO
from pathlib import Path

import h5py
from numpy._typing import NDArray
from tifffile import imread
import blosc2

from faim_ipa.hcs.source import Source


class CVSource(Source):

    def get_measurement_detail(self) -> BytesIO:
        raise NotImplementedError()

    def get_measurement_data(self) -> BytesIO:
        raise NotImplementedError()

    def get_measurement_settings(self, file_name: str) -> BytesIO:
        raise NotImplementedError()

    def get_trace_logs(self) -> [str]:
        raise NotImplementedError()


class CVSourceFS(CVSource):
    def __init__(
        self,
        acquisition_dir: Path | str,
        trace_log_files: None | list[Path | str] = None,
    ):
        self._acquisition_dir = Path(acquisition_dir)
        if trace_log_files:
            self._trace_log_files = [Path(f) for f in trace_log_files]
        else:
            self._trace_log_files = []

    def get_measurement_detail(self) -> BytesIO:
        mrf_file = self._acquisition_dir / "MeasurementDetail.mrf"
        if not mrf_file.exists():
            msg = "MeasurementDetail.mrf not found"
            raise FileNotFoundError(msg)

        with open(mrf_file, "rb") as fb:
            io_buffer = BytesIO(fb.read())

        return io_buffer

    def get_measurement_data(self):
        mlf_file = self._acquisition_dir / "MeasurementData.mlf"
        if not mlf_file.exists():
            msg = "MeasurementData.mlf not found"
            raise FileNotFoundError(msg)

        with open(mlf_file, "rb") as fb:
            io_buffer = BytesIO(fb.read())

        return io_buffer

    def get_measurement_settings(self, file_name: str) -> BytesIO:
        settings_file = self._acquisition_dir / file_name
        if not settings_file.exists():
            msg = f"Settings file not found: {settings_file}"
            raise FileNotFoundError(msg)

        with open(settings_file, "rb") as fb:
            io_buffer = BytesIO(fb.read())

        return io_buffer

    def get_trace_logs(self) -> [str]:
        for log_file in self._trace_log_files:
            log = []
            with open(log_file) as f:
                for line in f:
                    log.append(line)
            yield log

    def get_image(self, file_name: str) -> NDArray:
        absolute_path = self._acquisition_dir / file_name
        return imread(absolute_path)

    def exists(self, file_name: str) -> bool:
        return (self._acquisition_dir / file_name).exists()


class CVSourceHDF5(CVSource):
    def __init__(self, hdf5: Path | str | list, name: str):
        # hdf5 is either the name of an HDF5 archive or a list of file (multi-part HDF5 archive)
        if isinstance(hdf5, (Path, str)):
            self.hdf5_path = Path(hdf5)
            self.multipart = False

            # verify the single file given is not a multi-part file
        else:
            raise NotImplementedError('multi-part HDF5 archives are not yet implemented')
            
        self.name = name

    def get_measurement_detail(self) -> BytesIO:
        with h5py.File(self.hdf5_path, "r") as data:
            buffer_compressed = data[self.name]["MeasurementDetail.mrf.blosc2"][
                :
            ].tobytes()
            buffer = blosc2.decompress2(buffer_compressed)
            return BytesIO(buffer)

    def get_measurement_data(self) -> BytesIO:
        with h5py.File(self.hdf5_path, "r") as data:
            buffer_compressed = data[self.name]["MeasurementData.mlf.blosc2"][
                :
            ].tobytes()
            buffer = blosc2.decompress2(buffer_compressed)
            return BytesIO(buffer)

    def get_measurement_settings(self, file_name: str) -> BytesIO:
        with h5py.File(self.hdf5_path, "r") as data:
            buffer_compressed = data[self.name][f"{file_name}.blosc2"][:].tobytes()
            buffer = blosc2.decompress2(buffer_compressed)
            return BytesIO(buffer)

    def get_trace_logs(self) -> [str]:
        with h5py.File(self.hdf5_path, "r") as data:
            for k in data["LOG"].keys():
                buffer_compressed = data["LOG"][k][:].tobytes()
                buffer = blosc2.decompress2(buffer_compressed)
                log = []
                for line in BytesIO(buffer):
                    log.append(line.decode("utf-8"))
                yield log

    def get_image(self, file_name: str) -> NDArray:
        with h5py.File(self.hdf5_path, "r") as data:
            buffer_compressed = data[self.name][f"{file_name}.blosc2"][:].tobytes()
            buffer = blosc2.decompress2(buffer_compressed)
            return imread(BytesIO(buffer))

    def exists(self, file_name: str) -> bool:
        with h5py.File(self.hdf5_path, "r") as data:
            return f"{file_name}.blosc2" in data[self.name]
