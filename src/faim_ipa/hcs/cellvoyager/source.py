from io import BytesIO
from pathlib import Path

import h5py
from numpy._typing import NDArray
from tifffile import imread
import blosc2
from os.path import join

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
            with h5py.File(self.hdf5_path, "r") as data:
                if "hdf5_number_of_archive_files" in data.attrs.keys():
                    assert data.attrs["hdf5_number_of_archive_files"] == 1, f"{hdf5} belongs to a multi-part HDF5Vault archive, but only one was given."
                    
                self.compression=data.attrs["compression"]

        else:
            self.num_archives = len(hdf5)
            self.multipart = True

            self.hdf5_path = {}
            self.filelist = {}

            for h5file in hdf5:
                with h5py.File(h5file, "r") as data:
                    assert "hdf5_number_of_archive_files" in data.attrs.keys(), "missing attributes expected in multi-part HDF5 archives "
                    assert "hdf5_archive_file_number" in data.attrs.keys(), "missing attributes expected in multi-part HDF5 archives "
                    tmp_numh5s = data.attrs["hdf5_number_of_archive_files"]
                    assert  tmp_numh5s == self.num_archives, f"{tmp_numh5s} archives expected, but only {self.num_archives} given."
                    anum = data.attrs["hdf5_archive_file_number"]

                    # create a lookup dictionary with key filename and value file
                    for nbytes in data["__filelist__"]:
                        self.filelist[nbytes.tobytes().decode()] = anum

                    self.compression=data.attrs["compression"]

                self.hdf5_path[anum] = Path(h5file)

            #double check number of archives, from each archive's number
            num_arch_verif = len(self.hdf5_path.keys())
            assert  num_arch_verif == self.num_archives, f"expected {self.num_archives} distinct archives, but got {num_arch_verif}"
            
        self.name = name

    # for a given file name, return archive containing it
    def get_archive_containing_filename(self, filename: str | Path) -> Path:
        if not self.multipart:
            return self.hdf5_path

        else:
            full_filename = (Path(self.name) / filename).as_posix()
            assert full_filename in self.filelist.keys(), f"{full_filename} not found in any of {self.num_archives} archives"
            an = self.filelist[full_filename]
            return self.hdf5_path[an]

    def get_measurement_detail(self) -> BytesIO:
        
        mea_detail_file = "MeasurementDetail.mrf"
        archive_name = self.get_archive_containing_filename(mea_detail_file)

        with h5py.File(archive_name, "r") as data:
            buffer_compressed = data[self.name][mea_detail_file + "." + self.compression][
                :
            ].tobytes()
            buffer = blosc2.decompress2(buffer_compressed)
            return BytesIO(buffer)

    def get_measurement_data(self) -> BytesIO:
        mea_data_file = "MeasurementData.mlf"
        archive_name = self.get_archive_containing_filename(mea_data_file)

        with h5py.File(archive_name, "r") as data:
            buffer_compressed = data[self.name][mea_data_file + "." + self.compression][
                :
            ].tobytes()
            buffer = blosc2.decompress2(buffer_compressed)
            return BytesIO(buffer)

    def get_measurement_settings(self, file_name: str) -> BytesIO:
        archive_name = self.get_archive_containing_filename(file_name)

        with h5py.File(archive_name, "r") as data:
            buffer_compressed = data[self.name][f"{file_name}.{self.compression}"][:].tobytes()
            buffer = blosc2.decompress2(buffer_compressed)
            return BytesIO(buffer)

    # Return log files. May reside in more than one archive
    def get_trace_logs(self) -> [str]:
        logdir = "LOG"

        if not self.multipart:
            archives_w_logs = set(self.hdf5_path)
        else:
            archives_w_logs = set()

            for key in self.filelist.keys():
                path = Path(key)
                if path.parts[0] == logdir:
                    an=self.filelist[key]
                    archives_w_logs.add(self.hdf5_path[an])
        
        for archive_name in archives_w_logs:
            with h5py.File(archive_name, "r") as data:
                for k in data[logdir].keys():
                    buffer_compressed = data[logdir][k][:].tobytes()
                    buffer = blosc2.decompress2(buffer_compressed)
                    log = []
                    for line in BytesIO(buffer):
                        log.append(line.decode("utf-8"))
                    yield log

    def get_image(self, file_name: str) -> NDArray:

        archive_name = self.get_archive_containing_filename(file_name)

        with h5py.File(archive_name, "r") as data:
            buffer_compressed = data[self.name][f"{file_name}.{self.compression}"][:].tobytes()
            buffer = blosc2.decompress2(buffer_compressed)
            return imread(BytesIO(buffer))

    def exists(self, file_name: str) -> bool:
        if not self.multipart:
            with h5py.File(self.hdf5_path, "r") as data:
                return f"{file_name}.{self.compression}" in data[self.name]
        else:
            fullpath = (Path(self.name) / file_name).as_posix()
            return fullpath in self.filelist.keys()