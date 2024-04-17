import json
from glob import glob
from os.path import basename, exists, join, splitext

import numpy as np
import pandas as pd
from numpy._typing import ArrayLike
from pydantic import BaseModel, PositiveFloat, PositiveInt
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from tifffile import imread
from tqdm import tqdm

from faim_hcs.processor import AbstractProcessor
from faim_hcs.spot_detection import spot
from faim_hcs.spot_detection.roi import pad_bbox
from faim_hcs.spot_detection.spot import compute_axial_sigma


class SpotDetectionParameters(BaseModel):
    input_dir: str
    segmentation_dir: str
    output_dir: str
    wavelength_1: PositiveInt
    wavelength_2: PositiveInt
    numerical_aperture: PositiveFloat
    spacing: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    h_1: PositiveInt
    h_2: PositiveInt


class SpotDetectionProcessor(AbstractProcessor):
    def __init__(
        self,
        parameters: SpotDetectionParameters,
    ):
        super().__init__(name="spot_detection")
        self.parameters = parameters
        self.logger.info(f"Parameters: " f"{json.dumps(self.parameters, indent=4)}")

    def run(self):
        """
        Perform spot detection on all data in the input directory.
        """
        self.logger.info("Starting spot detection...")

        w1_files, w2_files, seg_files = self.collect_files()
        self.logger.info(f"Found {len(w1_files)} files.")

        for w1, w2, seg in tqdm(zip(w1_files, w2_files, seg_files)):
            self.process(w1_file=w1, w2_file=w2, seg_file=seg)

        self.logger.info("Done!")

    def collect_files(self) -> tuple[list[str], list[str], list[str]]:
        """
        Collect files from input_dir and assert that the corresponding files
        exist in segmentation_dir.
        """
        w1_files = glob(join(self.parameters.input_dir, "*w1Conf640.stk"))
        w2_files = self.find_matching_w2_files(w1_files=w1_files)
        seg_files = self.find_matching_seg_files(w1_files=w1_files)

        return w1_files, w2_files, seg_files

    def process(self, w1_file: str, w2_file: str, seg_file: str):
        """
        Process a single image.
        """
        w1_spots = self.detect_spots_in_rois(
            img_file=w1_file,
            seg_file=seg_file,
        )
        self.save(w1_spots, w1_file)
        self.logger.info(f"Found {len(w1_spots)} spots in {basename(w1_file)}.")

        w2_spots = self.detect_spots_in_rois(
            img_file=w2_file,
            seg_file=seg_file,
        )
        self.save(w2_spots, w2_file)
        self.logger.info(f"Found {len(w2_spots)} spots in {basename(w2_file)}.")

    @staticmethod
    def find_matching_w2_files(w1_files: list[str]):
        """Replace w1Conf640 with w2Conf561 in w1_files and assert that the
        file exists."""
        w2_files = [w1.replace("w1Conf640", "w2Conf561") for w1 in w1_files]
        for w2 in w2_files:
            assert exists(w2), f"File {w2} is missing."
        return w2_files

    @staticmethod
    def find_matching_seg_files(w1_files: list[str]):
        """Replace w1Conf640.stk with -CELL_SEG.tif in w1_files and assert that the
        file exists."""
        seg_files = [w1.replace("w1Conf640.stk", "-CELL_SEG.tif") for w1 in w1_files]
        for seg in seg_files:
            assert exists(seg), f"File {seg} is missing."
        return seg_files

    def detect_spots_in_rois(self, img_file: str, seg_file: str) -> pd.DataFrame:
        """
        Extract spots from each ROI in the image.

        Parameters
        ----------
        img_file
            Image file.
        seg_file
            Segmentation file.

        Returns
        -------
        Detected spots.
        """
        image = imread(img_file)
        segmentation = imread(seg_file)
        spots = []
        for roi in regionprops(segmentation):
            roi_img, mask, offset = self._crop_padded_roi_image(
                image=image,
                roi=roi,
            )
            roi_spots = spot.detection(
                image=roi_img,
                mask=mask,
                wavelength=self.parameters.wavelength_1,
                numerical_aperture=self.parameters.numerical_aperture,
                spacing=self.parameters.spacing,
                intensity_threshold=self.parameters.h_1,
            )
            roi_spots += np.array(offset)
            spots.extend(roi_spots.tolist())

        return pd.DataFrame(spots, columns=["axis-0", "axis-1", "axis-2"])

    def _crop_padded_roi_image(self, image: ArrayLike, roi: RegionProperties):
        """
        Crop a padded ROI from the image.
        """
        axial_padding = int(
            3
            * compute_axial_sigma(
                self.parameters.wavelength_1,
                self.parameters.numerical_aperture,
                self.parameters.spacing[1],
            )
        )
        lateral_padding = int(
            3
            * compute_axial_sigma(
                self.parameters.wavelength_1,
                self.parameters.numerical_aperture,
                self.parameters.spacing[0],
            )
        )

        padded_bbox = pad_bbox(
            bbox=roi.bbox,
            padding=(axial_padding, lateral_padding, lateral_padding),
            shape=image.shape,
        )
        padded_roi = np.pad(
            roi.image,
            pad_width=(
                (axial_padding, axial_padding),
                (lateral_padding, lateral_padding),
                (lateral_padding, lateral_padding),
            ),
        )
        return (
            image[
                padded_bbox[0] : padded_bbox[3],
                padded_bbox[1] : padded_bbox[4],
                padded_bbox[2] : padded_bbox[5],
            ],
            padded_roi,
            padded_bbox[:3],
        )

    def save(self, spots: pd.DataFrame, src_file: str):
        """
        Save spots to a CSV file.
        """
        name, _ = splitext(basename(src_file))
        path = join(self.parameters.output_dir, f"{name}-SPOTS.csv")
        spots.to_csv(path, index_label=True)
