"""Module for define APP plate process."""

import json
import os
import typing as tp
from pathlib import Path

import numpy as np

from src.services.preprocess_utils import PlatePredictor, OCRPredictor


class Storage:
    """Class for storing processed results"""
    def __init__(self, config: dict):
        self._config = config
        os.makedirs(config["dir_path"], exist_ok=True)
        os.makedirs(config["dir_upload"], exist_ok=True)

    def get(self, content_id: str) -> tp.Optional[str]:
        """
        Check and get early processed results
        :param content_id: id of image to process (equal to image name)
        :return: str or json object with results
        """
        content_path = self._get_path(content_id)
        if not os.path.exists(content_path):
            return "Start process image first"
        with open(content_path, "r") as json_data:
            return json.load(json_data)

    def save(self, content_json: dict, content_id: str) -> None:
        """
        Save processed results to json file
        :param content_json: dict with results
        :param content_id: id of results
        :return:
        """
        json_object = json.dumps(content_json, indent=4)
        with open(self._get_path(content_id), "w") as json_file:
            json_file.write(json_object)

    def _get_path(self, content_id: str) -> tp.Union[str, Path]:
        """
        Check if path exist
        :param content_id: id of results
        :return: oath for results
        """
        return Path(self._config["dir_path"]) / f"{content_id}.json"


class ProcessPlate:
    """Class for storing processed"""
    status = "Start process image first"

    def __init__(
        self,
        storage: Storage,
        plate_predictor: PlatePredictor,
        ocr_predictor: OCRPredictor
    ):
        self._storage = storage
        self.plate_predictor = plate_predictor
        self.ocr_predictor = ocr_predictor
        print("finish process plate init")

    def process(self, image: np.ndarray, content_id: str) -> dict:
        """
        Process image
        :param image: input image to process
        :param content_id: id of input image
        :return: dictionary with results in json format
        """
        print("Call", content_id)
        json_responce = self._storage.get(content_id)
        if json_responce == self.status:
            # 1st stage - get plate crop
            image_plate_crop, bbox_xxyy = self.plate_predictor.predict(
                image,
            )
            # 2nd stage - get OCR plate number
            plate_ocr = self.ocr_predictor.predict(
                image_plate_crop
            )
            # 3rd stage - save data
            json_responce = {
                "plates": [
                    {
                        "bbox": {
                            "x_min": bbox_xxyy[0],
                            "x_max": bbox_xxyy[1],
                            "y_min": bbox_xxyy[2],
                            "y_max": bbox_xxyy[3],
                        },
                        "value": plate_ocr,
                    }
                ]
            }
            self._storage.save(json_responce, content_id)
        return json_responce
