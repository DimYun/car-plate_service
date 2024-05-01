import os
import typing as tp

import numpy as np
import json
from pathlib import Path

from src.services.preprocess_utils import get_crop, get_ocr


class Storage:
    def __init__(self, config: dict):
        self._config = config
        os.makedirs(config["dir_path"], exist_ok=True)
        os.makedirs(config["dir_upload"], exist_ok=True)

    def get(self, content_id: str) -> tp.Optional[str]:
        content_path = self._get_path(content_id)
        if not os.path.exists(content_path):
            return "Start process image first"
        with open(content_path, 'r') as json_data:
            return json.load(json_data)

    def _get_path(self, content_id: str) -> tp.Union[str, Path]:
        return Path(self._config["dir_path"]) / f"{content_id}.json"

    def save(self, content_json: dict, content_id: str) -> None:
        # Save COCO
        json_object = json.dumps(content_json, indent=4)
        with open(self._get_path(content_id), 'w') as json_file:
            json_file.write(json_object)


class ProcessPlate:
    STATUS = "Start process image first"

    def __init__(self, storage: Storage):
        self._storage = storage

    def process(self, image: np.ndarray, content_id: str) -> dict:
        json_responce = self._storage.get(content_id)
        if json_responce == self.STATUS:
            # 1st stage - get plate crop
            image_plate_crop, bbox_xxyy = get_crop(image)
            print(image_plate_crop.shape, bbox_xxyy)
            # 2nd stage - get OCR plate number
            plate_ocr = get_ocr(image_plate_crop)
            # 3rd stage - save data
            json_responce = {
                'plates': [
                    {
                        'bbox': {
                            'x_min': bbox_xxyy[0],
                            'x_max': bbox_xxyy[1],
                            'y_min': bbox_xxyy[2],
                            'y_max': bbox_xxyy[3]
                        },
                        'value': plate_ocr
                    }
                ]
            }
            self._storage.save(json_responce, content_id)
        return json_responce
