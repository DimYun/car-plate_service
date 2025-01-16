"""Module for FastAPI requests infrastructure"""

import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File, UploadFile

from src.containers.containers import Container
from src.routes.routers import router
from src.services.plate_process import ProcessPlate, Storage


@router.get("/get_content")
@inject
def get_content(
    content_id: str,
    storage: Storage = Depends(Provide[Container.store]),
) -> dict:
    """
    Define GET content
    :param content_id: id of content
    :param storage: container with storage functionality
    :return: dict with content
    """
    return {
        "code": 200,
        "predictions": storage.get(content_id),
        "error": None,
    }


@router.post("/process_content")
@inject
def process_content(
    content_image: UploadFile = File(
        ...,
        title="PredictorInputImage",
        alias="image",
        description="Image for inference.",
    ),
    content_process: ProcessPlate = Depends(Provide[Container.content_process]),
) -> dict:
    """
    Define POST
    :param content_image: input image
    :param content_process: container with process functionality
    :return: dictionary with results in json format
    """
    image_data = content_image.file.read()
    content_image.file.close()

    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    str_process = content_process.process(
        image,
        str(content_image.filename),
    )
    return {
        "code": 200,
        "predictions": str_process,
        "error": None,
    }
