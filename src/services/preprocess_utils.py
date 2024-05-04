"""Model with preprocessing utilities"""

import typing as tp

import cv2
import numpy as np
import onnxruntime as ort
import torch

from src.services.predict_utils import matrix_to_string
from src.services.transforms import get_transforms

BATCH_SIZE = 1
PROVIDERS = (
    'CUDAExecutionProvider',
    "CPUExecutionProvider",
)


def get_crop(image: np.ndarray) -> tp.Tuple[np.ndarray, list]:
    """
    Predict plate area and crop box from unresized image
    :param image: RGB image
    :return: cropped RGB image
    """
    ONNX_MODEL_PLATE_DET = "models/exp-2_plate-model.onnx"
    ort_session_det = ort.InferenceSession(ONNX_MODEL_PLATE_DET, providers=PROVIDERS)
    onnx_input = onnx_preprocessing(image, image_size=(512, 512))
    onnx_input = np.concatenate([onnx_input] * BATCH_SIZE)
    ort_inputs = {ort_session_det.get_inputs()[0].name: onnx_input}
    ort_outputs = ort_session_det.run(None, ort_inputs)[0]
    pr_mask = ort_outputs.squeeze().round()
    # Cropping an image
    indexes = np.where(pr_mask > 0)
    x_min = indexes[1].min()
    x_max = indexes[1].max()
    y_min = indexes[0].min()
    y_max = indexes[0].max()
    # Convert to full-size image
    y_factor = image.shape[0] / pr_mask.shape[0]
    x_factor = image.shape[1] / pr_mask.shape[1]
    x_min_orig = int(x_min * x_factor)
    x_max_orig = int(x_max * x_factor)
    y_min_orig = int(y_min * y_factor)
    y_max_orig = int(y_max * y_factor)
    return (
        image[y_min_orig:y_max_orig, x_min_orig:x_max_orig, :],
        [x_min_orig, x_max_orig, y_min_orig, y_max_orig],
    )


def get_ocr(plate_image: np.ndarray) -> str:
    """
    Predict OCR in cropped plates
    :param image: input image with cropped plate
    :return: predicted OCR string
    """
    ONNX_MODEL_PLATE_OCR = "models/exp-1_plate-ocr-model.onnx"
    VOCAB = "#&0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZÄÅÖÜĆČĐŠŽАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЭЮЯ"
    ort_session_ocr = ort.InferenceSession(ONNX_MODEL_PLATE_OCR, providers=PROVIDERS)
    transforms = get_transforms(
        width=416,
        height=64,
        text_size=10,
        vocab=VOCAB,
        postprocessing=True,
        augmentations=False,
    )
    onnx_input_ocr = transforms(image=plate_image, text="")["image"][None]
    onnx_input_ocr = np.concatenate([onnx_input_ocr] * BATCH_SIZE)
    ort_inputs_ocr = {ort_session_ocr.get_inputs()[0].name: onnx_input_ocr}
    # выполняем инференс ONNX Runtime
    ort_outputs_ocr = ort_session_ocr.run(None, ort_inputs_ocr)[0]
    string_pred, _ = matrix_to_string(torch.from_numpy(ort_outputs_ocr), VOCAB)
    return string_pred


def onnx_preprocessing(
    image: np.ndarray,
    image_size: tp.Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Convert numpy-image to array for inference ONNX Runtime model.
    :param image: input raw image of road scene
    :param image_size: desire size for reshaped image
    :return: reshaped image
    """

    # resize
    image = cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_LINEAR)

    # normalize
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * 255.0
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32) * 255.0
    denominator = np.reciprocal(std, dtype=np.float32)
    image = image.astype(np.float32)
    image -= mean
    image *= denominator

    # transpose
    return image.transpose((2, 0, 1))[None]
