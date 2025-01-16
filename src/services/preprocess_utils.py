"""Model with preprocessing utilities"""

import typing as tp

import cv2
import numpy as np
import onnxruntime as ort
import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from src.services.predict_utils import matrix_to_string

BATCH_SIZE = 1
PROVIDERS = (
    'CUDAExecutionProvider',
    "CPUExecutionProvider",
)


class PlatePredictor:
    def __init__(self, config: dict):
        # Инициализировали один раз и сохранили в атрибут
        self.ort_session = ort.InferenceSession(
            config.plate_checkpoint,
            providers=PROVIDERS,
        )
        self.image_size = (
            config.plate_img_width,
            config.plate_img_height,
        )

    def onnx_preprocessing(
        self,
        image: np.ndarray,
        image_size: tp.Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        image = cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_LINEAR)
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * 255.0
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32) * 255.0
        denominator = np.reciprocal(std, dtype=np.float32)
        image = image.astype(np.float32)
        image -= mean
        image *= denominator
        return image.transpose((2, 0, 1))[None]

    def predict(
        self,
        image: np.ndarray,
    ) -> np.array:
        onnx_input = self.onnx_preprocessing(image, image_size=self.image_size)
        onnx_input = np.concatenate([onnx_input] * BATCH_SIZE)
        ort_inputs = {self.ort_session.get_inputs()[0].name: onnx_input}
        ort_outputs = self.ort_session.run(None, ort_inputs)[0]
        pr_mask = ort_outputs.squeeze().round()
        return self.get_crop(image, pr_mask)

    def get_crop(
        self,
        image: np.ndarray,
        pr_mask: np.ndarray,
    ) -> tp.Tuple[np.array, list]:
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


class OCRPredictor:
    def __init__(self, config: dict):
        # Инициализировали один раз и сохранили в атрибут
        self.ort_session = ort.InferenceSession(
            config.ocr_checkpoint,
            providers=PROVIDERS,
        )
        self.image_width = config.ocr_img_width
        self.image_height = config.ocr_img_height
        self.text_size = config.text_size
        self.vocabular = config.vocabular

    def get_transforms(
        self,
        text_size: int,
        vocab: tp.Union[str, tp.List[str]],
    ) -> tp.Union[albu.BasicTransform, albu.BaseCompose]:
        transforms = []
        transforms.append(
            PadResizeOCR(
                target_height=self.image_height,
                target_width=self.image_width,
                mode="left",
            ),
        )
        transforms.extend(
            [
                albu.Normalize(),
                ToTensorV2(),
            ],
        )
        return albu.Compose(transforms)

    def predict(
        self,
        image: np.ndarray,
    ) -> np.array:
        # выполняем инференс ONNX Runtime
        ort_outputs_ocr = self.ort_session.run(
            None,
            self.prepare_image(image)
        )[0]
        string_pred, _ = matrix_to_string(
            torch.from_numpy(ort_outputs_ocr),
            self.vocabular,
        )
        return string_pred

    def prepare_image(self, image: np.ndarray) -> dict:
        # готовим входной тензор
        transforms = self.get_transforms(
            text_size=self.text_size,
            vocab=self.vocabular,
        )
        onnx_input_ocr = transforms(image=image, text="")["image"][None]
        onnx_input_ocr = np.concatenate([onnx_input_ocr] * BATCH_SIZE)
        return {self.ort_session.get_inputs()[0].name: onnx_input_ocr}


class PadResizeOCR:
    """
    Приводит к нужному размеру с сохранением отношения сторон,
    если нужно добавляет падинги.
    """

    def __init__(
        self, target_width, target_height, pad_value: int = 0, mode: str = "random",
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.pad_value = pad_value
        self.mode = mode

        assert self.mode in {"random", "left", "center"}

    def __call__(self, force_apply=False, **kwargs) -> tp.Dict[str, np.ndarray]:
        """
        Call function for PadResizeOCR class
        :param force_apply: flag to force applying
        :param kwargs: keyword arguments
        :return: dictionary with transformed data
        """
        image = kwargs["image"].copy()

        h, w = image.shape[:2]

        tmp_w = min(int(w * (self.target_height / h)), self.target_width)
        image = cv2.resize(image, (tmp_w, self.target_height))

        dw = np.round(self.target_width - tmp_w).astype(int)
        if dw > 0:
            if self.mode == "random":
                pad_left = np.random.randint(dw)
            elif self.mode == "left":
                pad_left = 0
            else:
                pad_left = dw // 2

            pad_right = dw - pad_left

            image = cv2.copyMakeBorder(
                image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0,
            )

        kwargs["image"] = image
        return kwargs
