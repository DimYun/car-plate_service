"""Module for data transformation."""

from typing import Dict, List, Union

import albumentations as albu
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from numpy import random

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


def get_transforms(
    width: int,
    height: int,
    text_size: int,
    vocab: Union[str, List[str]],
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:
    """Process data in Dataset module for model training and validation.

    :param width: width of image to NN
    :param height: height of image to NN
    :param text_size: maximum number of characters in plate
    :param vocab: allowed characters
    :param preprocessing: prepare raw image for NN
    :param augmentations: add augmentation of image for NN
    :param postprocessing: prepare image for NN
    :return: compose list of augmentations
    """
    transforms = []

    if augmentations:
        transforms.extend(
            [
                CropPerspective(p_value=0.8),
                ScaleX(p_value=0.8),
            ]
        )

    if preprocessing:
        transforms.append(
            PadResizeOCR(
                target_height=height,
                target_width=width,
                mode="random" if augmentations else "left",
            )
        )

    if augmentations:
        transforms.extend(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.CLAHE(p=0.5),
                albu.Blur(blur_limit=3, p=0.3),
                albu.GaussNoise(p=0.3),
                albu.Downscale(scale_min=0.3, scale_max=0.9, p=0.5),
                albu.CoarseDropout(max_holes=6, max_height=6, max_width=6, p=0.3),
                albu.Affine(
                    scale=(0.7, 1.05),
                    translate_percent=0,
                    rotate=(-15, 15),
                    shear=(-15, 15),
                    cval=0,
                    mode=cv2.BORDER_CONSTANT,
                    p=0.3,
                ),
                albu.Perspective(pad_mode=cv2.BORDER_CONSTANT, p=0.3),
            ]
        )

    if postprocessing:
        transforms.extend(
            [
                albu.Normalize(),
                TextEncode(vocab=vocab, target_text_size=text_size),
                ToTensorV2(),
            ]
        )

    return albu.Compose(transforms)


class PadResizeOCR:
    """Lead to selected sizes with constant aspec ration, and add paddings if needed."""

    def __init__(
        self,
        target_width,
        target_height,
        pad_value: int = 0,
        mode: str = "random"
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.pad_value = pad_value
        self.mode = mode

        assert self.mode in {"random", "left", "center"}

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        """Call function for PadResizeOCR class.

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
                image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
            )

        kwargs["image"] = image
        return kwargs


class TextEncode:
    """Encode text."""

    symbols_non_latin = (
        "АВ5СЕКХНМОРТУ0123456789ӨҮՈ",
        "AB5CEKXHMOPTY0123456789&Y#",
    )

    tr_non_latin = {ord(a): ord(b) for a, b in zip(*symbols_non_latin)}

    def __init__(self, vocab: Union[str, List[str]], target_text_size: int):
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.target_text_size = target_text_size

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        """Call function for Text Encoding.

        :param force_apply: flag to force apply
        :param kwargs: keyword arguments
        :return: dictionary with transformed data
        """
        source_text = kwargs["text"].strip()
        source_text = source_text.upper().translate(self.tr_non_latin)

        postprocessed_text = [
            self.vocab.index(c) + 1 for c in source_text if c in self.vocab
        ]
        postprocessed_text = np.pad(
            postprocessed_text,
            (0, self.target_text_size - len(postprocessed_text)),
            mode="constant",
        )
        postprocessed_text = torch.IntTensor(postprocessed_text)

        kwargs["text"] = postprocessed_text

        return kwargs


class CropPerspective:
    """Selfwrited class for augmentations for geometric transformation."""

    def __init__(
        self,
        p_value: float = 0.5,
        width_ratio: float = 0.04,
        height_ratio: float = 0.08
    ):
        self.p_value = p_value
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        """Call function for CropPerspective transformation.

        :param force_apply: flag to force applying
        :param kwargs: keyword arguments
        :return: dictionary with transformed data
        """
        image = kwargs["image"].copy()

        if random.random() < self.p_value:
            img_h, img_w, img_c = image.shape

            pts1 = np.float32([[0, 0], [0, img_h], [img_w, img_h], [img_w, 0]])
            dw = img_w * self.width_ratio
            dh = img_h * self.height_ratio

            pts2 = np.float32(
                [
                    [random.uniform(-dw, dw), random.uniform(-dh, dh)],
                    [random.uniform(-dw, dw), img_h - random.uniform(-dh, dh)],
                    [img_w - random.uniform(-dw, dw), img_h - random.uniform(-dh, dh)],
                    [img_w - random.uniform(-dw, dw), random.uniform(-dh, dh)],
                ]
            )

            matrix = cv2.getPerspectiveTransform(pts2, pts1)
            dst_w = (pts2[3][0] + pts2[2][0] - pts2[1][0] - pts2[0][0]) * 0.5
            dst_h = (pts2[2][1] + pts2[1][1] - pts2[3][1] - pts2[0][1]) * 0.5
            image = cv2.warpPerspective(
                image,
                matrix,
                (int(dst_w), int(dst_h)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        kwargs["image"] = image
        return kwargs


class ScaleX:
    """Selfwrited class for augmentations for image resize."""

    def __init__(
        self,
        p_value: float = 0.5,
        scale_min: float = 0.8,
        scale_max: float = 1.2
    ):
        self.p_value = p_value
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        """Call function for ScaleX augmentations.

        :param force_apply: flag to force applying
        :param kwargs: keyword arguments
        :return: dictionary with transformed data
        """
        image = kwargs["image"].copy()

        if random.random() < self.p_value:
            img_h, img_w, _ = image.shape
            img_w = int(img_w * random.uniform(self.scale_min, self.scale_max))
            image = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        kwargs["image"] = image
        return kwargs
