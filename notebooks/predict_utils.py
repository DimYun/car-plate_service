"""Model prediction utilities."""
import itertools
import operator
from typing import List, Tuple

import numpy as np
import torch


def matrix_to_string(
    model_output: torch.Tensor, vocab: str
) -> Tuple[List[str], List[np.ndarray]]:
    """Decode ctc-matrix into string.

    :param model_output: tensor with model output
    :param vocab: alloved sharacters in vocabular
    :return: prepared string and decoded confidential levels
    """
    labels, confs = postprocess(model_output)
    labels_decoded, conf_decoded = decode(labels_raw=labels, conf_raw=confs)
    string_pred = labels_to_strings(labels_decoded, vocab)
    return string_pred, conf_decoded


def postprocess(model_output: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Postprocess model output.

    :param model_output: tensor with model output
    :return: labels and confidential values
    """
    output = model_output.permute(1, 0, 2)
    output = torch.nn.Softmax(dim=2)(output)
    confidences, labels = output.max(dim=2)
    confidences = confidences.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return labels, confidences


def decode(
    labels_raw: np.ndarray,
    conf_raw: np.ndarray
) -> Tuple[List[List[int]], List[np.ndarray]]:
    """Decode labels and confidential levels.

    :param labels_raw: rav labels from model prediction
    :param conf_raw: raw confidential values from model prediction
    :return: labels and confidentional values
    """
    result_labels = []
    result_confidences = []
    for label, conf in zip(labels_raw, conf_raw):
        result_one_labels = []
        result_one_confidences = []
        for l_group, group in itertools.groupby(
            zip(label, conf), operator.itemgetter(0)
        ):
            if l_group > 0:
                result_one_labels.append(l_group)
                result_one_confidences.append(max(list(zip(*group))[1]))
        result_labels.append(result_one_labels)
        result_confidences.append(np.array(result_one_confidences))
    return result_labels, result_confidences


def labels_to_strings(labels: List[List[int]], vocab: str) -> List[str]:
    """Convert labels to string.

    :param labels: predicted and processed labels
    :param vocab: alloved characters
    :return: predicted and processed list with strings
    """
    strings = []
    for single_str_labels in labels:
        try:
            output_str = "".join(
                vocab[char_index - 1] if char_index > 0 else "_"
                for char_index in single_str_labels
            )
        except IndexError:
            output_str = "Error"
        strings.append(output_str)
    return strings
