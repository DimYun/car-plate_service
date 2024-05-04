"""Tests for the plate_classifier module."""

from copy import deepcopy

import numpy as np

from src.containers.containers import Container


def test_predicts_not_fail(
    app_container: Container,
    sample_image_np: np.ndarray
) -> None:
    """
    Test that a plate models did not fail
    :param app_container: container with application
    :param sample_image_np: input image
    :return:
    """
    with app_container.reset_singletons():
        plate_analytics = app_container.content_process()
        plate_analytics.process(sample_image_np, "test")


def test_prob_less_or_equal_to_one(
    app_container: Container,
    sample_image_np: np.ndarray,
) -> None:
    """
    Test that the pedicted value didn't exeed 10 characters
    :param app_container: container with application
    :param sample_image_np: input image
    :return:
    """
    plate_analytics = app_container.content_process()
    plate2prob = plate_analytics.process(sample_image_np, "test")
    for prob in plate2prob['plates']:
        assert len(prob['value']) <= 10
        assert len(prob['value'])


def test_predict_dont_mutate_initial_image(
    app_container: Container,
    sample_image_np: np.ndarray,
) -> None:
    """
    Test that we didn't change image
    :param app_container: container with application
    :param sample_image_np: input image
    :return:
    """
    initial_image = deepcopy(sample_image_np)
    plate_analytics = app_container.content_process()
    plate_analytics.process(sample_image_np, "test")

    assert np.allclose(initial_image, sample_image_np)
