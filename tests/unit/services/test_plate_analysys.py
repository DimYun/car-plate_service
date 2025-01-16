"""Test module for plate_analysys"""

from copy import deepcopy

import numpy as np

from src.containers.containers import Container


class FakePlateClassifier:
    """Fake Plate response for tests"""
    def process(
        self,
        image: np.array,
        content_id: str
    ) -> dict:
        """
        Mock Plate Response for tests
        :param image: input image
        :param content_id: id string
        :return: mock predicted object
        """
        return {
            "plates": [
                {
                    "bbox": {
                        "x_min": 0,
                        "x_max": 1,
                        "y_min": 0,
                        "y_max": 1,
                    },
                    "value": 'TEST',
                }
            ]
        }


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
        with app_container.content_process.override(FakePlateClassifier()):
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
    with app_container.reset_singletons():
        with app_container.content_process.override(FakePlateClassifier()):
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
    with app_container.reset_singletons():
        with app_container.content_process.override(FakePlateClassifier()):
            initial_image = deepcopy(sample_image_np)
            plate_analytics = app_container.content_process()
            plate_analytics.process(sample_image_np, "test")

            assert np.allclose(initial_image, sample_image_np)
