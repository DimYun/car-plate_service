from copy import deepcopy

import numpy as np

from src.containers.containers import Container


def test_predicts_not_fail(app_container: Container, sample_image_np: np.ndarray):
    planet_analytics = app_container.content_process()
    planet_analytics.process(sample_image_np, "test")


def test_prob_less_or_equal_to_one(
    app_container: Container,
    sample_image_np: np.ndarray,
):
    planet_analytics = app_container.content_process()
    planet2prob = planet_analytics.process(sample_image_np, "test")
    for type_prob in planet2prob.split(","):
        prob = float(type_prob.split(":")[-1])
        assert prob <= 1
        assert prob >= 0


def test_predict_dont_mutate_initial_image(
    app_container: Container,
    sample_image_np: np.ndarray,
):
    initial_image = deepcopy(sample_image_np)
    planet_analytics = app_container.content_process()
    planet_analytics.process(sample_image_np, "test")

    assert np.allclose(initial_image, sample_image_np)
