from copy import deepcopy

import numpy as np

from src.containers.containers import Container


class FakePlanetClassifier:
    def process(self, image: bytes, content_id: str):
        return "".join(
            [
                "haze: 0, primary: 0, agriculture: 0, clear: 0, water: 0, ",
                "habitation: 0, road: 0, cultivation: 0, slash_burn: 0, ",
                "cloudy: 0, partly_cloudy: 0, conventional_mine: 0, bare_ground: 0, ",
                "artisinal_mine: 0, blooming: 0, selective_logging: 0, blow_down: 0",
            ],
        )


def test_predicts_not_fail(app_container: Container, sample_image_np: np.ndarray):
    with app_container.reset_singletons():
        with app_container.content_process.override(FakePlanetClassifier()):
            planet_analytics = app_container.content_process()
            planet_analytics.process(sample_image_np, "test")


def test_prob_less_or_equal_to_one(
    app_container: Container,
    sample_image_np: np.ndarray,
):
    with app_container.reset_singletons():
        with app_container.content_process.override(FakePlanetClassifier()):
            planet_analytics = app_container.content_process()
            planet2prob = planet_analytics.process(sample_image_np, "test")
            for type_prob in planet2prob.split(","):
                prob = int(type_prob.split(":")[-1])
                assert prob <= 1
                assert prob >= 0


def test_predict_dont_mutate_initial_image(
    app_container: Container,
    sample_image_np: np.ndarray,
):
    with app_container.reset_singletons():
        with app_container.content_process.override(FakePlanetClassifier()):
            initial_image = deepcopy(sample_image_np)
            planet_analytics = app_container.content_process()
            planet_analytics.process(sample_image_np, "test")

            assert np.allclose(initial_image, sample_image_np)
