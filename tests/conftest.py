"""Module with test fixtures"""
import os.path  # noqa: WPS301

import cv2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from src.containers.containers import Container
from src.routes import plates as plate_routes
from src.routes.routers import router as app_router

TESTS_DIR = "tests"


@pytest.fixture(scope="session")
def sample_image_bytes():
    """
    Fixture with image in bytes
    :return: image in bytes
    """
    with open(os.path.join(TESTS_DIR, "images", "test.jpg"), "rb") as image_file:
        yield image_file.read()


@pytest.fixture
def sample_image_np():
    """
    Fixture with image in numpy
    :return: image in numpy array
    """
    img = cv2.imread(os.path.join(TESTS_DIR, "images", "test.jpg"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def app_config():
    """
    Fixture with loaded test config
    :return: config object (dict)
    """
    return OmegaConf.load(os.path.join(TESTS_DIR, "test_config.yml"))


@pytest.fixture
def app_container(app_config):
    """
    Fixture with loded DPI conteiner from config
    :param app_config: config object
    :return: container
    """
    container = Container()
    container.config.from_dict(app_config)
    return container


@pytest.fixture
def wired_app_container(app_config):
    """
    Fixture with wired container
    :param app_config: config object
    :return: container
    """
    container = Container()
    container.config.from_dict(app_config)
    container.wire([plate_routes])
    yield container
    container.unwire()


@pytest.fixture
def test_app(wired_app_container):
    """
    Fixture with loaded test app
    :param wired_app_container: new container for FastAPI app
    :return:
    """
    app = FastAPI()
    app.include_router(app_router, prefix="/plates", tags=["plates"])
    return app


@pytest.fixture
def client(test_app):
    """
    Fixture with test client application
    :param test_app: FastAPI app instance
    :return: test client
    """
    return TestClient(test_app)
