import os.path  # noqa: WPS301

import cv2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from src.containers.containers import Container
from src.routes import planets as planets_routes
from src.routes.routers import router as app_router

TESTS_DIR = "tests"


@pytest.fixture(scope="session")
def sample_image_bytes():
    with open(os.path.join(TESTS_DIR, "images", "file_0.jpg"), "rb") as image_file:
        yield image_file.read()


@pytest.fixture
def sample_image_np():
    img = cv2.imread(os.path.join(TESTS_DIR, "images", "file_0.jpg"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def app_config():
    return OmegaConf.load(os.path.join(TESTS_DIR, "test_config.yml"))


@pytest.fixture
def app_container(app_config):
    container = Container()
    container.config.from_dict(app_config)
    return container


@pytest.fixture
def wired_app_container(app_config):
    container = Container()
    container.config.from_dict(app_config)
    container.wire([planets_routes])
    yield container
    container.unwire()


@pytest.fixture
def test_app(wired_app_container):
    app = FastAPI()
    app.include_router(app_router, prefix="/planets", tags=["planet"])
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)
