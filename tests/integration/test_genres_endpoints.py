from http import HTTPStatus

from fastapi.testclient import TestClient


def test_types_list(client: TestClient):
    response = client.get("/planets/get_content?content_id=test")
    assert response.status_code == HTTPStatus.OK

    planet_types = response.json()["content"]

    assert isinstance(planet_types, str)


def test_predict(client: TestClient, sample_image_bytes: bytes):
    files = {
        "image": sample_image_bytes,
    }
    response = client.post("/planets/process_content", files=files)
    assert response.status_code == HTTPStatus.OK

    predicted_scores = response.json()["scores"]

    assert isinstance(predicted_scores, str)
