from http import HTTPStatus

from fastapi.testclient import TestClient


def test_types_list(client: TestClient):
    """
    Test for response
    :param client: test FastAPI client
    :return:
    """
    response = client.get("/plates/get_content?content_id=test")
    assert response.status_code == HTTPStatus.OK

    plate_types = response.json()["content"]["plates"][0]['value'][0]

    assert isinstance(plate_types, str)


def test_predict(client: TestClient, sample_image_bytes: bytes):
    """
    Test for prediction
    :param client: test FastAPI client
    :param sample_image_bytes:
    :return:
    """
    files = {
        "image": sample_image_bytes,
    }
    response = client.post("/plates/process_content", files=files)
    assert response.status_code == HTTPStatus.OK

    predicted_string = response.json()["scores"]["plates"][0]["value"][0]

    assert isinstance(predicted_string, str)
