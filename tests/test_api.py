from io import BytesIO
import asyncio

from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile
import pytest

from api.main import health, model_info, predict
from potato.config import CLASS_NAMES, IMAGE_SIZE, MODEL_VERSION


def create_test_image() -> BytesIO:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(120, 180, 90))
    image_file = BytesIO()
    image.save(image_file, format="JPEG")
    image_file.seek(0)
    return image_file


def test_health_endpoint():
    assert asyncio.run(health()) == {
        "status": "ok",
        "model_loaded": True,
    }


def test_model_info_endpoint():
    data = asyncio.run(model_info())

    assert data["model_name"] == "potato-disease-classifier"
    assert data["model_version"] == MODEL_VERSION
    assert data["classes"] == CLASS_NAMES
    assert data["image_size"] == IMAGE_SIZE


def test_predict_endpoint_returns_prediction():
    upload = UploadFile(create_test_image(), filename="test.jpg")

    data = asyncio.run(predict(upload))

    assert data["class"] in CLASS_NAMES
    assert 0 <= data["confidence"] <= 1
    assert set(data["predictions"].keys()) == set(CLASS_NAMES)

    total_probability = sum(data["predictions"].values())
    assert total_probability == pytest.approx(1.0, abs=1e-5)


def test_predict_endpoint_rejects_invalid_file():
    upload = UploadFile(BytesIO(b"not an image"), filename="bad.txt")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(predict(upload))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid image file"
