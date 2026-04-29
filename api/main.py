from pathlib import Path
from io import BytesIO
import os
import sys

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from PIL import Image
from torchvision import transforms
import torch
import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from potato.config import CLASS_NAMES, IMAGE_SIZE, MODEL_VERSION
from potato.model import Pmodel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "potato" / "model_0.1.pth"))
MODEL_GCS_URI = os.getenv("MODEL_GCS_URI")
DEVICE = torch.device("cpu")


preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def download_model_from_gcs(gcs_uri: str, destination: Path) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("MODEL_GCS_URI must start with gs://")

    bucket_name, blob_name = gcs_uri.removeprefix("gs://").split("/", 1)
    destination.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination)


if MODEL_GCS_URI:
    download_model_from_gcs(MODEL_GCS_URI, MODEL_PATH)

model = Pmodel(len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

@app.get("/")
async def root():
    return {
        "message": "Potato disease API is running",
        "endpoints": {
            "health": "/ping",
            "predict": "/predict",
            "docs": "/docs",
        },
    }

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }

@app.get("/model-info")
async def model_info():
    return {
        "model_name": "potato-disease-classifier",
        "model_version": MODEL_VERSION,
        "classes": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

def read_file_as_image(data) -> Image.Image:
    image = Image.open(BytesIO(data)).convert("RGB")
    return image
    
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        image = read_file_as_image(file.file.read())
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())

    predictions = {
        class_name: float(probabilities[index].item())
        for index, class_name in enumerate(CLASS_NAMES)
    }

    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": predictions[CLASS_NAMES[predicted_index]],
        "predictions": predictions,
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
