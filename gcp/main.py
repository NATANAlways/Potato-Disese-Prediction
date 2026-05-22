from io import BytesIO
from pathlib import Path
import threading

from google.cloud import storage
from PIL import Image
import torch
from torchvision import transforms

from potato.config import CLASS_NAMES, IMAGE_SIZE, MODEL_VERSION
from potato.model import Pmodel


BUCKET_NAME = "torch-potato-disease-classification-12"
MODEL_BLOB_NAME = "models/model_0.1.pth"
LOCAL_MODEL_PATH = Path("/tmp/model_0.1.pth")
DEVICE = torch.device("cpu")

model = None
model_lock = threading.Lock()

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def download_model_if_needed():
    if LOCAL_MODEL_PATH.exists():
        return LOCAL_MODEL_PATH

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_BLOB_NAME)
    blob.download_to_filename(LOCAL_MODEL_PATH)

    print(f"Downloaded {MODEL_BLOB_NAME} from {BUCKET_NAME} to {LOCAL_MODEL_PATH}.")
    return LOCAL_MODEL_PATH


def load_model():
    global model

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        model_path = download_model_if_needed()
        loaded_model = Pmodel(len(CLASS_NAMES)).to(DEVICE)
        loaded_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        loaded_model.eval()
        model = loaded_model

    return model


def predict(request):
    if request.method == "GET":
        return {
            "status": "ok",
            "model_name": "potato-disease-classifier",
            "model_version": MODEL_VERSION,
            "model_downloaded": LOCAL_MODEL_PATH.exists(),
            "model_loaded": model is not None,
        }

    if "file" not in request.files:
        return {"error": "No image file uploaded. Use form field name 'file'."}, 400

    try:
        image_file = request.files["file"]
        image = Image.open(BytesIO(image_file.read())).convert("RGB")
    except Exception:
        return {"error": "Invalid image file"}, 400

    active_model = load_model()
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = active_model(input_tensor)
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
