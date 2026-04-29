import io
from pathlib import Path
import torch
from potato.model import Pmodel
from potato.config import CLASS_NAMES, IMAGE_SIZE
from torchvision import transforms
from PIL import Image


class PotatoDiseaseHandler:
    def __init__(self):
        self.model = None
        self.device = torch.device("cpu")
        self.transform = None


    def initialize(self, context):
        model_dir = Path(context.system_properties.get("model_dir"))
        model_path = model_dir / "model_0.1.pth"

        self.model = Pmodel(len(CLASS_NAMES))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def preprocess(self, data):
        image_bytes = data[0].get("body")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)

        return image_tensor
    
    def inference(self, image_tensor):
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_index = int(torch.argmax(probabilities).item())

        return predicted_index, probabilities
    
    def postprocess(self, inference_result):
        predicted_index, probabilities = inference_result
        predictions = {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(CLASS_NAMES)
        }

        return [{
            "class": CLASS_NAMES[predicted_index],
            "confidence": predictions[CLASS_NAMES[predicted_index]],
            "predictions": predictions,
        }]
    
    def handle(self, data, context):
        image_tensor = self.preprocess(data)
        inference_result = self.inference(image_tensor)
        return self.postprocess(inference_result)
