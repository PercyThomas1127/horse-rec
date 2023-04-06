import torch
import torchvision

class HorseDetector:
    def __init__(self):
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image):
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities[1].item()

import io
from PIL import Image
from flask import Flask, request, jsonify
from horse_detector import HorseDetector

app = Flask(__name__)
model = HorseDetector()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'image not found'})
    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    probability = model.predict(image)
    return jsonify({'probability': probability})
