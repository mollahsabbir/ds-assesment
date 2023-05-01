import os
import io
import torch

import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

from utils import calculate_mean_std
from train import train as train_model
from model import Conv2Model, Conv3Model, EnsembleModel

app = Flask(__name__)

MODEL_PATH = 'inference_model/model.pt'
CLASS_NAMES = ['berry', 'bird', 'dog', 'flower']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(MODEL_PATH, map_location=device)
model.eval()

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor([0.4973, 0.4484, 0.3651]),
                            std=torch.tensor([0.2267, 0.2191, 0.2191]))
    ])

@app.route('/train', methods=['POST'])
def train():
    ensemble = EnsembleModel(Conv2Model(), Conv3Model())
    train_model(ensemble)
    # Todo: Implement model saving mechanism
    return jsonify({'message': 'Training complete'})

@app.route('/infer', methods=['POST'])
def infer():

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)

    class_name = CLASS_NAMES[predicted[0]]

    return jsonify({'class': class_name})

if __name__ == '__main__':
    app.run(debug=True)