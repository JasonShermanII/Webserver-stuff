from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load the YOLOv5 model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # lightweight model

@app.route('/')
def home():
    return render_template('index.html')  # Serve the frontend form

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['file']
    img = Image.open(image.stream)
    
    # Get model predictions
    results = model(img)
    
    # Return results as JSON
    return jsonify(results.pandas().xyxy[0].to_dict(orient="records"))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
