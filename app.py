from flask import Flask, request
import torch
from PIL import Image
import io

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # lightweight model

@app.route('/upload', methods=['POST'])
def upload():
    image_bytes = request.data
    img = Image.open(io.BytesIO(image_bytes))
    results = model(img)
    return results.pandas().xyxy[0].to_json()  # send prediction as JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
