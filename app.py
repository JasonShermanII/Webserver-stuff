from flask import Flask, request
import torch
from PIL import Image
import io
import os

app = Flask(__name__)

# Cache model and avoid re-downloading on every deploy
if not os.path.exists('yolov5s.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # lightweight model
    model.save('yolov5s.pt')  # save model to disk
else:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', model='yolov5s.pt')  # load from disk

@app.route('/upload', methods=['POST'])
def upload():
    image_bytes = request.data
    img = Image.open(io.BytesIO(image_bytes))
    results = model(img)
    return results.pandas().xyxy[0].to_json()  # send prediction as JSON

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use dynamic port from Railway
    app.run(host='0.0.0.0', port=port)
