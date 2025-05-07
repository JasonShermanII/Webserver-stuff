import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return "ESP32-CAM Waste Classifier Server is Running!"

@app.route("/classify", methods=["POST"])
def classify():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    
    # TODO: Replace this with actual model prediction
    # image_file.read() would give you the raw bytes
    print("Received image:", image_file.filename)
    prediction = "biodegradable"  # Dummy response

    return jsonify({"classification": prediction})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port)
