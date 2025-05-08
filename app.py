from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model (ensure it's in your 'model/' directory or another location)
MODEL_PATH = 'model/waste_classifier_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Index page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and classify route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image
        image = Image.open(filepath).resize((224, 224))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        predictions = model.predict(image_array)

        classes = ['Biodegradable', 'Non-Biodegradable']
        predicted_class = classes[int(np.argmax(predictions))]

        return render_template('index.html', prediction=predicted_class, filename=filename)

    return redirect(url_for('index'))

# To serve uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    # Use the PORT environment variable if available (Render uses this)
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if not set
    app.run(host='0.0.0.0', port=port)
