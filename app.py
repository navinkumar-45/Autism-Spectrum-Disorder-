from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model_path = 'autism.h5'
model = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to match the input format of the model
    return img_array

def create_bar_chart(predictions, class_names, chart_path):
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, predictions[0], color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Prediction Confidence')
    plt.savefig(chart_path)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Prepare the image and make a prediction
        prepared_image = prepare_image(file_path, target_size=(224, 224))
        predictions = model.predict(prepared_image)
        predicted_class_index = np.argmax(predictions, axis=1)

        class_names = ['Autistic', 'Non_Autistic']
        predicted_class_name = class_names[predicted_class_index[0]]

        # Create a bar chart for the predictions
        chart_path = os.path.join('static', 'prediction_chart.png')
        create_bar_chart(predictions, class_names, chart_path)

        return render_template('result.html', predicted_class=predicted_class_name, predictions=predictions, chart_url=chart_path)
    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
