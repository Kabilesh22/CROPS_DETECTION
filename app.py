from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('plant_classification_model.h5')

# Define the target image size
target_size = (224, 224)

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values
    return img_array

# Function to make predictions
def predict(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    classes = ['corn', 'mango', 'potato', 'rice', 'sugarcane', 'tomato']
    predicted_class = classes[class_index]
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image_path = 'temp_image.jpg'
        file.save(image_path)
        prediction = predict(image_path)
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
