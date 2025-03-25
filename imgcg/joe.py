from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Orthogonal

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')

# Set the folder for uploading images
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create the uploads folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the pre-trained VGG16 model and restructure it to remove the fully connected layer
vgg_model = VGG16()  # Load the VGG16 model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)  # Remove the final dense layer

# Load the pre-trained image caption generator model and tokenizer
model = load_model('ImgCG.h5', custom_objects={'Orthogonal': Orthogonal}, compile=False)

# Load the tokenizer used for encoding captions
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set the maximum length of captions (same as used during training)
max_length = 35


# Helper function to convert an integer index to a word using the tokenizer
def idx_to_word(integer, tokenizer):
    """Converts an integer index to the corresponding word in the tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Function to generate a caption for an image
def predict_caption(model, image, tokenizer, max_length):
    """Generates a caption for the input image using the trained model."""
    in_text = 'startseq'  # Start the sequence with 'startseq'

    # Generate words one by one until reaching the maximum length or 'endseq'
    for _ in range(max_length):
        # Convert the current text sequence to a sequence of integers
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # Pad the sequence to ensure it's the same length as max_length
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Predict the next word in the sequence
        y_hat = model.predict([image, sequence], verbose=0)

        # Get the index of the predicted word with the highest probability
        y_hat = np.argmax(y_hat)

        # Convert the predicted index to a word
        word = idx_to_word(y_hat, tokenizer)

        # If no word is found (index out of range), break the loop
        if word is None:
            break

        # Add the predicted word to the input sequence
        in_text += " " + word

        # Stop if the 'endseq' token is predicted
        if word == "endseq":
            break

    # Return the generated caption, removing the 'startseq' and 'endseq' tags
    return in_text[8:-7]


# Route to render the main upload page
@app.route('/')
def index():
    """Renders the homepage where users can upload an image."""
    return render_template('image.html')


# Route to handle image upload and caption generation
@app.route('/predict', methods=['POST'])
def predict():
    """Handles the uploaded image, generates the caption, and displays the result."""

    # Check if the file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the user submitted an empty form
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file to the uploads directory
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and preprocess the uploaded image
        image = load_img(file_path, target_size=(224, 224))  # Resize image to 224x224 (VGG16 input size)
        image = img_to_array(image)  # Convert image to array
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # Reshape for the model
        image = preprocess_input(image)  # Normalize the image for VGG16

        # Extract image features using the VGG16 model
        feature = vgg_model.predict(image, verbose=0)

        # Generate a caption for the image using the trained caption generator model
        caption = predict_caption(model, feature, tokenizer, max_length)

        # Render the result: show the uploaded image and the generated caption
        return render_template('image.html', caption=caption, image_path="uploads/" + filename)


# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded image file from the uploads directory."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Main entry point: run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
