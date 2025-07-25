import streamlit as st
import numpy as np
import requests
import zipfile
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO

# Constants
FILE_ID = "1aOykMRB2qlUizJKEfqAxQGIAoSKnzmth"
ZIP_FILENAME = "improved_model.zip"
MODEL_FILENAME = "improved_cnn_model.h5"

# Function to download the ZIP file from Google Drive
def download_zip():
    if not os.path.exists(ZIP_FILENAME):
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        response = requests.get(url)
        with open(ZIP_FILENAME, 'wb') as f:
            f.write(response.content)

# Function to unzip the model file
def unzip_model():
    if not os.path.exists(MODEL_FILENAME):
        with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
            zip_ref.extractall()

# Function to load the model with caching
@st.cache_resource
def load_model():
    download_zip()
    unzip_model()
    model = tf.keras.models.load_model(MODEL_FILENAME)
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Tree species labels
labels = [
    "amla", "Asopalav", "babul", "bamboo", "banayan", "bili", "cactus",
    "champa", "coconut", "garmalo", "gulmohar", "gunda", "jamun", "kanchan",
    "kesudo", "khajur", "Mango", "motichanoti", "neem", "nilgiri", "Other",
    "Pilikaren", "pipal", "saptaparni", "shirish", "Simlo", "sitafal",
    "sonmahor", "sugarcane", "Vad"
]

# Load the model
with st.spinner("Loading model..."):
    model = load_model()

# Streamlit UI
st.title("🌳 Tree Species Classification")
uploaded_file = st.file_uploader("Upload an image of a tree leaf or plant...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = labels[np.argmax(prediction)]

    st.success(f"🌿 Predicted Tree Species: *{predicted_class}*")
