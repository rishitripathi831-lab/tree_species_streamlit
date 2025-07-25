import streamlit as st
import numpy as np
import requests
import zipfile
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO

# Google Drive ZIP File ID (edit this)
FILE_ID = "1aOykMRB2qlUizJKEfqAxQGIAoSKnzmth"  # Example: '1ABCD12345efghXYZ'
ZIP_FILENAME = 'improved_model.zip'
MODEL_FILENAME = 'improved_cnn_model.h5'

# Function to download the ZIP file
def download_zip():
    if not os.path.exists(improved_model.zip):
        url = "https://drive.google.com/uc?export=download&id=1aOykMRB2qlUizJKEfqAxQGIAoSKnzmth"
        response = requests.get(url)
        with open(improved_model.zip, 'wb') as f:
            f.write(response.content)

# Function to unzip the model
def unzip_model("improved_model.zip"):
    if not os.path.exists(improved_cnn_model.h5):
        with zipfile.ZipFile(improved_model.zip, 'r') as zip_ref:
            zip_ref.extractall()

# Download and load the model
@st.cache_resource
def load_model():
    download_zip(1aOykMRB2qlUizJKEfqAxQGIAoSKnzmth)
    unzip_model()
    model = tf.keras.models.load_model(improved_cnn_model.h5)
    return model

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Load model
model = load_model()

# Streamlit UI
st.title("🌳 Tree Species Classification")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert
