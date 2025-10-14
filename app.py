import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import time

# -------------------- CONFIG --------------------
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢", layout="centered")

# Load model once and cache it
@st.cache_resource
def load_digit_model():
    return load_model("classification_model.keras")

model = load_digit_model()

# -------------------- APP UI --------------------
st.title("🧠 AI-Powered MNIST Digit Classifier")
st.markdown("""
Upload an image of a **handwritten digit (0–9)**, and this AI model will predict the number!  
Try writing a number on paper, take a photo, and upload it here. ✨
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing the digit..."):
        time.sleep(1.2)  # simulate processing delay

        # --- Preprocessing ---
        img = image.convert("L")  # grayscale
        img = ImageOps.invert(img)  # invert if white background
        img = img.resize((28, 28))
        img_array = img_to_array(img)
        img_array = img_array.reshape(1, 784).astype("float32") / 255.0
