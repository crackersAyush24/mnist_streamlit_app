import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps

# Load your trained model
model = load_model("classification_model.keras")

st.title("🖼️ MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9).")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- Preprocess the image ---
    # 1. Convert to grayscale
    image = image.convert("L")

    # 2. Invert image if background is white
    image = ImageOps.invert(image)

    # 3. Resize to 28x28
    image = image.resize((28, 28))

    # 4. Convert to numpy array
    img_array = img_to_array(image)

    # 5. Flatten and normalize
    img_array = img_array.reshape(1, 784).astype("float32") / 255.0

    # --- Predict ---
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"### 🔮 Predicted Digit: {predicted_class}")
