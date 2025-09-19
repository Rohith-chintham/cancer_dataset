import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cancer_model.h5")

model = load_model()

st.set_page_config(page_title="Cancer Detection AI", page_icon="🩺")
st.title("🩺 Cancer Detection App")
st.write("Upload a medical image (X-ray or biopsy) and the AI will predict if it is **benign** or **malignant**.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    st.subheader("🔍 Prediction Result:")
    if prediction > 0.5:
        st.error("⚠️ Malignant (Cancer Detected)")
    else:
        st.success("✅ Benign (No Cancer Detected)")
