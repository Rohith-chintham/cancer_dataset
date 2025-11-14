import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cancer_model.h5")

model = load_model()

# App Config
st.set_page_config(page_title="Cancer Detection AI", page_icon="🩺", layout="wide")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    🩺 **AI Cancer Detection App**  
    Upload medical images (X-ray or biopsy) and let the AI predict whether they are:
    - ✅ **Benign (No Cancer)**
    - ⚠️ **Malignant (Cancer Detected)**
    
    Built with **TensorFlow + Streamlit**
    """
)

st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Prediction Confidence Threshold", 0.1, 0.9, 0.5)

# Main UI
st.title("🩺 AI Cancer Detection System")
st.write("Upload one or more medical images for cancer classification.")

# File uploader
uploaded_files = st.file_uploader("📤 Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Open image
        image = Image.open(uploaded_file).convert("RGB")
        img_resized = image.resize((150, 150))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        label = "Malignant" if prediction > confidence_threshold else "Benign"
        confidence = float(prediction) if prediction > confidence_threshold else 1 - float(prediction)

        # Show result for each image
        st.image(image, caption=f"🖼️ {uploaded_file.name}", width=250)
        st.markdown(f"**Prediction:** {label}")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence:.2f}**")

        # Store results
        results.append({"Filename": uploaded_file.name, "Prediction": label, "Confidence": round(confidence, 2)})

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Display results table
    st.subheader("📊 Results Summary")
    st.dataframe(df)

    # Download option
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="cancer_predictions.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("⚡ Developed with ❤️ using Streamlit & TensorFlow")
