import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import json

# Streamlit page configuration
st.set_page_config(page_title="Waste Type Classification", page_icon="♻️", layout="centered")

st.title("♻️ Waste Type Classification – Smart Recycling Demo")
st.write("Upload an image of waste, and the model will classify it into one of the following categories:")
st.markdown("**Plastic | Glass | Paper | Metal | Organic**")

MODEL_PATH = "models/waste_classifier.h5"
LABELS_PATH = "models/labels.json"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please train the model first using `train_model.py`.")
        return None, None
    model = load_model(MODEL_PATH)
    labels = None
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)
    return model, labels

model, labels = load_trained_model()

uploaded_file = st.file_uploader("📁 Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is not None:
        # Preprocess the image
        img_resized = image.resize((128, 128))  # same as training img size
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        class_name = labels.get(str(class_idx), f"Class {class_idx}") if labels else f"Class {class_idx}"

        # Display result
        st.subheader(f"🧠 Predicted Category: **{class_name.capitalize()}**")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
    else:
        st.info("⚙️ Please train your model first before running the demo.")
else:
    st.info("👆 Upload an image to begin.")
