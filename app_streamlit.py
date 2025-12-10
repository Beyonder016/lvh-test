import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from model.model_loader import load_model
from utils.preprocessing import preprocess_image
from utils.gradcam import get_gradcam  # make sure gradcam.py exposes this function

st.set_page_config(page_title="LVH Detection via Chest X-Ray", layout="centered")

st.title("💓 LVH Detection from Chest X-Ray (Grad-CAM Visualization)")
st.markdown("Upload a chest X-ray image to detect Left Ventricular Hypertrophy (LVH).")

# --- Load model once ---
@st.cache_resource
def load_my_model():
    model = load_model()
    model.eval()
    return model

model = load_my_model()

# --- Upload Section ---
uploaded = st.file_uploader("Upload X-Ray Image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = preprocess_image(img)  # must return torch tensor with batch dim
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    result = "🔴 LVH Detected" if prob > 0.5 else "🟢 Normal"
    st.markdown(f"### **Prediction:** {result}")
    st.markdown(f"**Confidence:** {prob:.2f}")

    # --- GradCAM Button ---
    if st.button("Generate Grad-CAM Heatmap"):
        cam_image = get_gradcam(model, input_tensor, target_layer=None)  # adjust target layer
        st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
