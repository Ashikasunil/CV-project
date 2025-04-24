
import streamlit as st
from PIL import Image
import torch
import numpy as np
from model import QRC_UNet
from utils import preprocess_image, postprocess_mask

st.set_page_config(page_title="Lung CT Segmentation", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🫁 Lung CT Scan Segmentation</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = QRC_UNet()
    model.load_state_dict(torch.load("qrc_unet_trained.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    - Upload a lung CT image.
    - The model will segment malignant nodules in real-time.
    - You’ll see a binary mask and an overlay on the original image.
    """)

uploaded_file = st.file_uploader("📤 Upload Lung CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((200, 200))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(resized_image, caption="🖼️ Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image)
    with st.spinner("🔍 Segmenting... Please wait..."):
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.sigmoid(output)
            binary_mask = postprocess_mask(pred_mask)

    binary_mask_resized = Image.fromarray(binary_mask).resize((256, 256))
    binary_mask_resized_np = np.array(binary_mask_resized)

    with col2:
        st.image(binary_mask_resized_np, caption="📌 Segmentation Mask", use_column_width=True)

    overlay = np.array(image.resize((256, 256))).copy()
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], binary_mask_resized_np)

    st.subheader("🔬 Overlay Visualization")
    st.image(overlay, use_column_width=True, caption="🩻 Image + Predicted Mask")

    confidence = pred_mask.mean().item()
    st.subheader("🧠 Prediction Confidence")
    st.progress(min(confidence, 1.0))
    st.success(f"Confidence Score: {confidence * 100:.2f}%")
