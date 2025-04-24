
import streamlit as st
from PIL import Image
import torch
import numpy as np
from model import QRC_UNet
from utils import preprocess_image, postprocess_mask

st.set_page_config(page_title="Lung CT Segmentation", layout="wide")

# --- 🧑‍🎨 CUSTOM CSS ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #e3f2fd, #f8f9fa);
        }
        .big-title {
            text-align: center;
            padding: 10px;
            border-radius: 12px;
            background-color: #1565c0;
            color: white;
            font-size: 30px;
            font-weight: bold;
        }
        .footer {
            font-size: 13px;
            text-align: center;
            color: #888;
            padding-top: 30px;
        }
        .card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🫁 Lung CT Scan Segmentation (QRC-U-Net)</div>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = QRC_UNet()
    model.load_state_dict(torch.load("qrc_unet_trained.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

with st.expander("ℹ️ What does this app do?", expanded=False):
    st.markdown("""
    This application uses a lightweight QRC-U-Net model to segment lung nodules from CT scan images.  
    **Upload a CT scan**, and the model will return:  
    - The **binary segmentation mask**  
    - An **overlay on the original image**  
    - A **confidence score**  
    """)

uploaded_file = st.file_uploader("📤 Upload a Lung CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((200, 200))
    st.image(resized_image, caption="🖼️ Uploaded Image", use_column_width=False)

    input_tensor = preprocess_image(image)
    with st.spinner("🔍 Segmenting..."):
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.sigmoid(output)
            binary_mask = postprocess_mask(pred_mask)

    binary_mask_resized = Image.fromarray(binary_mask).resize((200, 200))
    binary_mask_resized_np = np.array(binary_mask_resized)

    st.subheader("📌 Segmentation Mask")
    st.image(binary_mask_resized_np, caption="Detected Region", use_column_width=True)

    overlay = np.array(image.resize((200, 200))).copy()
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], binary_mask_resized_np)

    st.subheader("🩻 Overlayed Output")
    st.image(overlay, caption="Original + Mask", use_column_width=True)

    confidence = pred_mask.mean().item()
    st.subheader("🧠 Prediction Confidence")
    st.progress(min(confidence, 1.0))
    st.success(f"Confidence Score: {confidence * 100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Upload Another Image"):
        st.experimental_rerun()

# --- Footer ---
st.markdown("<div class='footer'>Built with ❤️ using QRC-U-Net • Streamlit • PyTorch</div>", unsafe_allow_html=True)
