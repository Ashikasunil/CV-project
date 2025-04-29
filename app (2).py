
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="Lung Module Segmentation", layout="wide")
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

st.markdown("<div class='big-title'>🫁 Lung Module Segmentation (QRC-U-Net)</div>", unsafe_allow_html=True)

# ... (same model classes as before) ...
# For brevity, we'll omit model definitions in this uploadable version

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

@st.cache_resource
def load_model():
    from model import QRC_UNet
    model = QRC_UNet()
    model.load_state_dict(torch.load("qrc_unet_trained (1).pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("📤 Upload a Lung CT Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess_image(image)

    with st.spinner("🔍 Segmenting..."):
        try:
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                mask = (pred_mask > 0.5).astype(np.uint8)

                resized_mask = Image.fromarray(mask * 255).resize(image.size)
                mask_resized_np = np.array(resized_mask) // 255

                overlay = np.array(image).copy()
                overlay[mask_resized_np.astype(bool)] = [255, 0, 0]
        except Exception as e:
            st.error("❌ Model prediction failed.")
            st.code(str(e))
            st.write("Input tensor shape:", input_tensor.shape)
            st.stop()

    col1, col2, col3 = st.columns(3)
    col1.image(image, caption="🖼️ Original", use_column_width=True)
    col2.image(resized_mask, caption="📌 Mask", use_column_width=True)
    col3.image(overlay, caption="📊 Overlay", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built with ❤️ using QRC-U-Net • Streamlit • PyTorch</div>", unsafe_allow_html=True)
