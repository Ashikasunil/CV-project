
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

class QuantumFourierConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        real = self.conv1(x)
        imag = self.conv2(x)
        return torch.sqrt(real**2 + imag**2)

class ResidualCapsuleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))) + self.skip(x))

class ADSCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return F.relu(self.pointwise(self.depthwise(x)))

class QRC_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('mobilevit_xxs', pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()
        self.qfc = QuantumFourierConv(enc_channels[-1])
        self.rescaps = ResidualCapsuleBlock(enc_channels[-1], 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.adsc1 = ADSCBlock(128 + enc_channels[3], 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.adsc2 = ADSCBlock(64 + enc_channels[2], 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.adsc3 = ADSCBlock(32 + enc_channels[1], 32)
        self.up4 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.adsc4 = ADSCBlock(16 + enc_channels[0], 16)
        self.final_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        x = self.qfc(e5)
        x = self.rescaps(x)
        x = self.adsc1(torch.cat([self.up1(x), e4], dim=1))
        x = self.adsc2(torch.cat([self.up2(x), e3], dim=1))
        x = self.adsc3(torch.cat([self.up3(x), e2], dim=1))
        x = self.adsc4(torch.cat([self.up4(x), e1], dim=1))
        return torch.sigmoid(self.final_conv(x))

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

@st.cache_resource
def load_model():
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
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (pred_mask > 0.5).astype(np.uint8)

        # ✅ Resize mask before applying overlay
        resized_mask = Image.fromarray(mask * 255).resize(image.size)
        mask_resized_np = np.array(resized_mask) // 255

        overlay = np.array(image).copy()
        overlay[mask_resized_np.astype(bool)] = [255, 0, 0]

# Create overlay
overlay = np.array(image).copy()
overlay[mask_resized_np.astype(bool)] = [255, 0, 0]

    col1, col2, col3 = st.columns(3)
    col1.image(image, caption="🖼️ Original", use_column_width=True)
    col2.image(resized_mask, caption="📌 Mask", use_column_width=True)
    col3.image(overlay, caption="📊 Overlay", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built with ❤️ using QRC-U-Net • Streamlit • PyTorch</div>", unsafe_allow_html=True)
