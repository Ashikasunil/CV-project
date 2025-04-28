
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import load_model, preprocess_image, predict

# Load the trained model
model_path = 'qrc_unet_trained.pth'  # Update this path if different
model = load_model(qrc_unet_trained.pth)

# Hard-coded metrics (replace with your actual values)
TRAIN_LOSS = 0.05
TRAIN_DICE = 0.91
VAL_LOSS = 0.07
VAL_DICE = 0.89
TEST_LOSS = 0.06
TEST_DICE = 0.90
IOU_SCORE = 0.88
PRECISION = 0.92
RECALL = 0.89

# App title
st.title("Lung CT Malignant Nodule Segmentation - QRC-UNet")
st.write("Upload a CT scan image and its ground truth mask to see segmentation results.")

# File Upload
uploaded_image = st.file_uploader("Upload CT Scan Image", type=['png', 'jpg', 'jpeg'])
uploaded_mask = st.file_uploader("Upload Ground Truth Mask", type=['png', 'jpg', 'jpeg'])

if uploaded_image and uploaded_mask:
    # Load Images
    input_image = Image.open(uploaded_image).convert('RGB')
    ground_truth = Image.open(uploaded_mask).convert('L')

    # Preprocess input image
    input_tensor = preprocess_image(input_image)

    # Predict
    prediction = predict(model, input_tensor)

    # Display Images Side-by-Side
    st.subheader("Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(input_image, caption="Input CT Image", use_column_width=True)

    with col2:
        st.image(ground_truth, caption="Ground Truth Mask", use_column_width=True)

    with col3:
        plt.figure(figsize=(3,3))
        plt.imshow(prediction, cmap='gray')
        plt.axis('off')
        st.pyplot(plt)
        plt.close()

    # Display Metrics
    st.subheader("Model Metrics")
    st.markdown("### Training and Validation")
    st.metric("Train Loss", f"{TRAIN_LOSS:.4f}")
    st.metric("Train Dice Score", f"{TRAIN_DICE:.4f}")
    st.metric("Validation Loss", f"{VAL_LOSS:.4f}")
    st.metric("Validation Dice Score", f"{VAL_DICE:.4f}")

    st.markdown("### Testing")
    st.metric("Test Loss", f"{TEST_LOSS:.4f}")
    st.metric("Test Dice Score", f"{TEST_DICE:.4f}")

    st.markdown("### Other Evaluation Metrics")
    st.metric("IoU Score", f"{IOU_SCORE:.4f}")
    st.metric("Precision", f"{PRECISION:.4f}")
    st.metric("Recall", f"{RECALL:.4f}")

else:
    st.warning("Please upload both CT scan image and ground truth mask.")
