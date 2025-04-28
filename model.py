
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np

# Assuming you have defined QRC_UNet model in your Colab, I'll mock a basic load function
# Replace this with your actual QRC-UNet class if needed.

class QRC_UNet(nn.Module):
    def __init__(self):
        super(QRC_UNet, self).__init__()
        # Initialize your MobileViT encoder and custom decoder blocks here
        pass

    def forward(self, x):
        # Define the forward pass
        return x

def load_model(model_path):
    model = QRC_UNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(uploaded_image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(uploaded_image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
    return output.squeeze(0).squeeze(0).numpy()  # Remove batch & channel dims
