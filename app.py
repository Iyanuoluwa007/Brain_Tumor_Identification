import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Load trained model
class BrainTumorCNN(torch.nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16*16*128, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load("brain_tumor_cnn.pth", map_location=device))
model.eval()

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

class_names = ['glioma','meningioma','notumor','pituitary']

# Streamlit UI
st.title("🧠 Brain Tumor Classification (MRI)")
st.write("Upload an MRI image and the model will predict tumor type.")

uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy().flatten()

    # Display uploaded image
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Top prediction
    top_idx = np.argmax(probs)
    st.subheader(f"Prediction: {class_names[top_idx]} ({probs[top_idx]*100:.2f}%)")

    # Confidence bar chart
    st.bar_chart({class_names[i]: probs[i] for i in range(len(class_names))})

    # Show example images of predicted class (if dataset is local)
    example_dir = os.path.join(r"C:\Users\okeiy\Downloads\Brain_Tumor\Training", class_names[top_idx])
    if os.path.exists(example_dir):
        st.write("Example images of predicted class:")
        example_imgs = os.listdir(example_dir)[:3]
        st.image([os.path.join(example_dir, img) for img in example_imgs], width=150)
