import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.model.vit_model import ViTClassifier
from src.explain.gradcam_vit import ViTGradCAM
from src.explain.integrated_gradients import ViTIntegratedGradients


st.title('🧠 XAI-ViT — Explainable Vision Transformer')


uploaded = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
method = st.selectbox('Explanation method', ['Grad-CAM', 'Integrated Gradients'])


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTClassifier(pretrained=True).to(device)
    return model


model = load_model()
preprocess = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])


if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded image', use_column_width=True)
if method == 'Grad-CAM':
    expl = ViTGradCAM(model)
    heatmap = expl.generate(img, preprocess)
    st.image(heatmap, caption='Grad-CAM Heatmap', use_column_width=True)
else:
    expl = ViTIntegratedGradients(model, preprocess)
    sal = expl.attribute(img)
    st.image(sal, caption='Integrated Gradients', use_column_width=True)