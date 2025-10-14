
# XAI-ViT — Explainable Vision Transformer


This repository contains an educational project demonstrating explainability techniques for Vision Transformers on CIFAR-10.


## Features
- Train or fine-tune a small Vision Transformer on CIFAR-10.
- Explanations: Attention Rollout, Grad-CAM for ViT patch embeddings, Integrated Gradients (Captum).
- Streamlit demo to upload images and interactively view explanations.


## Quickstart
1. Create a virtual environment and install dependencies:


```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
