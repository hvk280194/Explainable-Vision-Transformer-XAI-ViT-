import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


class ViTGradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        
        last_block = model.blocks[-1]
        self.hooks = []
        
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        self.hooks.append(last_block.register_forward_hook(forward_hook))
        self.hooks.append(last_block.register_backward_hook(backward_hook))
    
    def generate_cam(self, pil_img, preprocess, target_class=None):
        device = next(self.model.parameters()).device
        x = preprocess(pil_img).unsqueeze(0).to(device)
        logits = self.model(x)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        self.model.zero_grad()
        loss = logits[0, target_class]
        loss.backward(retain_graph=True)
        
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=-1)
        weights_np = weights[0].cpu().numpy()[1:]
        n_patches = acts.shape[1] - 1
        grid_size = int(np.sqrt(n_patches))
        cam = weights_np.reshape(grid_size, grid_size)
        cam_resized = cv2.resize(cam, pil_img.size)
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max()-cam_resized.min()+1e-8)
        return cam_norm


def clear(self):
    for h in self.hook_handles:
        h.remove()