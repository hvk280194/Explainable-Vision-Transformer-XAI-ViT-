import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


class ViTGradCAM:
    def __init__(self, model, target_layer_name='backbone.blocks'):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        # We'll hook into the last encoder block's normed token outputs
        # Most robust approach: register a forward and backward hook on the penultimate layer
        # We'll try to find the last block
        last_block = model.backbone.blocks[-1]
        self.hook_handles = []
    def forward_hook(module, inp, out):
    # out shape: B, N, C (for token embeddings)
        self.activations = out.detach()
    def backward_hook(module, grad_in, grad_out):
    # grad_out has same shape as out
        self.gradients = grad_out[0].detach()
        self.hook_handles.append(last_block.register_forward_hook(forward_hook))
        self.hook_handles.append(last_block.register_backward_hook(backward_hook))


    def generate(self, pil_img, preprocess, target_class=None):
        x = preprocess(pil_img).unsqueeze(0)
        x = x.to(next(self.model.parameters()).device)
        logits = self.model(x)
        if target_class is None:
        target_class = logits.argmax(dim=1).item()
        loss = logits[0, target_class]
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        grads = self.gradients # B, N, C
        acts = self.activations # B, N, C
        weights = grads.mean(dim=-1, keepdim=True) # B, N, 1
        cam_token = (weights * acts).sum(dim=-2) # aggregate across tokens -> B, C
        # map token attribution back to image patches
        # This is a simplistic approach; for ViT with patch size p, we can reshape tokens into grid
        num_tokens = acts.shape[1]
        grid_size = int(np.sqrt(num_tokens - 1))
        token_attributions = (weights.squeeze(-1)).cpu().numpy()[0,1:] # skip cls token
        heatmap = token_attributions.reshape(grid_size, grid_size)
        heatmap = cv2.resize(heatmap, (pil_img.size[0], pil_img.size[1]))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap


def clear(self):
    for h in self.hook_handles:
        h.remove()