from captum.attr import IntegratedGradients
import torch


class ViTIntegratedGradients:
def __init__(self, model, preprocess):
self.model = model.eval()
self.preprocess = preprocess
self.ig = IntegratedGradients(self.model)


def attribute(self, pil_img, target=None, baseline=None, n_steps=50):
x = self.preprocess(pil_img).unsqueeze(0)
if baseline is None:
baseline = torch.zeros_like(x)
attr = self.ig.attribute(x, baselines=baseline, target=target, n_steps=n_steps)
# aggregate across channels
saliency = attr.squeeze(0).sum(0).cpu().detach().numpy()
# normalize
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
return saliency