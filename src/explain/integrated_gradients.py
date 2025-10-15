from captum.attr import IntegratedGradients
import torch


class ViTIntegratedGradients:
    def __init__(self, model, preprocess):
        self.model = model.eval()
        self.preprocess = preprocess
        self.ig = IntegratedGradients(self.model)
        self.device = next(model.parameters()).device

    def attribute(self, pil_img, target_class=None, baseline=None, n_steps=50):
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x).to(self.device)
        attr = self.ig.attribute(x, baselines=baseline, target=target_class, n_steps=n_steps)
        saliency = attr.squeeze(0).sum(0).cpu().detach().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max()-saliency.min()+1e-8)
        return saliency