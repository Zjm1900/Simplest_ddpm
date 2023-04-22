import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def extract(v, t, x_shape):

    device = v.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view(t.shape[0], [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer(
            "betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            "sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_bar", torch.sqrt(1 - alphas_bar))
        
