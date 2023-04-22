import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model import UNet

def extract(v, t, x_shape):

    '''
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    '''
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        # Register betas as a buffer so that they are saved with the model
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        
        alphas = 1. - self.betas
        # print("alphas:", alphas)
        alphas_bar = torch.cumprod(alphas, dim=0)

        # caculation for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
    def forward(self, x_0):
        '''
        Algorithm 1: Diffusion process
        '''
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device) # t is a random integer in [0, T-1], size is [batch_size, ]
        
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        # print("extract(self.sqrt_alphas_bar, t, x_0.shape):", extract(self.sqrt_alphas_bar, t, x_0.shape).shape)
        # print("x_0:", x_0.shape)
        # print("extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0:", (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0).shape)

        # print('x_t', x_t.shape)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas_bar))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar_prev))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t , eps):
        assert x_t.shape == eps.shape

        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )
    
    def p_mean_var(self, x_t, t):
        var = torch.cat(self.posterior_var[1:2], self.betas[1:2])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        x_t_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)

        return x_t_prev_mean, var
    
    def forward(self, x_T):
        '''
        Algorithm 2: Sampling from the diffusion process
        '''
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print("time_step:", time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_var(x_t, t)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0

        x_0 = x_t
        return torch.clip(x_0, -1, 1)



if __name__ == '__main__':
    x = torch.randn(3, 32, 32)
    # t = torch.randint(1000, (batch_size, ))

    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    
    trainer = GaussianDiffusionTrainer(model, 0.1, 0.9, 1000)
    y = trainer(x)
    