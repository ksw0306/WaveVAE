import torch
from torch import nn
from torch.nn import functional as F
from wavenet_iaf import Wavenet_Student
from wavenet import Wavenet
from torch.distributions.normal import Normal
from math import log, pi


class WaveVAE(nn.Module):
    def __init__(self):
        super(WaveVAE, self).__init__()

        self.encoder = Wavenet(out_channels=2,
                               num_blocks=2,
                               num_layers=10,
                               residual_channels=128,
                               gate_channels=256,
                               skip_channels=128,
                               kernel_size=2,
                               cin_channels=80,
                               upsample_scales=[16, 16])
        self.decoder = Wavenet_Student(num_blocks_student=[1, 1, 1, 1, 1, 1],
                                       num_layers=10)
        self.log_eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, c):
        # Encode
        mu_logs = self.encoder(x, c)
        mu = mu_logs[:, 0:1, :-1]
        logs = mu_logs[:, 1:, :-1]
        q_0 = Normal(mu.new_zeros(mu.size()), mu.new_ones(mu.size()))

        mean_q = (x[:, :, 1:] - mu) * torch.exp(-logs)

        # Reparameterization, Sampling from prior
        z = q_0.sample() * torch.exp(self.log_eps) + mean_q
        z_prior = q_0.sample()

        z = F.pad(z, pad=(1, 0), mode='constant', value=0)
        z_prior = F.pad(z_prior, pad=(1, 0), mode='constant', value=0)
        c_up = self.encoder.upsample(c)

        # Decode
        # x_rec : [B, 1, T] (first time step zero-padded)
        # mu_tot, logs_tot : [B, 1, T-1]
        x_rec, mu_p, log_p = self.decoder(z, c_up)
        x_prior = self.decoder.generate(z_prior, c_up)

        loss_recon = -0.5 * (- log(2.0 * pi) - 2. * log_p - torch.pow(x[:, :, 1:] - mu_p, 2) * torch.exp((-2.0 * log_p)))
        loss_kl = 0.5 * (mean_q ** 2 + torch.exp(self.log_eps) ** 2 - 1) - self.log_eps
        return x_rec, x_prior, loss_recon.mean(), loss_kl.mean()

    def generate(self, z, c):
        c_up = self.encoder.upsample(c)
        x_sample = self.decoder.generate(z, c_up)
        return x_sample
    
    def remove_weight_norm(self):
        self.encoder.remove_weight_norm()
        self.decoder.remove_weight_norm()


