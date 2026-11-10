"""Tiny networks whose objectives can be inspected in a live class."""
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, latent_dim=2, width=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, 2 * latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, width), nn.Tanh(), nn.Linear(width, 2))

    def forward(self, data):
        mean, log_variance = self.encoder(data).chunk(2, dim=-1)
        z = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
        return self.decoder(z), mean, log_variance


class Velocity(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2))

    def forward(self, points, time):
        return self.net(torch.cat((points, time), dim=-1))


@torch.no_grad()
def euler_sample(model, noise, steps=50):
    if steps < 1:
        raise ValueError("steps must be positive")
    points = noise.clone()
    for step in range(steps):
        time = points.new_full((len(points), 1), step / steps)
        points = points + model(points, time) / steps
    return points
