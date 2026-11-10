"""Objectives with explicit event-dimension sums and batch averages."""
import math

import torch
import torch.nn.functional as F


def diagonal_gaussian_kl(mean, log_variance):
    """KL(q || N(0,I)), one value per example, summing latent dimensions."""
    return 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(-1)


def gaussian_nll(target, mean, variance=1.0):
    """Fixed isotropic Gaussian decoder NLL, including its normalizer."""
    if variance <= 0:
        raise ValueError("variance must be positive")
    return 0.5 * ((target - mean).square() / variance + math.log(2 * math.pi * variance)).sum(-1)


def scaled_attention(query, key, value, causal=False):
    """Single-head attention over (..., sequence, feature); no padding mask."""
    scores = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    if causal:
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("causal self-attention requires equal sequence lengths")
        mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, -torch.inf)
    weights = scores.softmax(dim=-1)
    return weights @ value, weights


def flow_path(noise, data, time):
    """Independent coupling, noise at t=0 and data at t=1."""
    return (1 - time) * noise + time * data, data - noise


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    """Inputs are completion log probabilities (summed, not length averaged)."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = policy_chosen - policy_rejected - reference_chosen + reference_rejected
    return -F.logsigmoid(beta * margin).mean()
