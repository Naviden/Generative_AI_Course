import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from genai_course.objectives import diagonal_gaussian_kl, dpo_loss, flow_path, gaussian_nll, scaled_attention
from genai_course.models import VAE, euler_sample


def test_kl_matches_distribution_reference():
    mean = torch.tensor([[1., 2.], [-0.3, 0.7]], dtype=torch.float64)
    logvar = torch.tensor([[0., 0.], [0.2, -0.8]], dtype=torch.float64)
    q = torch.distributions.Normal(mean, (logvar / 2).exp())
    p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    expected = torch.distributions.kl_divergence(q, p).sum(-1)
    torch.testing.assert_close(diagonal_gaussian_kl(mean, logvar), expected)
    assert expected[0] == 2.5


def test_kl_sums_latent_dimensions():
    mean, logvar = torch.ones(3, 2), torch.zeros(3, 2)
    torch.testing.assert_close(diagonal_gaussian_kl(mean.repeat(1, 2), logvar.repeat(1, 2)), 2 * diagonal_gaussian_kl(mean, logvar))
    assert diagonal_gaussian_kl(torch.zeros(2, 4), torch.zeros(2, 4)).sum() == 0


def test_gaussian_nll_matches_distribution_reference():
    target, mean = torch.randn(4, 3), torch.randn(4, 3)
    expected = -torch.distributions.Normal(mean, math.sqrt(0.1)).log_prob(target).sum(-1)
    torch.testing.assert_close(gaussian_nll(target, mean, 0.1), expected)
    with pytest.raises(ValueError):
        gaussian_nll(target, mean, 0)


@pytest.mark.parametrize("causal", [True, False])
def test_attention_matches_torch(causal):
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, 6, 8)
    actual, weights = scaled_attention(q, k, v, causal=causal)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 6))


def test_causal_future_invariance_and_past_sensitivity():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, 6, 8)
    before, weights = scaled_attention(q, k, v, causal=True)
    assert torch.count_nonzero(weights.triu(1)) == 0
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, -1] += 30
    changed_v[:, -1] += 50
    after, _ = scaled_attention(q, changed_k, changed_v, causal=True)
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    changed_v[:, 0] += 20
    after_past, _ = scaled_attention(q, k, changed_v, causal=True)
    assert not torch.allclose(before[:, 0], after_past[:, 0])


def test_flow_endpoints_and_derivative():
    x0, x1 = torch.randn(8, 2), torch.randn(8, 2)
    torch.testing.assert_close(flow_path(x0, x1, 0)[0], x0)
    torch.testing.assert_close(flow_path(x0, x1, 1)[0], x1)
    left, target = flow_path(x0.double(), x1.double(), 0.3)
    right, _ = flow_path(x0.double(), x1.double(), 0.30001)
    torch.testing.assert_close((right - left) / 0.00001, target)


def test_euler_constant_velocity_exact_and_input_unchanged():
    noise = torch.randn(7, 2)
    original = noise.clone()
    actual = euler_sample(lambda x, t: torch.full_like(x, 2), noise, steps=20)
    torch.testing.assert_close(actual, noise + 2)
    torch.testing.assert_close(noise, original)
    with pytest.raises(ValueError):
        euler_sample(lambda x, t: x, noise, steps=0)


def test_euler_refinement_reduces_error_for_linear_ode():
    noise = torch.ones(4, 2)
    exact = noise * math.e
    coarse = euler_sample(lambda x, t: x, noise, 5)
    fine = euler_sample(lambda x, t: x, noise, 50)
    assert (fine - exact).abs().max() < (coarse - exact).abs().max()


def test_dpo_reference_loss_and_gradient_direction():
    chosen = torch.tensor([-2.], requires_grad=True)
    rejected = torch.tensor([-3.], requires_grad=True)
    loss = dpo_loss(chosen, rejected, chosen.detach(), rejected.detach(), beta=0.2)
    torch.testing.assert_close(loss, torch.tensor(math.log(2)))
    loss.backward()
    assert chosen.grad < 0 and rejected.grad > 0
    assert dpo_loss(chosen.detach() + 1, rejected.detach(), chosen.detach(), rejected.detach()) < math.log(2)


def test_dpo_extreme_margins_are_finite():
    loss = dpo_loss(torch.tensor([-10000.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]))
    assert torch.isfinite(loss)
    with pytest.raises(ValueError):
        dpo_loss(*(torch.zeros(1) for _ in range(4)), beta=0)


def test_vae_gradients_reach_encoder_and_decoder():
    torch.manual_seed(1)
    model = VAE()
    data = torch.randn(5, 2)
    reconstructed, mean, logvar = model(data)
    loss = (gaussian_nll(data, reconstructed, 0.1) + diagonal_gaussian_kl(mean, logvar)).mean()
    loss.backward()
    for layer in (model.encoder, model.decoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        assert sum(float(p.grad.abs().sum()) for p in layer.parameters()) > 0


def test_original_attention_stable_and_scaled():
    path = Path(__file__).resolve().parents[1] / "Transformers/attention_calculation.py"
    spec = importlib.util.spec_from_file_location("original_attention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    np.testing.assert_allclose(module.softmax(np.array([1000., 1001.])), [1 / (1 + math.e), math.e / (1 + math.e)])
    raw = np.array([module.S_loves_he, module.S_loves_loves, module.S_loves_cats])
    expected = torch.softmax(torch.tensor(raw / math.sqrt(2)), dim=0).numpy()
    np.testing.assert_allclose(module.attention_weights, expected)
