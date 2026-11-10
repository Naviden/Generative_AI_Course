"""Run with python -m genai_course.labs {attention,vae,flow,dpo}."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluation import sliced_wasserstein
from .models import VAE, Velocity, euler_sample
from .objectives import diagonal_gaussian_kl, dpo_loss, flow_path, gaussian_nll, scaled_attention


def mixture(count, seed):
    """Eight equally weighted isotropic Gaussians on a radius-two circle."""
    rng = np.random.default_rng(seed)
    angles = rng.integers(0, 8, count) * (2 * math.pi / 8)
    centers = 2 * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    return torch.tensor(centers + rng.normal(0, 0.15, (count, 2)), dtype=torch.float32)


def attention_lab(args):
    query = torch.randn(2, 6, 8)
    value = torch.randn(2, 6, 4)
    output, weights = scaled_attention(query, query, value, causal=True)
    changed = value.clone()
    changed[:, -1] += 100
    perturbed, _ = scaled_attention(query, query, changed, causal=True)
    return {"row_sum_error": float((weights.sum(-1) - 1).abs().max()),
            "future_leakage": float((output[:, :-1] - perturbed[:, :-1]).abs().max()),
            "output_shape": list(output.shape)}, {}


def generative_lab(args):
    train, heldout = mixture(1024, args.seed), mixture(256, args.seed + 10000)
    model = VAE() if args.lab == "vae" else Velocity()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    history = []
    for step in range(args.steps):
        batch = train[torch.randint(len(train), (64,))]
        if args.lab == "vae":
            reconstruction, mean, log_variance = model(batch)
            loss = (gaussian_nll(batch, reconstruction, variance=0.1) +
                    args.beta * diagonal_gaussian_kl(mean, log_variance)).mean()
        else:
            t = torch.rand(len(batch), 1)
            points, target = flow_path(torch.randn_like(batch), batch, t)
            loss = (model(points, t) - target).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training objective")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    # Independent, fixed evaluation randomness; no fitting on held-out samples.
    torch.manual_seed(args.seed + 20000)
    with torch.no_grad():
        if args.lab == "vae":
            reconstruction, mean, log_variance = model(heldout)
            nll = gaussian_nll(heldout, reconstruction, variance=0.1)
            kl = diagonal_gaussian_kl(mean, log_variance)
            # Sample the likelihood, not only the decoder's conditional mean.
            generated = model.decoder(torch.randn(len(heldout), 2)) + math.sqrt(0.1) * torch.randn_like(heldout)
            diagnostics = {"heldout_negative_elbo_mc": float((nll + kl).mean()),
                           "heldout_kl": float(kl.mean()), "heldout_reconstruction_nll_mc": float(nll.mean())}
        else:
            t = torch.rand(len(heldout), 1)
            points, target = flow_path(torch.randn_like(heldout), heldout, t)
            diagnostics = {"heldout_velocity_mse_sum": float((model(points, t) - target).square().sum(-1).mean())}
            generated = euler_sample(model, torch.randn_like(heldout), args.sampling_steps)
    # This baseline measures finite-sample noise, not an attainable lower bound.
    diagnostics.update({"train_objective_first": history[0], "train_objective_last": history[-1],
                        "sample_sliced_w1": sliced_wasserstein(heldout.numpy(), generated.numpy()),
                        "reference_sliced_w1": sliced_wasserstein(heldout.numpy(), mixture(256, args.seed + 30000).numpy())})
    return diagnostics, {"train": train.numpy(), "heldout": heldout.numpy(), "generated": generated.numpy(),
                         "train_objective": np.asarray(history)}


def dpo_lab(args):
    # A categorical policy for one context and three one-token completions.
    # Preferences are deliberately synthetic: completion 0 is preferred to 1.
    logits = nn.Parameter(torch.zeros(3))
    reference = torch.full((3,), -math.log(3))
    optimizer = torch.optim.SGD([logits], lr=0.5)
    history = []
    for _ in range(args.steps):
        log_probs = logits.log_softmax(-1)
        loss = dpo_loss(log_probs[0:1], log_probs[1:2], reference[0:1], reference[1:2], args.beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    probabilities = logits.detach().softmax(-1)
    return {"initial_preferred_probability": 1 / 3, "final_probabilities": probabilities.tolist(),
            "loss_first": history[0], "loss_last": history[-1],
            "kl_policy_reference": float((probabilities * (probabilities.log() - reference)).sum())}, {"train_objective": np.asarray(history)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", choices=["attention", "vae", "flow", "dpo"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--beta", type=float, default=1.0, help="VAE KL weight or DPO inverse reward scale")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    if args.steps < 1 or args.sampling_steps < 1 or not math.isfinite(args.beta) or args.beta <= 0:
        parser.error("steps, sampling-steps and beta must be finite and positive")
    if not 0 <= args.seed < 2**32 - 30000:
        parser.error("seed must be in [0, 2**32 - 30000)")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    if args.lab == "attention":
        metrics, arrays = attention_lab(args)
    elif args.lab == "dpo":
        metrics, arrays = dpo_lab(args)
    else:
        metrics, arrays = generative_lab(args)
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive run folders prevent accidentally overwriting earlier evidence.
    run = Path(tempfile.mkdtemp(prefix=f"{args.lab}-seed{args.seed}-", dir=args.output))
    np.savez(run / "arrays.npz", **arrays)
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    config = vars(args).copy()
    config["output"] = str(args.output)
    record = {"config": config, "metrics": metrics, "seconds": time.perf_counter() - start,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": version("torch"), "numpy": version("numpy"), "device": "cpu", "threads": 1},
              "git_commit": commit, "git_dirty": dirty,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
              "data_sha256": {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in arrays.items() if name in ("train", "heldout")}}
    (run / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"run": str(run), "metrics": metrics}, indent=2, allow_nan=False))
    return run


if __name__ == "__main__":
    main()
