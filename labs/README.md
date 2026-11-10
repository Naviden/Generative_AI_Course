# In-class lab workbook

Run from the repository root after [setup](../docs/setup.md). There are four exercises across three classes. All required code is complete; research extensions are optional. The experiments are intentionally small enough to inspect and rerun during teaching.

## Lab 1 — Audit a VAE objective (class 1, 35 minutes)

**Predict (5 min):** derive the diagonal-Gaussian KL. Predict the effect of changing the latent dimension if KL is incorrectly averaged over coordinates. Explain why β=1 has a special interpretation.

**Run and inspect (15 min):**

```bash
python -m genai_course.labs vae --steps 300 --seed 0 --beta 1
python -m genai_course.labs vae --steps 300 --seed 0 --beta 0.1
```

Read `genai_course/objectives.py` and `genai_course/models.py`. Trace the reparameterization gradient and the observation variance. Open each generated `metrics.json`. `heldout_negative_elbo_mc` always uses NLL+KL, even when training uses a different beta. It is a stochastic estimator of a bound, not an exact likelihood. Decoder sampling includes Gaussian observation noise.

**Discuss (10 min):** compare held-out KL, reconstruction NLL and sample sliced-W1. Does one metric improve while another deteriorates? A short run can remain underfit; explain the observation without promising good visual quality. Sample metrics are evaluated on independently generated data.

**Submit (5 min):** one equation, the two run-folder names and a two-sentence interpretation. Full training is not required for credit.

Optional plot after installing `.[notebooks]`:

```python
import numpy as np
import matplotlib.pyplot as plt
# Substitute the actual run directory printed by the CLI.
a = np.load("runs/REPLACE_WITH_RUN/arrays.npz")
plt.scatter(*a["heldout"].T, s=8, alpha=0.4, label="held-out")
plt.scatter(*a["generated"].T, s=8, alpha=0.4, label="generated")
plt.axis("equal")
plt.legend()
plt.show()
```

## Lab 2 — Can attention see the future? (class 1, 30 minutes)

```bash
python -m genai_course.labs attention
```

Spend 5 minutes predicting tensor shapes, 10 minutes tracing scale/mask/softmax in `scaled_attention`, 10 minutes testing perturbations and 5 minutes writing the conclusion. The printed row-sum error should be near floating-point precision and future leakage should be zero. Change a future key and value in a scratch copy and compare causal versus unmasked attention. Changing a past value should generally affect later outputs. The automated tests include a reference comparison to PyTorch attention.

Explain why a causal mask is necessary but not sufficient for a full autoregressive LM. Identify the absent positional encoding, token embedding, residual/normalization blocks and output head. As a numerical exercise, apply softmax to logits near 1,000: subtracting their maximum preserves probabilities and avoids overflow.

## Lab 3 — Train a flow and control the sampling budget (class 2, 55 minutes)

**Derive (10 min):** write x_t=(1−t)x₀+t x₁ and target x₁−x₀. Identify the conditional expectation learned by squared-error regression. State the time direction.

**Run (15 min):**

```bash
python -m genai_course.labs flow --steps 300 --sampling-steps 5 --seed 0
python -m genai_course.labs flow --steps 300 --sampling-steps 50 --seed 0
```

Keep seed, training steps and all other parameters fixed. These two commands retrain deterministically with identical training draws and use identical initial sampling noise; only the Euler budget changes. Record wall time and sliced-W1. More steps need not improve the metric monotonically when the velocity is inaccurate and evaluation uses finite samples.

**Compare (20 min):** different pairs repeat with seeds 1 and 2, then pool findings. Compare generated-vs-held-out distance with `reference_sliced_w1`, the independent real-vs-real diagnostic. Inspect scatterplots if available. Discuss coverage, finite samples and the distinction between solver error and model error. Three seeds are a sensitivity exercise, not definitive confidence intervals.

**Submit (10 min):** a table with seed, sampling steps, distance and runtime, plus the main confound you controlled. Do not compare flow MSE numerically with a VAE ELBO as if they were the same quantity.

Optional extension: implement Heun's method, compare equal numbers of velocity evaluations, and distinguish quality-at-fixed-cost from quality-at-fixed-step-count.

## Lab 4 — Understand DPO before scaling it (class 3, 25 minutes)

```bash
python -m genai_course.labs dpo --steps 100 --beta 0.1
python -m genai_course.labs dpo --steps 100 --beta 1
```

**Predict (5 min):** the initial policy equals the reference, so the loss is log 2. Which sign should the chosen log-probability gradient have?

**Inspect (10 min):** the policy has one context and three possible one-token completions. The repeated synthetic preference favors completion 0 over completion 1. Record the preferred probability, unpaired completion probability and KL to the fixed reference. A relative preference does not directly constrain every other probability.

**Interpret (10 min):** why is a lower training loss not evidence of factuality or generalization? Explain how completion log probabilities, prompt masks, held-out contexts and contradictory labels would enter a real LM experiment. Different beta values change the optimization dynamics; finite-step outcomes need not mimic an exact reward-objective optimum.

Optional extension: introduce conflicting preference labels with a known probability, derive the optimum log-odds margin, and test convergence. Keep the reference fixed.

## If execution fails during class

Use the [recorded validation results](../docs/validation.md) and inspect the objective code in pairs; submit the derivation and proposed diagnostic. Students should not lose the learning exercise to package installation. The instructor resolves environment issues outside the timed activities.
