# 02 — Diffusion and flow matching

Previous: [01 — Objectives, VAEs and attention](../01-objectives-vae-attention/README.md) · [Learning path](../README.md) · Next: [03 — Post-training and research](../03-post-training-research/README.md)

**Question:** how can learned dynamics turn noise into data, and how do we separate a model's error from its sampler's error?

**You will produce:** a diffusion derivation and a paired comparison of two flow-sampling budgets. These follow the [Class 2 agenda](../../docs/syllabus.md#class-2--diffusion-and-flow-matching); the implementation lab is flow matching, not a DDPM training pipeline.

## Step 1 — Derive the noisy observation

Read [theory §5](../../docs/theory.md#5-diffusion-and-scores). Start with two forward Gaussian transitions and derive the mean and variance of `x₂` given `x₀`. Combine independent noise variances, not standard deviations.

**Worked checkpoint:** for `α₁ = 0.9` and `α₂ = 0.8`, what are the mean coefficient, variance and standard deviation of `q(x₂ | x₀)`? Then relate the injected noise to the conditional score. Check [answer 4](../checkpoints.md#4-diffusion-variance).

Contrast the reverse SDE and probability-flow ODE. Mark the integration direction and explain the factor `1/2` in the ODE drift. State the exact-score/regularity assumptions; finite-step generation with an approximate model is not an exact equivalence claim.

## Step 2 — Connect a conditional path to a learned velocity

Read [theory §6](../../docs/theory.md#6-conditional-flow-matching). For a noise endpoint `x₀`, data endpoint `x₁`, and time `t`, write the straight interpolation and its derivative. Explain why the network learns a conditional expectation at a position and time rather than receiving the endpoint pair at generation time.

Inspect, in order:

1. [objectives.py](../../genai_course/objectives.py): `flow_path` — verify both endpoints and the target velocity.
2. [models.py](../../genai_course/models.py): `Velocity` and `euler_sample` — identify how time enters and how sampling advances from 0 to 1.
3. [labs.py](../../genai_course/labs.py): `generative_lab` — distinguish training randomness from evaluation randomness.

This uses the same eight-cluster 2D data distribution as the VAE. It does not use optimal-transport endpoint matching. See [Checkpoint 5](../checkpoints.md#5-flow-path-and-sampling).

## Step 3 — Change the sampling budget, hold training fixed

Predict what increasing the number of Euler steps might improve and what it cannot fix. Then run:

```bash
python -m genai_course.labs flow --steps 300 --sampling-steps 5 --seed 0
python -m genai_course.labs flow --steps 300 --sampling-steps 50 --seed 0
```

`--steps` controls **training updates**; `--sampling-steps` controls **generation integration steps**. The two invocations retrain deterministically with the same training randomness and use the same initial sampling noise. Changing only the sampling budget is the intended paired comparison.

Record values from each `metrics.json`:

| Seed | Sampling steps | `heldout_velocity_mse_sum` | `sample_sliced_w1` | `reference_sliced_w1` | `seconds` |
|---|---|---|---|---|---|
| 0 | 5 | | | | |
| 0 | 50 | | | | |

`seconds` is at the JSON top level and covers the lab computation, including training and evaluation; it is **not a sampler-only latency measurement**. The other listed values are inside `metrics`. Also record the run paths and verify equal configuration except the sampling budget and equal data hashes. See the [output guide](../../labs/outputs.md).

The two runs should have equal training histories and held-out velocity MSE on the same deterministic environment. A difference in these quantities is a reason to check settings before interpreting sample quality. The sample metric need not improve monotonically: a more accurate solver still follows an imperfect velocity field, and the metric uses finite samples.

## Step 4 — Evaluate the claim, not just the loss

Read [theory §7](../../docs/theory.md#7-evaluation-and-reproducibility). Inspect `sliced_wasserstein` in [evaluation.py](../../genai_course/evaluation.py). It averages one-dimensional projected distances; it is not exact multivariate Wasserstein distance.

Pairs repeat Step 3 using `--seed 1` or `--seed 2` and pool their tables within the existing lab time. For self-study, repeat both budgets for each seed when practical. Compare the generated-vs-held-out distance with the real-vs-real reference. If plotting dependencies are already available, use the optional [plotting example](../../labs/outputs.md#optional-plotting) to examine coverage. Plotting is not required for completion.

Write: “For these runs, changing ___ while holding ___ fixed changed ___.” Then name two limits of that conclusion. Do not compare the numeric flow training loss with the VAE ELBO. Consult [Checkpoint 6](../checkpoints.md#6-a-controlled-flow-comparison).

## Step 5 — Connect to larger systems and finish

Return to the classifier-free guidance and latent-diffusion paragraphs in theory §5. Use R14 and R16 in the [reading guide](../../docs/readings.md) for the scheduled research comparison: which additional model, data and compute choices would be needed to extend a result to images or language? Full papers and new implementations are optional.

Complete [Class 2's exit ticket](../../docs/assessment.md#b-class-2-exit-ticket--25-points-10-minutes). Move on when you can derive the noisy marginal, state the flow path and sampling direction, and explain what the budget comparison does and does not establish.

Optional extension, outside the required path: implement Heun sampling and compare equal **velocity evaluations**, not equal step counts.

Next: [03 — Post-training and research](../03-post-training-research/README.md).
