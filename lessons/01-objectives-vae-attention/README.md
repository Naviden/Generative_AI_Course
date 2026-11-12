# 01 — Objectives, VAEs and attention

Previous: [00 — Get ready](../00-get-ready/README.md) · [Learning path](../README.md) · Next: [02 — Diffusion and flow matching](../02-diffusion-flow-matching/README.md)

**Question:** what does a model learn when we minimize its loss, and how can we check that the code implements the intended mathematics?

**You will produce:** an ELBO derivation, a two-run VAE comparison, and an explanation of the causal-attention test. These are the existing Class 1 activities; see the [240-minute agenda](../../docs/syllabus.md#class-1--objectives-latent-variables-and-transformers).

## Step 1 — Understand what a loss measures

Read [theory §1](../../docs/theory.md#1-what-a-generative-objective-measures). Write the autoregressive factorization for three tokens and identify what is observed during teacher forcing.

Before coding, answer: could training loss decrease while generated samples become less useful? Name one reason. Keep “fit to training data,” “held-out objective” and “sample quality” separate throughout this lesson.

## Step 2 — Derive the VAE objective

Read [theory §2](../../docs/theory.md#2-latent-variables-the-elbo-and-the-vae). Derive the ELBO using Jensen's inequality, then rearrange it to expose the gap to log likelihood. Identify the approximate posterior, prior and true posterior: they have different roles.

**Worked checkpoint:** with posterior mean `(1, 2)` and log variance `(0, 0)`, compute the KL to a standard normal. Predict what happens if you duplicate the latent coordinates, then what happens if you duplicate the example in a batch. Attempt this before consulting [Checkpoint 1](../checkpoints.md#1-vae-reductions).

Open these files in this order; search for the function or class name rather than reading every file:

1. [objectives.py](../../genai_course/objectives.py): `diagonal_gaussian_kl` and `gaussian_nll` — which dimensions are summed?
2. [models.py](../../genai_course/models.py): `VAE.forward` — where does parameter-free noise enter, and where do gradients flow?
3. [labs.py](../../genai_course/labs.py): the VAE branch of `generative_lab` — where are event losses averaged over a batch?

The toy dataset contains eight Gaussian clusters in **two dimensions**. This experiment is about probability objectives, not image generation. The decoder likelihood has fixed variance `0.1`, and generated observations include that noise.

## Step 3 — Compare two VAE runs

Predict the tradeoff when the KL weight changes from 1 to 0.1. Then run:

```bash
python -m genai_course.labs vae --steps 300 --seed 0 --beta 1
python -m genai_course.labs vae --steps 300 --seed 0 --beta 0.1
```

Use existing runs if their settings match. Open the `metrics.json` in each printed folder; the quantities below are inside its `metrics` object. See [how to inspect outputs](../../labs/outputs.md) if this is your first run.

| Record | β = 1 | β = 0.1 | Why it matters |
|---|---|---|---|
| Run-folder name | | | Links the claim to its settings and evidence |
| `heldout_reconstruction_nll_mc` | | | Fit of the Gaussian reconstruction distribution |
| `heldout_kl` | | | Posterior-to-prior divergence, summed over latent dimensions |
| `heldout_negative_elbo_mc` | | | Comparable NLL + KL diagnostic for both runs |
| `sample_sliced_w1` | | | Sample-distribution diagnostic; smaller is better for this metric |

**Interpret:** which measurements moved together, and which disagreed? Is the β = 0.1 training objective itself a negative ELBO? Why can Gaussian NLL be negative? Write a two-sentence conclusion tied to these runs, not a universal claim about β. Consult [Checkpoint 2](../checkpoints.md#2-interpreting-the-vae-comparison) after answering. Exact recorded values are not pass/fail targets.

## Step 4 — Place the VAE among other model families

Read [theory §3](../../docs/theory.md#3-contrasts-gans-and-normalizing-flows). Make a three-row comparison: VAE, GAN and invertible flow. For each, state whether the mechanism supplies a likelihood bound, an implicit sampler or a tractable density, and identify one limitation. This is a conceptual comparison; no historical GAN program is required.

## Step 5 — Check whether attention can see the future

Read [theory §4](../../docs/theory.md#4-transformers-and-scaling). For queries and keys shaped `(2, 6, 8)` and values shaped `(2, 6, 4)`, predict the attention-weight and output shapes. Which entries must be masked?

```bash
python -m genai_course.labs attention
```

You may reuse the setup result. In [objectives.py](../../genai_course/objectives.py), inspect `scaled_attention`: dot product → scaling → mask → row-wise softmax → weighted values. In [labs.py](../../genai_course/labs.py), read `attention_lab` and identify exactly which values are perturbed and which outputs are compared.

Expected diagnostics: output shape `[2, 6, 4]`; future leakage zero; row-sum error near floating-point precision. The CLI changes the final value vector. The broader test below also checks a changed future key and sensitivity to allowed past values:

```bash
python -m pytest tests/test_objectives.py -k causal_future_invariance_and_past_sensitivity -v
```

Read that test in [test_objectives.py](../../tests/test_objectives.py). Explain why future invariance needs a positive control using a past token, and why a passing mask test does not make a complete autoregressive LM. See [Checkpoint 3](../checkpoints.md#3-causal-attention).

## Step 6 — Explain before moving on

Return to the tokenization, scaling and data-leakage discussion in theory §4. A larger model or smaller training loss does not remove the need for a controlled evaluation.

Complete [Class 1's exit ticket](../../docs/assessment.md#a-class-1-exit-ticket--25-points-10-minutes): derive the bound, explain the KL reduction, and describe the causal-mask test and its limits. Use [your learning log](../learning-log.md) for the comparison table and conclusions. Full-paper readings R1 and R4 in the [reading guide](../../docs/readings.md) are optional extensions.

**Move on when you can explain the calculations and diagnostics, not merely reproduce the commands.**

Next: [02 — Diffusion and flow matching](../02-diffusion-flow-matching/README.md).
