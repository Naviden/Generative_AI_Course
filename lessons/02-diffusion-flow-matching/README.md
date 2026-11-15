# 02 — Diffusion and flow matching

Previous: [01 — Objectives, VAEs and attention](../01-objectives-vae-attention/README.md) · [Learning path](../README.md) · Next: [03 — Post-training and research](../03-post-training-research/README.md)

**Question:** how can learned dynamics turn noise into data, and how do we separate a model's error from its sampler's error?

**You will produce:** a diffusion derivation and a paired comparison of two flow-sampling budgets. These follow the [Class 2 agenda](../../docs/syllabus.md#class-2--diffusion-and-flow-matching); the implementation lab is flow matching, not a DDPM training pipeline.

## Step 1 — Derive the noisy observation

Work through the following reading and calculation **during this lesson**:

1. Read [Learn to reverse a controlled corruption process](../../docs/theory.md#learn-to-reverse-a-controlled-corruption-process) and [Worked example: accumulating noise variance](../../docs/theory.md#worked-example-accumulating-noise-variance). **Stop before “A score is a direction, not a quality rating”.** These passages introduce the forward process and calculate the two-step example.
2. Reproduce that example with $`\alpha_1=0.9`$ and $`\alpha_2=0.8`$. Start from the explicit transitions below, where the two noise draws are independent standard normals:

```math
x_1=\sqrt{0.9}\,x_0+\sqrt{0.1}\,\epsilon_1,
\qquad
x_2=\sqrt{0.8}\,x_1+\sqrt{0.2}\,\epsilon_2.
```

Substitute the first expression into the second. The clean-data coefficient becomes $`\sqrt{0.72}`$; the independent noise variances are 0.08 and 0.20, giving total variance 0.28 and standard deviation about 0.5292. Record the substitution and identify why the variances add. [Answer 4](../checkpoints.md#4-diffusion-variance) provides a further explanation.

3. Read [A score is a direction, not a quality rating](../../docs/theory.md#a-score-is-a-direction-not-a-quality-rating) and [Two dynamics with the same ideal marginals](../../docs/theory.md#two-dynamics-with-the-same-ideal-marginals). **Stop before “Conditioning and guidance”; Step 5 covers it.** Follow the Gaussian score explanation before comparing the reverse SDE and probability-flow ODE.
4. Make a two-row note for those dynamics: drift, whether random increments remain, and integration direction. The supplied formulas use score coefficients 1 and 1/2 respectively; explain that the ODE transports the initial randomness without further Brownian increments. Include the stated exact-score and regularity conditions. Their origin is the continuous-time distributional construction, not the arithmetic of the two-step example.

## Step 2 — Connect a conditional path to a learned velocity

**Read the first three subsections of theory §6**, from [Prescribe a path, then learn how to move along its distribution](../../docs/theory.md#prescribe-a-path-then-learn-how-to-move-along-its-distribution), through **Worked example: the target is a velocity** and **Conservation of probability**. **Stop before “Training error and integration error are different”; Step 3 covers it.**

Follow the worked path from noise endpoint −1 to data endpoint 3. Its position is $`x_t=-1+4t`$ and its velocity is 4; at time 0.25 it reaches 0. Verify the positions at times zero and one. This establishes what “path” and “velocity” mean before you inspect their implementation.

Then use the supplied example of equally likely target velocities 2 and −2 to explain why squared-error prediction chooses their conditional mean, zero, despite positive residual error. Record that explanation and identify which inputs the network receives: intermediate position and time, not the endpoint pair.

Inspect, in order:

1. [objectives.py](../../genai_course/objectives.py): `flow_path` — verify both endpoints and the target velocity.
2. [models.py](../../genai_course/models.py): `Velocity` and `euler_sample` — identify how time enters and how sampling advances from 0 to 1.
3. [labs.py](../../genai_course/labs.py): `generative_lab` — distinguish training randomness from evaluation randomness.

This uses the same eight-cluster 2D data distribution as the VAE. It does not use optimal-transport endpoint matching. See [Checkpoint 5](../checkpoints.md#5-flow-path-and-sampling).

## Step 3 — Change the sampling budget, hold training fixed

**Read [Training error and integration error are different](../../docs/theory.md#training-error-and-integration-error-are-different), the final subsection of theory §6. Stop at “7. Evaluation and reproducibility”.** Follow the worked Euler calculation: one step gives 2, two steps give 2.25, and ten give about 2.5937, compared with the exact value about 2.7183.

Use this example to state the purpose of the next experiment: compare numerical sampling budgets for the same learned field. More steps can reduce integration error, while an incorrect field can still produce incorrect samples. Then run:

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

**Read all three subsections of theory §7**, from [Decide which claim a measurement is meant to support](../../docs/theory.md#decide-which-claim-a-measurement-is-meant-to-support) through **Worked example: what a projected sample distance measures** and **Paired comparisons and the unit of uncertainty**. **Stop at “8. SFT, efficient adaptation and DPO”.**

First reproduce the supplied distance calculation between `(0, 2)` and `(1, 3)`: pair the sorted coordinates, take absolute differences, then average to obtain 1. Next inspect `sliced_wasserstein` in [evaluation.py](../../genai_course/evaluation.py) and identify where the code repeats this idea across projection directions. Finally use the three-seed example in the reading to interpret the repeated runs below.

Pairs repeat Step 3 using `--seed 1` or `--seed 2` and pool their tables within the existing lab time. For self-study, repeat both budgets for each seed when practical. Compare the generated-vs-held-out distance with the real-vs-real reference. If plotting dependencies are already available, use the optional [plotting example](../../labs/outputs.md#optional-plotting) to examine coverage. Plotting is not required for completion.

Write: “For these runs, changing ___ while holding ___ fixed changed ___.” Then name two limits of that conclusion. Do not compare the numeric flow training loss with the VAE ELBO. Consult [Checkpoint 6](../checkpoints.md#6-a-controlled-flow-comparison).

## Step 5 — Connect to larger systems and finish

**Read [Conditioning and guidance](../../docs/theory.md#conditioning-and-guidance), the final subsection of theory §5, including its scalar guidance example and latent-diffusion explanation. Stop at “6. Conditional flow matching”.** The earlier board exercise repeats the two-transition calculation completed in Step 1; no additional derivation is required here.

For images, identify the encoder and decoder that latent diffusion adds and explain one error that can come from compression. For language, read only the two final paragraphs about LLaDA and masked diffusion in [Reading a research claim critically](../../docs/theory.md#reading-a-research-claim-critically), stopping before “Connection to the activity”. State why equal generation-step counts would not necessarily give equal computation. R14 and R16 in the [reading guide](../../docs/readings.md) are optional source readings; no additional implementation is assigned.

Complete [Class 2's exit ticket](../../docs/assessment.md#b-class-2-exit-ticket--25-points-10-minutes). Move on when you can derive the noisy marginal, state the flow path and sampling direction, and explain what the budget comparison does and does not establish.

Optional extension, outside the required path: implement Heun sampling and compare equal **velocity evaluations**, not equal step counts.

Next: [03 — Post-training and research](../03-post-training-research/README.md).
