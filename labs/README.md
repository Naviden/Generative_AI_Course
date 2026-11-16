# Lab workbook

Follow the [numbered lessons](../lessons/README.md) for the learning sequence. This workbook explains the same four experiments in detail: it is a reference to use at the corresponding lesson step, not another set of assignments. Start with [setup](../docs/setup.md) if the environment is not installed.

| Lab | Use during | Question answered |
|---|---|---|
| [1 — VAE](#lab-1--audit-a-vae-objective) | Lesson 01, Steps 2–3 | What changes when reconstruction and latent KL receive different weights? |
| [2 — Attention](#lab-2--can-attention-see-the-future) | Lesson 01, Step 5 | Can unavailable future information affect a current output? |
| [3 — Flow](#lab-3--control-the-flow-sampling-budget) | Lesson 02, Steps 2–4 | What changes when the sampler gets more steps but training stays fixed? |
| [4 — DPO](#lab-4--understand-a-preference-update) | Lesson 03, Steps 1–2 | How does one relative preference change a normalized policy? |

## How to use the examples

Keep the repository root as your working directory and activate its installed environment. Blocks labelled `bash` are **terminal commands**. For a block labelled `python`, open your text editor, create the repository’s `runs/` folder if it does not exist, and save the **whole block** as `runs/worked_example.py`. From the repository root, run `python runs/worked_example.py`. For the next example, replace that file’s contents with the next whole block and run the same command. Each block includes its imports and inputs; it does not depend on variables left by an earlier example. Saving a script also handles multi-line functions and loops consistently. No editing of the core implementation is needed.

The short calculations explain a mechanism before the training experiment. Their fixed expected answers differ from training measurements: the latter can vary across environments and are not grades. Keep examples and [worked checkpoint answers](../lessons/checkpoints.md) open while learning. Record your own runs and conclusions in the [learning log](../lessons/learning-log.md).

**Reading the code:** a tensor is an array with named-by-convention axes, such as `(batch, coordinates)`. Indexing starts at zero; `dim=-1` means the last axis. `.sum(-1)` adds coordinates of each example, whereas `.mean()` averages the remaining examples. `nn.Linear(a, b)` learns a matrix and bias mapping `a` input features to `b` outputs; `nn.Tanh()` applies a smooth nonlinear transformation coordinatewise. `.backward()` computes derivatives of a scalar loss, `optimizer.zero_grad()` clears old accumulated derivatives, and `optimizer.step()` updates parameters using the optimizer's rule. Adam, used by VAE/flow, adapts update scales using running gradient statistics; DPO uses ordinary gradient descent. [Foundations §7](../docs/foundations.md#7-reading-the-python-lab-code) works through tensors and automatic differentiation.

**What is generated:** VAE and flow learn points with two coordinates. Their training generator selects one of eight cluster centres on a circle of radius 2, then adds Gaussian coordinate noise with standard deviation 0.15. It draws 1,024 training points and 256 independent held-out points. Attention uses random tensors rather than text; DPO uses three abstract outcomes rather than sentences. There are no images, pretrained models or external datasets to obtain.

## Lab 1 — Audit a VAE objective

### Understand the calculation first

A variational autoencoder (VAE) uses an encoder that maps an observed point to a Gaussian latent distribution. A sampled latent goes through a decoder that predicts the mean of a Gaussian observation distribution. **Reconstruction negative log likelihood (NLL)** penalizes making the observed point unlikely. **Kullback–Leibler divergence (KL) to the prior** penalizes moving its latent distribution away from the standard normal used for generation. Training minimizes reconstruction NLL plus `beta × KL`.

For the derivation, read [theory §2](../docs/theory.md#2-latent-variables-the-elbo-and-the-vae) from **Why introduce an unobserved variable?** through **The likelihood determines what reconstruction means**, stopping before §3. Its two-state example explains why ELBO is a lower bound; its Gaussian example explains the reduction axes.

**Worked input:** one example with latent mean `(1, 2)` and log variance `(0, 0)`. The variances are therefore `(1, 1)`. Against the standard-normal prior, KL is $`(1^2+2^2)/2=2.5`$. Duplicating coordinates makes the KL 5; duplicating the example leaves the batch mean 2.5. For observed point `(1, 2)`, decoder mean `(0.8, 1.6)` and fixed observation variance 0.1, squared error sums to 0.20. Gaussian NLL is $`0.20/(2\times0.1)+\log(2\pi\times0.1)\approx0.5353`$. Thus the training losses are approximately 3.0353 at beta 1 and 0.7853 at beta 0.1.

Verify the arithmetic with the functions used by training:

```python
import torch
from genai_course.objectives import diagonal_gaussian_kl, gaussian_nll

mu = torch.tensor([[1.0, 2.0]])          # one example, two latent coordinates
logvar = torch.zeros_like(mu)            # log(variance), not standard deviation
kl = diagonal_gaussian_kl(mu, logvar)
print("KL per example:", kl.tolist())
print("Duplicated coordinates:", diagonal_gaussian_kl(mu.repeat(1, 2), logvar.repeat(1, 2)).tolist())
print("Duplicated batch mean:", diagonal_gaussian_kl(mu.repeat(2, 1), logvar.repeat(2, 1)).mean().item())
x = torch.tensor([[1.0, 2.0]])
decoder_mean = torch.tensor([[0.8, 1.6]])
nll = gaussian_nll(x, decoder_mean, variance=0.1)
print("Reconstruction NLL:", nll.item())
print("Loss at beta 1:", (nll + kl).mean().item())
print("Loss at beta 0.1:", (nll + 0.1 * kl).mean().item())
```

Expect `[2.5]`, `[5.0]`, `2.5`, and approximately `0.5353`, `3.0353`, `0.7853`. `repeat(1, 2)` duplicates columns; `repeat(2, 1)` duplicates rows. Averaging coordinates instead of summing them would halve this KL and silently alter the balance of the loss.

**Worked gradient:** sampling does not stop learning. Here $`\mu`$ is the latent mean, $`\ell`$ is log variance, $`\epsilon`$ is standard-normal noise, and $`\odot`$ multiplies corresponding coordinates. With fixed noise `(0.5, −1)`, $`z=\mu+\exp(\ell/2)\odot\epsilon`$ gives `(1.5, 1)`. For the illustrative scalar loss $`J=\sum_j z_j^2/2`$, the derivatives are $`\partial J/\partial\mu=z`$ and $`\partial J/\partial\ell=z\odot\epsilon\odot\exp(\ell/2)/2`$, giving `(1.5, 1)` and `(0.375, −0.5)` respectively. This small loss isolates reparameterization; it is not the full VAE training loss.

```python
import torch

mu = torch.tensor([[1.0, 2.0]], requires_grad=True)
logvar = torch.zeros_like(mu, requires_grad=True)
epsilon = torch.tensor([[0.5, -1.0]])
z = mu + (0.5 * logvar).exp() * epsilon
loss = 0.5 * z.square().sum()
loss.backward()
print("Latent sample:", z.detach().tolist())
print("Mean gradient:", mu.grad.tolist())
print("Log-variance gradient:", logvar.grad.tolist())
```

The noise tensor has no learned parameters; gradients pass through the mean and scale used to transform it. `.detach()` in the print statement only makes a value for inspection; detaching `z` before forming the loss would break this path.

### Trace one batch through the implementation

Open only the named functions, in this order:

| Location | Follow this operation | Expected shape or meaning |
|---|---|---|
| [models.py](../genai_course/models.py), `VAE.forward` | Encoder maps a batch to four outputs and `.chunk(2, dim=-1)` splits them | Input `(64, 2)`; mean and log variance each `(64, 2)` |
| Same function | Draw noise, multiply by `exp(logvar/2)`, add mean | Latent samples `(64, 2)`; the factor 1/2 converts log variance to log standard deviation |
| Same function | Apply decoder | Decoder means `(64, 2)`, not sampled observations |
| [objectives.py](../genai_course/objectives.py), `gaussian_nll` and `diagonal_gaussian_kl` | Sum coordinate losses | Each returns `(64,)`, one value per example |
| [labs.py](../genai_course/labs.py), VAE branch of `generative_lab` | Add weighted KL and take `.mean()` | One scalar used by `.backward()` |
| Same function, evaluation branch | Sample standard-normal latents, decode, add observation noise | 256 new observations, distinct from reconstructions of held-out inputs |

The decoder observation variance is fixed at 0.1; the encoder's learned latent variances are different quantities. Evaluation uses `model.eval()` to select evaluation behaviour and `torch.no_grad()` to avoid recording gradients. The small networks have no dropout; latent and observation sampling remain random in evaluation.

### Run the controlled comparison

Lowering beta makes latent divergence less expensive relative to reconstruction. After the worked example, note what you expect may happen to reconstruction and KL; treat the sample-quality effect as a question to measure.

```bash
python -m genai_course.labs vae --steps 300 --seed 0 --beta 1
python -m genai_course.labs vae --steps 300 --seed 0 --beta 0.1
```

`--steps 300` means 300 parameter updates, each on a sampled batch of 64; it does not mean 300 epochs. `--seed 0` fixes random choices for comparison. Reuse existing results if their settings match. Each command prints a unique run path; open its `metrics.json`. The [output guide](outputs.md#find-the-files) explains where fields live and how to read them.

| Record for both runs | Meaning |
|---|---|
| Run path and `config.beta` | Identify which run produced the evidence |
| `metrics.heldout_reconstruction_nll_mc` | Reconstruction NLL on held-out inputs, using a random encoder sample |
| `metrics.heldout_kl` | Encoder-to-prior KL, summed over latent coordinates and averaged over examples |
| `metrics.heldout_negative_elbo_mc` | Unweighted NLL + KL for **both** runs; not the beta-weighted objective when beta differs from 1 |
| `metrics.sample_sliced_w1` | Distance between generated and held-out point sets; smaller means closer under this diagnostic |
| `metrics.reference_sliced_w1` | Distance between two independent real point sets, showing finite-sample variation; not a strict lower bound |

**What the sample metric does:** project both sets onto a line, sort the projected coordinates and average paired absolute differences; repeat over 64 fixed directions. For projected sets `(0, 2)` and `(1, 3)`, the paired differences are 1 and 1, averaging to 1. A single scalar does not identify every coverage or fidelity failure. [Theory §7](../docs/theory.md#worked-example-what-a-projected-sample-distance-measures) gives the 2D example.

### Interpret, then continue

**Worked interpretation with illustrative values:** suppose beta 1 gives reconstruction NLL 0.6, KL 2.8 and sample distance 0.30; beta 0.1 gives −0.2, 5.0 and 0.41. The comparable negative ELBOs are 3.4 and 4.8. “Reducing the KL weight improved reconstruction but worsened this generated-sample metric. These two runs do not establish a universal effect or better prior generation.” The negative NLL is possible because a Gaussian density can exceed one. These values illustrate how to reason; use your measured values in your notes.

Record the objective equation, both run paths, the table and a two-sentence interpretation. You are done when you can explain why good reconstruction and good unconditional generation can disagree, and why the beta-weighted losses are not directly interchangeable. [Checkpoint 2](../lessons/checkpoints.md#2-interpreting-the-vae-comparison) provides another explanation. Plotting is optional; use the single [plotting guide](outputs.md#optional-plotting) if you want to inspect cluster coverage.

Return to [Lesson 01, Step 4](../lessons/01-objectives-vae-attention/README.md#step-4--place-the-vae-among-other-model-families).

## Lab 2 — Can attention see the future?

### Understand and test one row

Read [What attention computes](../docs/theory.md#what-attention-computes) and **Worked example: masking changes the distribution**, stopping before **From one head to a language model**. A query asks for information, keys determine weights and values supply the information being mixed. Softmax exponentiates scores and normalizes their sum to one. A causal mask excludes key positions after the query position before that normalization.

Use one sequence of three positions with scalar queries equal to 1, keys `(0, log 3, 10)` and values `(2, 6, 100)`. Query/key width is 1, so the scaling factor is 1. At the **second** query, a causal mask allows only the first two keys. Their exponentials are 1 and 3, giving weights `(1/4, 3/4, 0)` and output $`(1/4)2+(3/4)6=5`$. Changing the third key or value must leave this output unchanged. Changing the first value from 2 to 6 must raise it to 6: this positive control rules out a constant-output implementation.

```python
import math
import torch
from genai_course.objectives import scaled_attention

q = torch.ones(1, 3, 1, dtype=torch.float64)
k = torch.tensor([[[0.0], [math.log(3.0)], [10.0]]], dtype=torch.float64)
v = torch.tensor([[[2.0], [6.0], [100.0]]], dtype=torch.float64)
causal, weights = scaled_attention(q, k, v, causal=True)
unmasked, _ = scaled_attention(q, k, v, causal=False)
print("Second-row causal weights:", weights[0, 1].tolist())
print("Second-row causal output:", causal[0, 1, 0].item())
future_v = v.clone()
future_v[0, 2, 0] = 200.0
future_k = k.clone()
future_k[0, 2, 0] = -10.0
past_v = v.clone()
past_v[0, 0, 0] = 6.0
c_value, _ = scaled_attention(q, k, future_v, causal=True)
c_key, _ = scaled_attention(q, future_k, v, causal=True)
c_past, _ = scaled_attention(q, k, past_v, causal=True)
u_value, _ = scaled_attention(q, k, future_v, causal=False)
u_key, _ = scaled_attention(q, future_k, v, causal=False)
print("Causal output after future value/key changes:", c_value[0, 1, 0].item(), c_key[0, 1, 0].item())
print("Causal output after past value change:", c_past[0, 1, 0].item())
print("Unmasked outputs, original/value-change/key-change:", unmasked[0, 1, 0].item(), u_value[0, 1, 0].item(), u_key[0, 1, 0].item())
```

Here the axes are `(batch, position, feature)`. Index `[0, 1, 0]` selects the second position because indexing starts at zero; `.clone()` gives a separate tensor so each intervention changes only the intended input. Expect weights `[0.25, 0.75, 0.0]`, causal outputs `5`, `5`, `5`, then `6` for the past change. The unmasked outputs are approximately `99.9828`, `199.9646`, and `5.0011`: future information influences them. This unmasked comparison deliberately permits future access; it is not a failure of softmax.

### Work through numerical stability

For scores `(1000, 1001)`, subtract the maximum to get `(−1, 0)`. The normalized weights are $`(1/(1+e),e/(1+e))\approx(0.2689,0.7311)`$. Subtracting a shared constant cancels between numerator and denominator; it preserves probabilities while avoiding huge exponentials.

```python
import torch

scores = torch.tensor([1000.0, 1001.0])
raw_exp = scores.exp()
shifted_exp = (scores - scores.max()).exp()
manual = shifted_exp / shifted_exp.sum()
print("Raw exponentials finite:", torch.isfinite(raw_exp).all().item())
print("Stable manual weights:", manual.tolist())
print("PyTorch softmax:", scores.softmax(dim=-1).tolist())
```

Expect `False` for finite raw exponentials, and approximately `[0.2689, 0.7311]` for both stable calculations. This is an intentional overflow demonstration; the actual lab calls PyTorch's stable softmax.

### Run and interpret the implementation diagnostic

```bash
python -m genai_course.labs attention
python -m pytest tests/test_objectives.py -k causal_future_invariance_and_past_sensitivity -v
```

Reuse the attention setup run if available. The CLI uses random queries/keys of shape `(2, 6, 8)` and values `(2, 6, 4)`. Dot products produce scores `(2, 6, 6)`; softmax yields weights of that shape; weighting values produces `(2, 6, 4)`. This lab uses its query tensor as its key tensor; the explicit example above separates them to make the interventions easy to see.

Trace [objectives.py](../genai_course/objectives.py), `scaled_attention`: matrix product → divide by square root of key width → mask future entries → softmax across keys → weighted values. In [labs.py](../genai_course/labs.py), `attention_lab` changes the last value vector by 100 and compares **earlier** outputs. The final output is excluded because it may attend to its own position. The automated test additionally changes keys and checks permitted-past sensitivity.

| Output field | What it measures | Expected result |
|---|---|---|
| `output_shape` | Batch size, positions, value width | `[2, 6, 4]` |
| `row_sum_error` | Largest absolute difference between a weight-row sum and 1 | Near zero, typically below `1e-6` |
| `future_leakage` | Largest change in earlier outputs after the CLI's future-value intervention | `0.0` |

The CLI saves these diagnostics, not attention matrices; its `arrays.npz` is empty. Use the explicit example to inspect actual weights. Record your shape calculation, the three diagnostics and the future/past comparisons.

**Worked conclusion:** “The second row remained 5 after a future key or value changed, while changing an allowed past value raised it to 6. This supports causal information access in this computation; it does not test token embeddings, position information, residual/normalization blocks, shifted targets or a vocabulary output head.” A full language model needs those components. Continue with [Lesson 01, Step 6](../lessons/01-objectives-vae-attention/README.md#step-6--explain-before-moving-on).

## Lab 3 — Control the flow sampling budget

### Follow one point before training a field

Read [theory §6](../docs/theory.md#6-conditional-flow-matching) from **Prescribe a path** through **Training error and integration error are different**, stopping before §7. Here time 0 is noise and time 1 is data, the reverse of the diffusion-corruption convention. A **velocity field** predicts displacement per unit time from the current position and time. Training pairs are drawn independently; this is not optimal-transport endpoint matching.

For noise `(-1, 1)` and data `(3, −1)`, the straight path is `(1−t) × noise + t × data`. At time 0.25 it is `(0, 0.5)`, with target velocity `(4, −2)`. A prediction `(3, −1)` has summed squared error $`(3-4)^2+(-1+2)^2=2`$. A field given only position/time cannot always identify the endpoints: equally likely target velocities +2 and −2 at a condition have best prediction 0 and residual mean squared error (MSE) 4, as the theory calculation shows.

```python
import torch
from genai_course.objectives import flow_path
from genai_course.models import euler_sample

noise = torch.tensor([[-1.0, 1.0]])
data = torch.tensor([[3.0, -1.0]])
point, target = flow_path(noise, data, 0.25)
prediction = torch.tensor([[3.0, -1.0]])
print("Point at t=0.25:", point.tolist())
print("Target velocity:", target.tolist())
print("Summed squared error:", (prediction - target).square().sum(-1).mean().item())
# A known constant field verifies the sampling direction; it is not a trained model.
def constant_velocity(x, t):
    return target.expand_as(x)

sample = euler_sample(constant_velocity, noise, steps=4)
print("After four Euler steps:", sample.tolist())
print("Original noise preserved:", noise.tolist())
```

Expect `[[0.0, 0.5]]`, `[[4.0, -2.0]]`, `2.0`, final `[[3.0, -1.0]]` and unchanged initial noise `[[-1.0, 1.0]]`. With step size 1/4, each Euler update adds `(1, −0.5)`; four steps add the total displacement `(4, −2)`. This exact result uses a constant field; learned velocities vary along the path.

### Locate the training and sampling operations

| Location | What the student should identify | Worked shape trace |
|---|---|---|
| [objectives.py](../genai_course/objectives.py), `flow_path` | Construct positions and targets without a numerical solver | Noise/data `(64, 2)` and time `(64, 1)` produce points/targets `(64, 2)` |
| [models.py](../genai_course/models.py), `Velocity.forward` | Append time to the position using `torch.cat` | Two coordinates plus one time feature give `(64, 3)`; the network outputs two velocity coordinates |
| [labs.py](../genai_course/labs.py), flow branch of `generative_lab` | Predict targets, sum squared errors across coordinates, then average examples | `(64, 2)` errors → `(64,)` event losses → scalar batch loss |
| [models.py](../genai_course/models.py), `euler_sample` | Repeatedly evaluate the trained velocity and advance from 0 to 1 | `sampling_steps=5` means five increments, each of duration 0.2 |
| [labs.py](../genai_course/labs.py), evaluation branch | Reset evaluation randomness separately from training | Held-out velocity error and 256 generated points; neither is exact sample likelihood |

Time `(64, 1)` is broadcast across the two point coordinates during interpolation: each example uses the same time for both coordinates. During training, `--steps` counts optimizer updates. During sampling, `--sampling-steps` counts numerical integration steps. These are separate loops.

### Run, check the controls, and compare

```bash
python -m genai_course.labs flow --steps 300 --sampling-steps 5 --seed 0
python -m genai_course.labs flow --steps 300 --sampling-steps 50 --seed 0
```

The commands deterministically retrain with the same draws and use the same initial sampling noise. To establish that the comparison was actually matched, run the [paired-comparison example](outputs.md#check-a-paired-flow-comparison) with the two printed folders. It checks full training histories, configuration, data hashes, code and environment. Only `sampling_steps` should differ; all five consistency checks should be true. A mismatch is a reason to investigate before interpreting the sample metric.

| Record for each run | Location | Interpretation |
|---|---|---|
| Seed, training steps, sampling steps | `config` | Confirm which variable changed |
| `heldout_velocity_mse_sum` | `metrics` | Coordinate-summed velocity prediction error; should match within each pair |
| `sample_sliced_w1` | `metrics` | Generated-vs-held-out projected sample distance |
| `reference_sliced_w1` | `metrics` | Independent real-vs-real comparison; should match within each pair |
| `seconds` | Top level | Timed lab computation including training and evaluation, **not sampler-only latency** |
| Both run paths | Printed output | Preserve the provenance of the comparison |

To probe sensitivity to training randomness, repeat the pair for seeds 1 and 2:

```bash
python -m genai_course.labs flow --steps 300 --sampling-steps 5 --seed 1
python -m genai_course.labs flow --steps 300 --sampling-steps 50 --seed 1
python -m genai_course.labs flow --steps 300 --sampling-steps 5 --seed 2
python -m genai_course.labs flow --steps 300 --sampling-steps 50 --seed 2
```

Check each pair separately; histories should agree **within** a seed, not across seeds. In class you may pool these runs. For self-study you may stop at seed 0, leave the other rows blank and state that you have not tested run-to-run variation. Use optional [plots](outputs.md#optional-plotting) to inspect whether all eight clusters are covered; the numeric exercise does not require plotting packages.

**Worked interpretation with illustrative values:** distances `(0.30, 0.26)`, `(0.25, 0.24)` and `(0.20, 0.22)` for `(5, 50)` steps give differences `−0.04, −0.01, +0.02`, mean `−0.01`. One seed worsened despite a lower average. A suitable conclusion is: “With matched training within each seed, more Euler steps lowered the mean of this sample-distance estimate by 0.01 across three runs, with one reversal. This does not establish a universal improvement or a sampler-only speed advantage.” Use your own measurements; more integration steps cannot repair a wrong velocity field, and finite samples/projections affect the metric.

You are done when your table identifies the controls and you can separate **training error**, **numerical integration error** and **measurement variability**. Flow MSE and VAE ELBO measure different quantities and cannot be compared as one shared loss. Return to [Lesson 02, Step 5](../lessons/02-diffusion-flow-matching/README.md#step-5--connect-to-larger-systems-and-finish).

**Optional extension:** [the worked Heun exercise](extensions.md#heun--check-a-solver-before-comparing-generators) supplies an implementation, a known-solution check and the conditions for a matched-compute study. It is outside the core completion criteria.

## Lab 4 — Understand a preference update

### Work through one update first

Read [Worked example: interpreting the DPO margin](../docs/theory.md#worked-example-interpreting-the-dpo-margin) through its lab connection, stopping before §9. If the reference policy or loss is unfamiliar, read the three preceding subsections of §8 first; they derive the objective with a two-response example.

This lab has one context and three abstract one-token completions labelled 0, 1 and 2. It repeats the preference **0 over 1**. Outcome 2 is unpaired. A **logit** is an unconstrained score; softmax turns the three logits into probabilities. The **reference policy** stays fixed and uniform. The adjusted margin compares the winner/loser log odds of the current policy against those of the reference. Direct preference optimization (DPO) loss is $`-\log\sigma(\beta m)`$, where $`\sigma(a)=1/(1+\exp(-a))`$ and $`m`$ is that margin.

With logits `(0, 0, 0)`, both policies are uniform, margin is zero and loss is log 2, approximately 0.6931. At beta 1 the logit gradients are `(−0.5, +0.5, 0)`. A step of size 0.5 gives logits `(0.25, −0.25, 0)` and probabilities approximately `(0.4192, 0.2543, 0.3265)`. The preferred probability rises, but the unpaired probability also changes through normalization even though its logit stayed fixed.

```python
import math
import torch
from genai_course.objectives import dpo_loss

logits = torch.zeros(3, requires_grad=True)
reference = torch.full((3,), -math.log(3.0))  # fixed log probabilities
log_probs = logits.log_softmax(dim=-1)
loss = dpo_loss(log_probs[0:1], log_probs[1:2], reference[0:1], reference[1:2], beta=1.0)
loss.backward()
print("Initial loss:", loss.item())
print("Logit gradients:", logits.grad.tolist())
with torch.no_grad():
    logits -= 0.5 * logits.grad

updated = logits.log_softmax(dim=-1)
after = dpo_loss(updated[0:1], updated[1:2], reference[0:1], reference[1:2], beta=1.0)
print("Updated logits:", logits.detach().tolist())
print("Updated probabilities:", logits.detach().softmax(dim=-1).tolist())
print("Loss after update:", after.item())
```

Expect loss `0.6931`, the gradients and probabilities above, then loss approximately `0.4741`. Slices `[0:1]` and `[1:2]` retain a one-element batch for the loss function. `.log_softmax()` computes normalized log probabilities directly; the reference tensor is never updated. The `no_grad` block applies an update without building another derivative graph.

### Run the finite-training comparison

```bash
python -m genai_course.labs dpo --steps 100 --seed 0 --beta 0.1
python -m genai_course.labs dpo --steps 100 --seed 0 --beta 1
```

In [labs.py](../genai_course/labs.py), `dpo_lab`, trace three trainable logits → normalized log probabilities → `dpo_loss` for outcomes 0 and 1 → gradient descent with learning rate 0.5. `--steps 100` repeats this same preference 100 times. There is no held-out context, text tokenization or learned reward model in this experiment. The seed is recorded for consistency, but this particular three-logit computation has no random training draws.

| Record from `metrics` | What it means |
|---|---|
| `loss_first`, `loss_last` | Pair loss before the first and before the final optimizer updates |
| `final_probabilities[0]` | Preferred outcome probability **after** all updates |
| `final_probabilities[1]` | Rejected outcome probability after all updates |
| `final_probabilities[2]` | Unpaired outcome probability after all updates |
| `kl_policy_reference` | Final policy-to-uniform-reference divergence; zero means identical distributions |

Also record both run paths and beta values. Do not equate `loss_last` with loss recomputed at `final_probabilities`: they describe adjacent points in the update sequence. The one-update example above shows the distinction directly.

**Worked interpretation:** the one-step beta-1 example moved probabilities from `(1/3, 1/3, 1/3)` to approximately `(0.4192, 0.2543, 0.3265)` and lowered loss from 0.6931 to 0.4741. It fitted the supplied relative preference more strongly and altered an unpaired probability. It did not test whether any response was true, safe or useful on another context. For your two 100-step runs, describe what actually changed; beta affects gradient scales and finite-step fitting, so observed KL need not follow the exact fixed-reward optimum's trend.

**Connection to a real LM, with a concrete example:** for one prompt and two candidate completions, sum each completion's token log probabilities; exclude prompt and padding positions from this sum. A two-token completion with target probabilities 0.4 and 0.5 has log probability $`\log0.4+\log0.5=\log0.2`$, not their average. Keep the reference model fixed, collect preferences for training prompts, and evaluate on separate prompts. If annotators disagree on the same pair, repeatedly treating only one direction as truth misrepresents those labels; the optional experiment below isolates that issue. A toy decrease in pair loss cannot replace held-out evaluation.

You are done when you can explain the gradient sign, the unpaired probability change and one missing piece of generalization evidence. Continue with [Lesson 03, Step 3](../lessons/03-post-training-research/README.md#step-3--evaluate-reasoning-retrieval-and-tool-use).

**Optional extension:** [conflicting preferences](extensions.md#conflicting-preferences--fit-a-known-label-probability) gives a fully specified label distribution, derives its target margin and supplies runnable optimization code. It is not an additional required task.

## If you cannot run an experiment

Stop at the first execution error and use [setup troubleshooting](../docs/setup.md#troubleshooting). You can still work through the fixed-input examples and the [recorded validation results](../docs/validation.md), clearly labelling supplied evidence instead of claiming to have run it. Fill the interpretation tasks using that evidence and identify what you could not check. Submission or grading arrangements belong to your instructor; self-study needs no submission or partner.
