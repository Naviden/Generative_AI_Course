# How to read a lab result

[Learning path](../lessons/README.md) · [Compact lab workbook](README.md)

Every lab command prints JSON containing a `run` path and a `metrics` object. A typical path has the form `runs/vae-seed0-<unique-suffix>`. Use the **actual path printed by your command**. New runs get new folders so earlier evidence is retained.

## Find the files

In your editor or file browser, open the repository's `runs/` folder and then the run you just produced. Open `metrics.json` as text. If you ran every setup command earlier, there may be several folders; identify a run by its `config`, not just its timestamp or name.

| File | Contains | How to use it |
|---|---|---|
| `metrics.json` | Configuration, scalar metrics, environment, elapsed seconds, source and data hashes | Read directly; use for your comparison tables |
| `arrays.npz` | Numerical arrays | Open with NumPy when a lesson asks for histories or sample plots |

Inside `metrics.json`, `metrics` contains the measured quantities. `config`, `seconds`, `environment`, `git_commit`, `git_dirty`, `source_sha256` and `data_sha256` are separate top-level entries. A dirty-worktree flag records local modifications; it is not itself a training failure. Preserve the relevant modifications if you want to reproduce the result.

## Interpret the quantities by lab

| Lab | Inspect first | Interpretation |
|---|---|---|
| Attention | `output_shape`, `future_leakage`, `row_sum_error` | Shape should be `[2, 6, 4]`; leakage zero; row-sum error near floating-point precision |
| VAE | `heldout_reconstruction_nll_mc`, `heldout_kl`, `heldout_negative_elbo_mc` | Reconstruction and KL trade off; the reported negative ELBO always uses NLL + KL |
| VAE / flow | `sample_sliced_w1`, `reference_sliced_w1` | Generated-vs-held-out distance and independent real-vs-real reference; the latter is not a strict lower bound |
| Flow | `heldout_velocity_mse_sum` | Held-out squared velocity error, summed over coordinates and averaged over examples; not sample likelihood |
| DPO | `loss_first`, `loss_last`, `final_probabilities`, `kl_policy_reference` | Fitting one preference versus a fixed reference; no held-out generalization measurement |

`train_objective_first` and `train_objective_last` describe training objectives for VAE/flow, not interchangeable units. VAE's β-weighted training objective differs from its reported unweighted negative ELBO when β differs from one. Negative Gaussian NLL is possible because densities can exceed one.

`seconds` measures the timed lab computation, including training and evaluation and excluding installation. It is **not sampler-only latency**. The CLI uses CPU execution and one torch thread. Measured times and stochastic estimates may differ across hardware or package versions; passing does not require matching every published decimal.

## Inspect saved arrays

VAE and flow save `train`, `heldout`, `generated` and `train_objective`. DPO saves only `train_objective`. Attention saves an empty array archive because its output is scalar diagnostics and a shape.

For VAE/flow, the data arrays have shapes `(1024, 2)`, `(256, 2)` and `(256, 2)` respectively. `train_objective` has one entry per training update. Read the actual run settings before comparing.

The following is **Python code**, not a shell command. Paste it into a Python session or save it as a script and replace the run path:

```python
from pathlib import Path
import json
import numpy as np

run = Path("runs/REPLACE_WITH_THE_PRINTED_RUN_FOLDER")
record = json.loads((run / "metrics.json").read_text())
print(record["config"])
print(record["metrics"])
with np.load(run / "arrays.npz") as arrays:
    print({name: arrays[name].shape for name in arrays.files})
```

## Optional plotting

This requires the optional notebook/plotting environment described in [setup](../docs/setup.md#optional-environments). Do not interrupt the required lesson to install it; numeric interpretation is sufficient.

For a VAE or flow run, execute this Python code after the preceding snippet:

```python
import matplotlib.pyplot as plt

with np.load(run / "arrays.npz") as arrays:
    heldout = arrays["heldout"]
    generated = arrays["generated"]
plt.scatter(*heldout.T, s=8, alpha=0.4, label="held-out")
plt.scatter(*generated.T, s=8, alpha=0.4, label="generated")
plt.axis("equal")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend()
plt.show()
```

Use the same axes for both runs. Check whether any points fall outside those limits rather than silently treating the cropped plot as complete. Inspect coverage of all eight clusters, not just whether individual points look plausible. Plotting a training set alone cannot establish generalization.

## What should I write about a result?

Record the changed setting, controlled settings, run paths, relevant measured quantities and a narrow conclusion. For example: “With the same seed and training settings, changing the Euler budget changed this sample-distance estimate. This does not show the same improvement across other model families or quantify sampler-only speed.” Supply your actual observations; an expected trend is not a substitute for evidence.

Continue in your current [lesson](../lessons/README.md), where each table is tied to a specific question.
