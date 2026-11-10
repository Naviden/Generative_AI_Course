# Validation record — 10 November 2026

## Executed locally

Validated on macOS 27.0.1, Apple Silicon, Python 3.12.16, PyTorch 2.15.0 and NumPy 2.5.4, CPU with one torch thread. A second clean environment was installed from `requirements/core.lock`, then the local package was installed and checked. The lock was generated with uv 0.13.7 using the official PyPI and PyTorch CPU indexes.

- **26 tests passed** in the clean locked environment (7.32 seconds). Tests cover distribution-reference KL/NLL, gradient flow, attention equivalence and causal invariance, DPO signs/stability, flow endpoints, Euler convergence, evaluation diagnostics, reproducible runs and invalid CLI inputs.
- Dependency compatibility check passed.
- Local Markdown targets, Python syntax and notebook schemas passed. Notebook cells were **not** executed by this check.
- Timing check passed: three continuous 240-minute agendas, with 20 break minutes each; total 720 scheduled minutes and 660 active minutes.
- All eleven configurations below completed in the clean environment with finite metrics. The initial attention check had row-sum error 1.19e-7 and zero measured future leakage.

## Classroom experiment record

The table is measured output, not a convergence or generalization guarantee. Time measures experiment execution after imports, excluding environment setup and interpreter startup. Values can vary on other platforms. VAE/flow use 300 updates; DPO uses 100. All use seed 0 unless stated.

| Experiment | Setting | Main diagnostic | Value | Seconds |
|---|---|---|---:|---:|
| attention | causal | Future leakage | 0.000000 | 0.0279 |
| vae | beta=1.0 | Held-out negative ELBO (MC) | 3.351373 | 0.0924 |
| vae | beta=0.1 | Held-out negative ELBO (MC) | 4.801694 | 0.0852 |
| dpo | beta=0.1 | Preferred action probability | 0.892184 | 0.0264 |
| dpo | beta=1.0 | Preferred action probability | 0.900123 | 0.0292 |
| flow | seed=0, Euler steps=5 | Sample sliced W1 | 0.269943 | 0.0743 |
| flow | seed=0, Euler steps=50 | Sample sliced W1 | 0.248755 | 0.0779 |
| flow | seed=1, Euler steps=5 | Sample sliced W1 | 0.269659 | 0.0774 |
| flow | seed=1, Euler steps=50 | Sample sliced W1 | 0.218484 | 0.0782 |
| flow | seed=2, Euler steps=5 | Sample sliced W1 | 0.265329 | 0.0732 |
| flow | seed=2, Euler steps=50 | Sample sliced W1 | 0.252498 | 0.0780 |

The raw [validation records](validation-results.json) include configuration, environment, source hashes and data hashes. They record the pre-commit working tree on base `f90f602`; `git_dirty: true` is expected. Package source hashes identify the tested implementation independently of the later documentation commit. Recorded metrics are examples, not hard-coded test expectations.

At the tested budgets, more Euler steps reduced sample distance for all three illustrated seeds; this small experiment does not establish a universal result. In the beta-VAE comparison, a smaller training KL weight improved reconstruction but worsened the ordinary held-out negative ELBO and sample distance. The toy DPO policy increased its preferred action probability; it has no held-out contexts and establishes no language-model performance claim.

## Automated checks and limitations

The GitHub workflow is configured for Ubuntu and Python 3.11, 3.12 and 3.13, and publishes the four lab runs as artifacts. Consult the pull request checks for actual remote run results; workflow configuration alone is not evidence of successful execution.

Full historical TensorFlow training, downloaded BERT/GPT-2 models, legacy notebook execution, Windows and accelerator execution were not validated locally. See the [repository audit](repository-audit.md) for known limitations. Optional dependency groups are not covered by core tests. Primary-source pages were reviewed for the selected readings; this was not an exhaustive literature review or an independent replication of their claims. Existing third-party PDFs were not content-reviewed or relicensed.
