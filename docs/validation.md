# Validation record — 10 November 2026

## Executed locally

Validated on macOS 27.0.1, Apple Silicon, Python 3.12.16, PyTorch 2.15.0 and NumPy 2.5.4, CPU with one torch thread. A second clean environment was installed from `requirements/core.lock`, then the local package was installed and checked. The lock was generated with uv 0.13.7 using the official PyPI and PyTorch CPU indexes.

- **26 tests passed** in the clean locked environment (7.32 seconds). Tests cover distribution-reference KL/NLL, gradient flow, attention equivalence and causal invariance, DPO signs/stability, flow endpoints, Euler convergence, evaluation diagnostics, reproducible runs and invalid CLI inputs.
- Dependency compatibility check passed.
- Local Markdown targets, Python syntax and notebook schemas passed. Notebook cells were **not** executed by this check.
- Timing check passed: three continuous 240-minute agendas.
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


## Guided learning-path update — 12 November 2026

This documentation update adds a numbered student path, lesson checkpoints,
a learning-log template, an output-interpretation guide and orientation notes
in historical folders. Core Python implementations and dependency files are
unchanged. The historical lab heading anchors are retained for book links.

Validation on the existing Apple Silicon / Python 3.12 core environment:

- All 26 tests passed (4.08 seconds).
- All 11 documented configurations completed: attention; VAE beta 1 and 0.1;
  flow budgets 5 and 50 for seeds 0, 1 and 2; DPO beta 0.1 and 1.
- Checked documented array shapes, initial DPO loss, attention diagnostics,
  the VAE NLL-plus-KL identity and identical training histories/data hashes/
  held-out velocity error for each paired flow comparison.
- All 27 installed packages were compatible (`uv pip check` against the
  existing environment); no dependency versions changed.
- The material checker passed, including the unchanged three 240-minute
  agendas. A separate navigation check validated 176 relative file links
  and Markdown section anchors in the then-current repository documents.
- Python inspection examples in the output guide match the generated file
  formats. Optional plotting remains optional; its interactive display was
  not executed for this documentation change. No legacy training was added
  to the executed scope.

This verifies that the guided commands and references match the implementation;
it is not evidence that every student will complete each concept at the same pace.


## Physical archive reorganization — 12 November 2026

Moved the seven historical topic/reference/data folders and the 2023 environment
snapshot into `archive/` (54 tracked files). All 47 non-README files were checked
against their pre-move SHA-256 hashes and are byte-identical. The seven moved
orientation READMEs have updated relative links. No core model implementation,
lesson sequence, current dependency or timetable changed.

Updated active documentation links and the historical-attention regression-test
path. Validation after relocation: 26 tests passed; all four classroom CLI
commands completed; dependency compatibility passed in the installed Python 3.12
core environment; 193 relative file links/section anchors passed a navigation
check; material validation passed for local links, Python syntax, notebook
schemas and the three 240-minute agendas. Historical notebook cells and legacy
training remain outside the executed scope.


## Scheduling wording correction — 12 November 2026

Removed the prescribed break allocation from the README, lesson overview,
instructor guidance and syllabus. The three-lesson structure remains;
discussion and theory allocations fill the existing agenda. Material validation
now checks continuous 240-minute agendas without enforcing a break policy.
