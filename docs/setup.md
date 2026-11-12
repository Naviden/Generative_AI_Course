# Installation and compute

## First enter the repository

If you have not cloned it yet, run these shell commands on separate lines:

```bash
git clone https://github.com/Naviden/Generative_AI_Course.git
cd Generative_AI_Course
```

If you already cloned it, enter the existing `Generative_AI_Course` folder instead of cloning a second copy. The next commands must run from the folder containing `pyproject.toml` and `requirements/core.lock`, not its parent folder.

## Supported core path

Use Python 3.12, a clean virtual environment, and run commands from the repository root. Python 3.11–3.13 is declared compatible; only environments listed in [validation](validation.md) have actually been executed. The newest system Python need not have compatible scientific-package wheels.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/core.lock
python -m pip install --no-deps -e .
python -m pip check
python -m pytest
python scripts/check_materials.py
python -m genai_course.labs attention
```

**Setup complete? Open [Lesson 01, Step 1](../lessons/01-objectives-vae-attention/README.md#step-1--understand-what-a-loss-measures).** The attention command above is only a smoke check. The lesson explains what to read and how to interpret experiments before you run the remaining labs.

Windows: use `py -3.12 -m venv .venv` and `.venv\Scripts\activate` (Command Prompt) or `.venv\Scripts\Activate.ps1` (PowerShell). The lock selects CPU-only PyTorch wheels on Linux/Windows from the official PyTorch index and the standard macOS wheel on macOS. For other accelerators, use [PyTorch's official installation page](https://pytorch.org/get-started/locally/) in a separate environment. The package version and platform must be recorded; accelerator builds are not interchangeable bit for bit.

`requirements/core.lock` is a resolved, exact-version teaching environment, not a guarantee that every version is the newest available. `python -m pip install -r requirements.txt` instead resolves current versions within the supported bounds in `pyproject.toml`; rerun tests before teaching with it. The historical freeze in `requirements/legacy-2023.txt` is archival and should not be installed as the recommended environment.

## Offline classroom preparation

Internet is needed for the initial package installation. After that, the four core labs use generated numerical data only and run offline. Run all four once before teaching. Download wheels on a matching OS/Python architecture in advance if classroom access is unreliable:

```bash
python -m pip download -r requirements/core.lock -d wheelhouse
# On the same platform/Python version, after installing this local project:
python -m pip install --no-index --find-links wheelhouse -r requirements/core.lock
```

The local editable-project installation may also need setuptools; install it while online or include a compatible setuptools wheel in the wheelhouse and use `--no-build-isolation` for the project install. Do not assume one platform's wheelhouse supports another platform.

## Optional environments

These are supplementary, outside the required classroom path, and are not claimed to have been fully executed by core CI:

```bash
python -m pip install -e '.[notebooks]'   # JupyterLab and plotting
python -m pip install -e '.[classical]'   # original GMM/HMM/NLP examples
python -m pip install -e '.[llm]'         # original BERT/GPT-2 downloads
```

Use a separate environment for `python -m pip install -e '.[tensorflow]'`; legacy TensorFlow programs may require further migration. Follow [TensorFlow's platform instructions](https://www.tensorflow.org/install/pip). The old environment combined `tensorflow` and `tensorflow-macos` and froze transitive dependencies; it is no longer the portable default. Optional extras are dependency declarations, not a promise that all legacy examples are working. See the [audit](repository-audit.md) before running them.

## Budgets and outputs

The core CLI forces CPU execution and one torch thread. Default networks have only a few thousand parameters, 1,024 training points, 256 held-out points and 300 update steps. Reserve five classroom minutes for running and inspecting each experiment, with longer discussion slots in the syllabus. Actual timings are recorded per run and summarized in [validation](validation.md); hardware changes timings.

Each invocation creates a unique directory with `metrics.json` and `arrays.npz`. Metadata includes seed, hyperparameters, elapsed time, Python/package/platform versions, source commit, dirty-worktree indicator and train/held-out hashes. Training and evaluation data use distinct random seeds. A beta-VAE objective with beta other than one is not labelled an ordinary negative ELBO. Flow and VAE objective values are not directly comparable. These short runs demonstrate mechanisms, not publication-level results.


## Troubleshooting

| Symptom | Likely cause | Next action |
|---|---|---|
| `requirements/core.lock` not found; no `pyproject.toml` | Commands ran in the parent folder | Enter `Generative_AI_Course`, then resume installation there |
| `No module named pytest` or `genai_course` after an installation error | Required installation steps did not complete | Resolve the first error, activate the intended environment, then rerun dependency and editable-project installation |
| Commands worked earlier but imports now fail | A new terminal is using another Python environment | Activate the environment you used for this checkout; check `python -m pip --version` |
| A shell complains about pasted Python syntax | A Python example was pasted into the shell | Run it in a Python session or save it as a `.py` script |

If you created `.venv` in the parent folder before entering the repository, that environment can still be used: its location does not prevent installation, but the working directory must be the repository root. Avoid switching between environments without noticing. If another error persists, record the first failing command and its complete message before retrying later commands.
