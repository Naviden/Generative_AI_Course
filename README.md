# Generative AI: a 12-hour doctoral short course

**Format: three 4-hour classes.** Each 240-minute meeting includes two 10-minute breaks: 12 scheduled hours, 11 hours of active teaching. The course assumes graduate-level machine learning; it is a focused research introduction, not a substitute for a semester of generative modelling.

**Curriculum reviewed: 10 November 2026.** Foundational results are taught alongside selected 2025–2026 research. New preprints are explicitly distinguished from established results. No claim of exhaustive frontier coverage is made.

## Start here

1. Read the [prerequisites and diagnostic](docs/prerequisites.md).
2. Follow the [setup guide](docs/setup.md) before the first class.
3. Use the [three-class timetable](docs/syllabus.md), [theory notes](docs/theory.md), and [lab workbook](labs/README.md) during teaching.
4. Consult the [reading list](docs/readings.md), [assessment](docs/assessment.md), and [instructor guide](docs/instructor-guide.md).

| Class | Focus | In-class evidence of learning |
|---|---|---|
| 1 · 4 hours | Likelihood, latent variables, VAE; autoregressive transformers | Derive the ELBO, audit a VAE objective, test causal attention |
| 2 · 4 hours | Diffusion, scores, flow matching; sampling and evaluation | Derive a denoising target, train a 2D flow, compare solver budgets |
| 3 · 4 hours | SFT, preference optimization, reasoning, retrieval and research evaluation | Derive DPO, run a preference experiment, defend an evaluation protocol |

GANs and normalizing flows provide conceptual comparisons. Multimodality, LoRA/QLoRA, sparse attention, diffusion language models and agents receive short research discussions or optional readings; full implementations of these topics are outside the 12-hour scope.

## Run the supported labs

Use Python 3.12 (the supported range is 3.11–3.13). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements/core.lock
python -m pip install --no-deps -e .
python -m pytest
python scripts/check_materials.py
python -m genai_course.labs attention
python -m genai_course.labs vae --steps 300 --beta 1
python -m genai_course.labs flow --steps 300 --sampling-steps 50
python -m genai_course.labs dpo --steps 100 --beta 0.1
```

The four core labs run on CPU with synthetic data, without API keys, model downloads, paid services, or a GPU. Each run writes its settings, environment, source revision, metrics and arrays into a new `runs/` subfolder. See [setup](docs/setup.md) for installation alternatives and compute expectations. Smoke tests check correctness and execution, not research-level performance.

## Repository map and support status

- `genai_course/`, `labs/`, `tests/`: the supported, automated-test teaching path.
- `docs/`: course notes, timing, readings, assessment and reproducibility standards.
- `Traditional Generative Models/`, `Neural Generative Models/`, `Transformers/`, `LM/`, `other/`: earlier examples, retained for reference. Some are incomplete sketches or require downloads. **Do not run all legacy files as a course setup test.** See the [file-level audit](docs/repository-audit.md).
- `requirements/legacy-2023.txt`: historical environment snapshot, not the supported installation.
- `Books and Tutorials/` and existing PDFs: inherited third-party resources. The core course uses linked primary sources; this refresh does not grant redistribution rights to those files or add a repository-wide license.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the release checks and refresh policy, and [validation](docs/validation.md) for what was actually executed.
