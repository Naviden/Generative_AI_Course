# Generative AI: a short course

**Start with the [numbered learning path](lessons/README.md).** This repository teaches how generative models learn, how to inspect their implementations, and how to evaluate their outputs. It assumes graduate-level machine learning and is organized into **three lessons**.

## What should I do next?

- **I have not installed it:** follow [00 — Get ready](lessons/00-get-ready/README.md). This includes cloning, entering the correct folder, installation and a single smoke check.
- **Installation works:** open [01 — Objectives, VAEs and attention](lessons/01-objectives-vae-attention/README.md). Start at Step 1 and work down the page. Running a lab is only one part of a lesson.
- **I already ran all four commands:** use your saved outputs in `runs/` while following Lesson 1's interpretation tasks. You do not need to reinstall or repeat an unchanged run.
- **I am teaching:** use the same lesson pages with the [timed syllabus](docs/syllabus.md) and [instructor guide](docs/instructor-guide.md).

## Follow this order

| Order | Lesson | What you will do | Evidence you understand it |
|---|---|---|---|
| 00 | [Get ready](lessons/00-get-ready/README.md) | Check prerequisites, install, run attention once | A working environment and a prerequisite self-check |
| 01 | [Objectives, VAEs and attention](lessons/01-objectives-vae-attention/README.md) | Derive an ELBO, compare two VAE runs, inspect a causal mask | Explain the loss terms and test future-token leakage |
| 02 | [Diffusion and flow matching](lessons/02-diffusion-flow-matching/README.md) | Derive the noise marginal, compare sampling budgets | Separate model error, solver error and evaluation noise |
| 03 | [Post-training and research](lessons/03-post-training-research/README.md) | Inspect DPO, critique reasoning/RAG evidence, design an experiment | Make a narrow claim supported by a controlled comparison |

Each lesson identifies **what to read, what to predict before running, the exact command, which output fields matter, which functions to inspect, and when to move on**. Checkpoint answers are available after you attempt the questions. The four labs are small, complete CPU experiments on synthetic data; no GPU, paid service, API key or pretrained-model download is needed after installation. The course does not train a full LLM or image diffusion model.

## Which folders should I open?

Open `lessons/` and follow its numbered folders. All historical topic folders are grouped under `archive/`.

```text
lessons/                   STUDENT PATH — open these in numeric order
  00-get-ready/
  01-objectives-vae-attention/
  02-diffusion-flow-matching/
  03-post-training-research/
  learning-log.md           Copyable worksheet for your own results
  checkpoints.md            Answers and interpretation guidance

genai_course/              IMPLEMENTATION — open functions when a lesson asks
labs/                      REFERENCE — compact lab instructions and output guide
docs/                      REFERENCE — theory, readings, setup and teaching plans
tests/                     VERIFICATION — mathematical and implementation checks
requirements/              CURRENT ENVIRONMENT — core dependencies
scripts/                   MAINTENANCE — material validation
archive/                   HISTORICAL MATERIAL — outside the required lessons
runs/                      YOUR OUTPUTS — created locally, not committed
```

The old model examples, notebooks, PDFs, data and 2023 environment snapshot have been moved into [archive/](archive/README.md). Their contents are preserved. The [historical-material guide](docs/historical-material.md) explains their limitations and optional uses; they are not additional required lessons.

## Reference shelf

Use these when a lesson links to them; you do not need to read the entire shelf before starting.

- [Theory notes](docs/theory.md) and [primary-source reading guide](docs/readings.md)
- [Lab workbook](labs/README.md) and [how to read lab outputs](labs/outputs.md)
- [Prerequisites](docs/prerequisites.md), [setup and troubleshooting](docs/setup.md)
- [Three-class timetable](docs/syllabus.md), [assessment](docs/assessment.md), [instructor guide](docs/instructor-guide.md)
- [Executed validation](docs/validation.md), [repository audit](docs/repository-audit.md), [maintenance](CONTRIBUTING.md)

The book is a standalone reference; these lesson pages guide use of this repository. Existing third-party files retain their own rights; this learning path does not grant redistribution permission or introduce a repository-wide license.
