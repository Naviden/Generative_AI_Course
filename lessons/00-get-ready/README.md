# 00 — Get ready

[Learning path](../README.md) · Next: [01 — Objectives, VAEs and attention](../01-objectives-vae-attention/README.md)

**Purpose:** introduce the concepts used in the lessons and prepare the lab environment. The foundations are part of the course material.

## Step 1 — Learn the foundations

Open [Foundations used in the lessons](../../docs/foundations.md). It explains probability, Gaussian distributions, tensor shapes, losses, gradients, neural networks, Python and data splits, with worked examples. Read unfamiliar topics now or follow the links back to them as you progress. The practice questions have explained answers and no pass mark.

## Step 2 — Install from inside the cloned repository

If you already cloned and installed successfully, skip to Step 3. Otherwise, follow the [setup guide](../../docs/setup.md), including the `cd Generative_AI_Course` command after cloning. The repository root is the folder containing `pyproject.toml`, `requirements/`, `genai_course/` and this README.

## Step 3 — Confirm that one small computation works

From the repository root, in your active environment:

```bash
python -m genai_course.labs attention
```

If you already ran it, use that result. Look for `output_shape: [2, 6, 4]`, `future_leakage: 0.0`, and a `row_sum_error` near zero (typically below `1e-6`). The command prints a unique run-folder path. These are implementation diagnostics, not a claim that the code is a complete language model. Lesson 01 explains them.

## Ready to continue when

- The setup tests pass and the attention command completes.
- You can find the run folder named by the command.
- You know where to find the foundation explanations and worked answers when a concept is unfamiliar.

**Now open [Lesson 01, Step 1](../01-objectives-vae-attention/README.md#step-1--understand-what-a-loss-measures).** Do not run all remaining labs as an installation task.
