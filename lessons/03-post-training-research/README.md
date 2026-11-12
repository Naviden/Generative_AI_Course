# 03 — Post-training and credible research

Previous: [02 — Diffusion and flow matching](../02-diffusion-flow-matching/README.md) · [Learning path](../README.md)

**Question:** what changes when a model fits preferences, and what evidence is needed to claim improved reasoning or useful generation?

**You will produce:** a DPO interpretation and a one-page research protocol. Follow the [Class 3 agenda](../../docs/syllabus.md#class-3--post-training-and-credible-research). This lesson does not require training an LLM, calling an API or building a production agent.

## Step 1 — Separate adaptation objectives

Read [theory §8](../../docs/theory.md#8-sft-efficient-adaptation-and-dpo). Explain what supervised fine-tuning trains on, what LoRA changes in the parameterization, and what a preference pair supplies. LoRA is not itself a preference-learning objective.

Derive the optimal policy for the KL-regularized reward objective, then substitute it into the pairwise preference model. Identify why the context-dependent partition function cancels and why the reference policy stays fixed.

**Worked checkpoint:** when policy and reference are equal, compute the DPO loss and the sign of its derivative with respect to the chosen-minus-rejected log-probability margin. Attempt [Checkpoint 7](../checkpoints.md#7-dpo-at-the-reference-policy) before running.

## Step 2 — Inspect a preference experiment small enough to understand

```bash
python -m genai_course.labs dpo --steps 100 --beta 0.1
python -m genai_course.labs dpo --steps 100 --beta 1
```

Read `dpo_loss` in [objectives.py](../../genai_course/objectives.py), then `dpo_lab` in [labs.py](../../genai_course/labs.py). This is **one context, three one-token completions, and a repeated preference for completion 0 over completion 1**. Completion 2 is unpaired. There is no held-out context or factuality test in this toy experiment.

| Record from `metrics` | β = 0.1 | β = 1 |
|---|---|---|
| `loss_first` | | |
| `loss_last` | | |
| `final_probabilities[0]` — preferred | | |
| `final_probabilities[1]` — rejected | | |
| `final_probabilities[2]` — unpaired | | |
| `kl_policy_reference` | | |

Record the run paths too. Initially the three probabilities are `1/3`, and the loss is approximately `0.6931`. Inspect whether the observed preference is fitted and what happens to the unpaired option. Lower training loss is not evidence of factuality, safety or generalization. Explain why finite-step β comparisons need not reproduce an exact reward-objective optimum; see [Checkpoint 8](../checkpoints.md#8-what-the-dpo-lab-establishes).

## Step 3 — Evaluate reasoning, retrieval and tool use

Read [theory §9](../../docs/theory.md#9-reasoning-and-recent-research) and [§10](../../docs/theory.md#10-retrieval-tools-and-responsible-experiments).

For reasoning, distinguish final-answer correctness, robustness to meaning-preserving perturbations, process validity and inference cost. If a system gets more attempts, a higher pass@k alone is not a fixed-compute comparison.

For retrieval, design a comparison using no retrieval, ordinary retrieval and supplied correct evidence. Explain which comparison diagnoses retrieval failure and which tests generation with adequate evidence. For tools, distinguish task completion from success of an individual action. These are design exercises; no external write access or private data is required.

## Step 4 — Critique one paper with a concrete question

Use the [Class 3 critique packet](../../docs/readings.md#class-3-critique-packet-no-full-paper-homework-required): R18 by default or R19 as the stated alternative. Read the selected abstract and relevant experimental setup/table in class; full-paper homework is not required.

Write five short notes: the authors' claim, the experimental unit, the baseline, one possible confound, and one experiment that could overturn your initial interpretation. Treat preprint claims as claims to examine. The core labs are not replications of these papers.

## Step 5 — Turn an observation into a research protocol

Choose one of the four topics in [Assessment C](../../docs/assessment.md#c-class-3-research-design-and-defense--40-points): fixed-compute flow sampling, noisy preferences, robustness under text perturbation, or retrieval under distracting evidence. Use the [learning-log template](../learning-log.md#lesson-03--research-protocol).

Specify a falsifiable hypothesis, primary outcome, independent unit, splits, baseline, controlled budget, ablation, uncertainty method and one data/ethical issue. Defend it in two minutes within the scheduled group activity. The [worked design check](../checkpoints.md#9-a-research-protocol-that-can-fail) shows the required level of precision.

## Finish the course

Complete the individual final reflection in [Assessment D](../../docs/assessment.md#d-individual-final-reflection--10-points-5-minutes): one supported conclusion, one unsupported conclusion, and the next measurement needed.

You have completed the core path when you can explain the objectives, interpret the four experiments, and defend your evaluation design. A longer implementation project is an [optional extension](../../docs/assessment.md#optional-research-extension-outside-the-course), not an unfinished required lesson.

[Return to the learning path](../README.md).
