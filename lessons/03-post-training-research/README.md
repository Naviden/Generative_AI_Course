# 03 — Post-training and credible research

Previous: [02 — Diffusion and flow matching](../02-diffusion-flow-matching/README.md) · [Learning path](../README.md)

**Question:** what changes when a model fits preferences, and what evidence is needed to claim improved reasoning or useful generation?

**You will produce:** a DPO interpretation and a one-page research protocol. Follow the [Class 3 agenda](../../docs/syllabus.md#class-3--post-training-and-credible-research). This lesson does not require training an LLM, calling an API or building a production agent.

## Step 1 — Separate adaptation objectives

**Read all four subsections of theory §8**, starting at [Three different choices: data, objective and parameters](../../docs/theory.md#three-different-choices-data-objective-and-parameters) and ending with **Worked example: interpreting the DPO margin**. **Stop at “9. Reasoning and recent research”.** Work through them in this order:

1. **Three different choices: data, objective and parameters.** Use the supplied examples to make three entries in your notes: SFT learns from demonstrated completions; preferences compare two completions for one context; LoRA restricts which parameter updates are learned. Explain why LoRA can be used with either objective.
2. **Why keep a reference policy?** Follow the reward-minus-KL objective and the differentiation argument with a normalization multiplier. Reproduce the two-completion example: equal reference probabilities, reward difference $`\log 3`$ and coefficient 1 give optimal probabilities 0.75 and 0.25. Annotate the role of the normalizing factor.
3. **From rewards to pairwise classification.** Follow the substitution into the Bradley–Terry model. Write the reward difference for a chosen and rejected completion of the **same prompt**, then cross out the identical prompt-dependent normalization terms. Keep the supplied derivation open as you explain this cancellation.
4. **Worked example: interpreting the DPO margin.** Follow the numerical example first. Then set policy equal to reference: the adjusted log-probability margin is zero, the loss is about 0.6931, and the derivative with respect to the margin is $`-\beta/2`$. Explain why gradient descent increases the margin. [Checkpoint 7](../checkpoints.md#7-dpo-at-the-reference-policy) provides the same reasoning in compact form.

These are guided calculations within the lesson. Record the annotated derivation and interpretations, then use them to understand the experiment in Step 2.

## Step 2 — Inspect a preference experiment small enough to understand

```bash
python -m genai_course.labs dpo --steps 100 --beta 0.1
python -m genai_course.labs dpo --steps 100 --beta 1
```

Read `dpo_loss` in [objectives.py](../../genai_course/objectives.py), then `dpo_lab` in [labs.py](../../genai_course/labs.py). This is **one context, three one-token completions, and a repeated preference for completion 0 over completion 1**. Completion 2 is unpaired. There is no held-out context or factuality test in this toy experiment.

| Record from `metrics` | $`\beta=0.1`$ | $`\beta=1`$ |
|---|---|---|
| `loss_first` | | |
| `loss_last` | | |
| `final_probabilities[0]` — preferred | | |
| `final_probabilities[1]` — rejected | | |
| `final_probabilities[2]` — unpaired | | |
| `kl_policy_reference` | | |

Record the run paths too. Initially the three probabilities are `1/3`, and the loss is approximately `0.6931`. Inspect whether the observed preference is fitted and what happens to the unpaired option. Lower training loss is not evidence of factuality, safety or generalization. Explain why finite-step $`\beta`$ comparisons need not reproduce an exact reward-objective optimum; see [Checkpoint 8](../checkpoints.md#8-what-the-dpo-lab-establishes).

## Step 3 — Evaluate reasoning, retrieval and tool use

Read and discuss these specific passages in order:

| Read exactly | Worked example to follow | What to record after reading |
|---|---|---|
| [What would count as improved reasoning?](../../docs/theory.md#what-would-count-as-improved-reasoning) and **Worked example: more attempts versus a better attempt**; stop before **Reading a research claim critically** | A per-attempt success probability of 0.2 gives about 0.6723 for at least one success in five independent attempts | State what changed—the number of attempts—and why selecting a correct answer needs a verifier |
| [Retrieval supplies evidence at answer time](../../docs/theory.md#retrieval-supplies-evidence-at-answer-time) and **Worked diagnosis: isolate retrieval from evidence use**; stop before **Tools change what a model can do** | The hypothetical handbook task scores 40/100 without retrieval, 55/100 with ordinary retrieval and 85/100 with sufficient evidence supplied | Name one measurement of missing evidence and one measurement of failure to use available evidence |
| [Tools change what a model can do](../../docs/theory.md#tools-change-what-a-model-can-do) through the end of theory §10 | The simulated record-update task distinguishes successful search from a correct final update | List the conditions that establish completion: correct record, correct field change, preservation of unrelated fields and an accurate report |

Use these supplied cases before designing a new experiment in Step 5. No tool execution or external access is needed for this step.

## Step 4 — Critique one paper with a concrete question

**First read [Reading a research claim critically](../../docs/theory.md#reading-a-research-claim-critically), the remaining subsection of theory §9. Stop at “10. Retrieval, tools and responsible experiments”.** It explains how budget, data and scoring choices can affect a headline result.

Then read the complete [critique packet](../../docs/readings.md#class-3-critique-packet-no-full-paper-homework-required), stopping at **Updating this list**. Use its R18 summary as the default case and identify the claim, the perturbation being studied and one proposed control. In a taught session, examine the primary-source abstract and experimental passage supplied by the instructor. In self-study, the packet supports a preliminary critique; explicitly distinguish questions raised by that summary from conclusions supported by experiments you have actually inspected. Full-paper reading remains optional.

Write five short notes: the authors' claim, the experimental unit, the baseline, one possible confound, and one experiment that could overturn your initial interpretation. Treat preprint claims as claims to examine. The core labs are not replications of these papers.

## Step 5 — Turn an observation into a research protocol

Choose one of the four topics in [Assessment C](../../docs/assessment.md#c-class-3-research-design-and-defense--40-points): fixed-compute flow sampling, noisy preferences, robustness under text perturbation, or retrieval under distracting evidence. Use the [learning-log template](../learning-log.md#lesson-03--research-protocol).

**Read the complete [worked design check](../checkpoints.md#9-a-research-protocol-that-can-fail) first**, from its heading to the end of the checkpoint page. It develops one flow-sampling hypothesis into a matched-compute experiment. Use it as a model for your selected topic.

Fill the learning-log template with a falsifiable hypothesis, primary outcome, independent unit, splits, baseline, controlled budget, ablation, uncertainty method and one data/ethical issue. For each entry, explain the choice using the examples from this lesson. In a taught session, present and discuss the protocol; in self-study, write one plausible objection and the measurement that would address it.

## Finish the course

Complete the individual final reflection in [Assessment D](../../docs/assessment.md#d-individual-final-reflection--10-points-5-minutes): one supported conclusion, one unsupported conclusion, and the next measurement needed.

You have completed the core path when you can explain the objectives, interpret the four experiments, and defend your evaluation design. A longer implementation project is an [optional extension](../../docs/assessment.md#optional-research-extension-outside-the-course), not an unfinished required lesson.

[Return to the learning path](../README.md).
