# 03 — Learning from preferences

[Download as PDF](https://github.com/Naviden/Generative_AI_Course/raw/refs/heads/main/reading-group/pdf/03-preferences-and-evidence.pdf)

Previous: [02 — Learning to generate by denoising](02-generative-models.md) · [Reading-group home](README.md)

**Central question:** if training makes a model agree better with preference labels, what would justify claiming that its responses are better?

This meeting follows one paper through three questions: what the preference loss learns, what its evaluation measures, and how to test one limitation.

## Before the meeting: one paper, selected passages

**Rafailov et al. — [Direct Preference Optimization: Your Language Model is Secretly a Reward Model, v3](https://arxiv.org/pdf/2305.18290v3).** Read the short explanation in section 1 below, then these passages only:

1. **The abstract and Figure 1 with its caption.** Identify the difference between fitting a separate reward model and directly updating the language model from preferences.
2. **In §4, equation (7) and the final paragraph labelled “DPO outline”.** Read that paragraph through to the start of §5. Use the worked example below to understand what equation (7) scores; its derivation and gradient equation are optional.
3. **§6.4 in full, including Table 2 and its caption.** Focus on the DPO column, its comparator and the three win-rate rows. Stop before §7. The appendices and the other experimental subsections are not required.

Before the meeting, submit **one 200–300-word note on DPO**: a specific claim, its supporting passage or result, and one justified question. The [grading rubric](README.md#assessment--30-points) applies only to this core reading and discussion. The questions below are for the meeting, not additional written homework.

## 1. What is learned from a preference pair?

A language model assigns probabilities to possible responses given a prompt. In this paper that distribution is called a **policy**. **Post-training** changes an already trained model using additional examples. Here an example contains a prompt, two candidate responses, and a label saying which response was preferred.

For example, the prompt might ask for a one-sentence summary. Response A is short and accurate; response B repeats details and misses the main point. A reviewer chooses A. This is a comparison for that prompt, not a numerical score for every possible response. A different reviewer or judging instruction could produce a different label.

DPO uses these labelled pairs to update the policy directly, relative to a **reference policy**: a fixed starting model whose probabilities are kept for comparison. A separate **reward model**, in the alternative pipeline shown in Figure 1, would learn to assign scores to responses before those scores are used to train the policy. DPO avoids that separate fitted scoring model.

### A small worked version of the loss

Imagine exactly three possible complete responses: A, B and C. The reference assigns each probability `1/3`. A candidate updated policy assigns `(0.50, 0.25, 0.25)`. For the labelled pair “A preferred to B”, compare how much each probability changed relative to the reference:

- A's ratio is `0.50 / (1/3) = 1.50`.
- B's ratio is `0.25 / (1/3) = 0.75`.

With the paper's adjustable coefficient $`\beta`$ set to 1, equation (7) scores this pair using the preference probability `1.50 / (1.50 + 0.75) = 2/3`. Its negative log-loss is $`-\ln(2/3)\approx0.4055`$. At the unchanged reference, both ratios are 1, the pair probability is `1/2`, and the loss is $`-\ln(1/2)\approx0.6931`$. The candidate thus fits this preference label better. These are constructed probabilities, not measurements from the paper or a claimed training trajectory.

**Reading equation (7):** $`x`$ is the prompt; $`y_w`$ and $`y_l`$ are the preferred and rejected responses. $`\pi_\theta`$ is the adjustable policy; $`\pi_{\mathrm{ref}}`$ is the fixed reference. $`\beta`$ scales the comparison. The function $`\sigma`$ converts the difference of scaled log-ratios into a number between zero and one: the predicted preference probability. The outer negative logarithm penalizes a low probability for the observed preference label; the average combines training pairs. The ratio calculation above is this same rule at $`\beta=1`$, without needing to manipulate the full equation.

### Discussion 1 — Does fitting a preference mean generating a correct answer?

**Passage:** §4, equation (7) and “DPO outline”. In the worked example, is `2/3` the probability of generating A? What does the lower loss establish, and what does it leave open?

<details>
<summary>Worked answer</summary>

The probability of generating A is 0.50. The value `2/3` instead models the preference for A over B using their changes relative to the reference; these are different questions. Lower loss means better agreement with this labelled pair. It does not establish factual correctness, agreement with every reviewer, or improvement on unseen prompts. A collection of labels favouring confident but incorrect answers could reward the wrong property. Checking unseen responses requires separate evidence.

</details>

## 2. What does the evaluation measure?

A **win rate** is the fraction of comparisons in which an evaluator selects one system's response over a comparator's. In a hypothetical set of 100 comparisons with no ties, 58 wins gives `58/100 = 58%`. This is not “58% of responses are correct”: two incorrect responses can still be ranked against each other.

In **Table 2**, the DPO summaries are compared against summaries from **PPO**, an alternative preference-training method, generated at temperature zero. Responses are generated one **token** (piece of text) at a time. **Temperature** controls randomness in token selection; zero here means choosing the most probable next token. DPO uses temperature 0.25 in this comparison. State these settings when describing the comparison.

| Evaluator in Table 2 | Reported DPO win rate against PPO at temperature zero |
|---|---|
| GPT-4, simple judging instruction | 47% |
| GPT-4, instruction also asking for conciseness | 54% |
| Human judgments | 58% |

The 47% and 54% values are evaluations of generated summaries, not the preference probabilities in the training loss. The added conciseness instruction changes what the judge is asked to value. “Better” therefore needs a stated task, comparator and judging criterion. The figures above are reported results; our earlier 100-comparison example only illustrated how a percentage works.

### Discussion 2 — What can change the apparent winner?

**Passage:** §6.4 and Table 2. What changes between the two GPT-4 rows, and what stays fixed? Does the human result establish that DPO is generally more truthful?

<details>
<summary>Worked answer</summary>

The judging instruction changes; the comparison remains between the DPO and PPO summaries from this evaluation. DPO's reported win rate moves from 47% to 54%, a difference of 7 percentage points. The human rate of 58% supports a preference result for these summaries under the study's conditions. It does not measure truthfulness across arbitrary tasks. We also cannot assess statistical uncertainty from the three percentages alone; we would need the underlying judgments and how examples and raters were sampled.

</details>

## 3. Turn one limitation into a testable proposal

A useful research proposal identifies a claim, changes one relevant condition, and specifies what observations would distinguish possible explanations. It does not need a new architecture or a completed experiment.

For example: **does the judging instruction explain an apparent advantage, or do readers prefer the summaries under both instructions?** Compare the same saved pairs of summaries under a simple instruction and an instruction emphasizing conciseness. Keeping the pairs fixed prevents new generated responses from being mistaken for an evaluator effect. Hide the model names, randomize which response appears first, and include human judgments under explicit instructions.

Record both win rates and which individual pairs change winner. Repeated judgments can reveal disagreement, but repeated judgments of the same pair are not additional independent examples. Include different held-out source texts to investigate whether the pattern extends beyond the chosen pairs. This is a proposed study, not a report of results.

### Discussion 3 — What would your proposed comparison establish?

**Passage:** Table 2 and the explanation of the two judging instructions in §6.4. State one question, what you would hold fixed, what you would measure and one conclusion the experiment could not support.

<details>
<summary>Worked answer</summary>

Question: does adding a conciseness instruction change which model wins on the same summary pairs? Hold the source texts and generated summaries fixed; change only the judging instruction and counterbalance response order. Measure the change in win rate and the fraction of pairs whose winner changes. If DPO moves from below to above 50%, the declared winner is sensitive to that instruction on those pairs. This would not establish which summaries are factually correct; factual consistency needs a separate check against the source text. Any proposed numerical outcome must be labelled hypothetical.

</details>

Your individual final proposal may develop this example or another specific claim from the assigned DDPM or DPO excerpts. The two-page limit and scoring criteria are on the reading-group home page; optional material is not needed for full credit.

## Optional further reading

[Extended notes](further-reading/03-preferences-and-evidence.md) preserve the derivations, RAG discussion and other extensions. None is required for assessment.

[Reading-group home and assessment](README.md#assessment--30-points).
