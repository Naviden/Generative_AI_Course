# 02 — Learning to generate by denoising

[Download as PDF](https://github.com/Naviden/Generative_AI_Course/raw/refs/heads/main/reading-group/pdf/02-generative-models.pdf)

Previous: [01 — Shared foundations](01-foundations.md) · [Reading-group home](README.md) · Next: [03 — Learning from preferences](03-preferences-and-evidence.md)

**Central question:** how can learning to predict added noise produce a generator, and what evidence shows that it works?

This meeting follows one paper through three questions: what is learned, how generation uses it, and what the results establish. The worked examples below supply the background needed for that discussion.

## Before the meeting: one paper, selected passages

**Ho, Jain and Abbeel — [Denoising Diffusion Probabilistic Models, v2](https://arxiv.org/pdf/2006.11239v2).** Read the short explanation in section 1 below, then these passages only:

1. **The abstract and Algorithms 1–2**, labelled “Training” and “Sampling”. Identify which algorithm has access to an observed image and which starts with noise. Use the notation guide below; do not derive the update coefficients.
2. **§3.4, equation (14) and its opening sentence.** Read the equation as a noise-prediction penalty. The rest of §3.4 and the derivations leading to it are optional.
3. **§4.1, its two prose paragraphs and Table 1 with its caption.** In the table, focus on the two “Ours” rows. Stop before §4.2; Table 2 is not assigned.

No other paper sections or appendices are required. Before the meeting, submit **one 200–300-word note on DDPM**: a specific claim, the passage or result supporting it, and one justified question. The [grading rubric](README.md#assessment--30-points) applies only to this core reading and discussion. The three questions below are for the meeting, not additional written homework.

## 1. What does the model learn?

In Lesson 1, training changed a table of probabilities. Here the model learns a rule for changing a noisy image. **Noise** means random perturbations of its pixel values, like adding grain to a photograph. Training deliberately corrupts a real image, so the program knows both the original and the noise it added. The network receives the corrupted image and the noise level, and predicts the added noise. Comparing its prediction with the known noise gives a training loss.

For one image value, use this illustrative corruption rule:

```math
x_t=0.8x_0+0.6\epsilon.
```

Here $`x_0`$ is the original value, $`\epsilon`$ is the noise drawn for this example, and $`x_t`$ is the corrupted value at noise level $`t`$. The coefficients specify how much signal and noise remain. They are chosen for this example, not fitted from it.

If $`x_0=1`$ and $`\epsilon=0.5`$, then $`x_t=0.8(1)+0.6(0.5)=1.1`$. Suppose the network predicts noise 0.2. Its squared error is $`(0.5-0.2)^2=0.09`$. Predicting 0.4 would give $`(0.5-0.4)^2=0.01`$. Lower error means better noise prediction on this example; it does not by itself establish better generated images.

**Reading equation (14):** $`\epsilon_\theta`$ is the network's noise prediction; $`\theta`$ denotes its adjustable weights. The squared norm adds squared prediction errors across image values. $`\mathbb E`$ means an average over examples, noise draws and selected noise levels. The square-root coefficients inside the network input play the roles of 0.8 and 0.6 above. “Take a gradient descent step” in Algorithm 1 means adjust the weights in a direction that locally reduces this penalty; no derivative calculation is required here.

### Discussion 1 — Why is this a learnable task?

**Passage:** Algorithm 1 and equation (14). What is known to the training program, what is shown to the network, and what is the prediction target? Does knowing the injected noise make the task trivial?

<details>
<summary>Worked answer</summary>

In the numerical example, the program knows the original 1 and the noise 0.5. The network is shown the corrupted value 1.1 and its noise level, not the target noise 0.5. It predicts 0.2 and receives penalty 0.09. Many original-value/noise pairs can produce the same corrupted value, so the target is not generally recoverable by simple subtraction from the input. Learning uses patterns across many examples. Knowing the target lets the program score a prediction; it does not give the target to the predictor.

</details>

## 2. How does this become generation?

**During training**, real images provide starting points for constructing prediction tasks. **During generation**, start with fresh noise and repeatedly use the trained predictions to update it. The weights stay fixed. There is no observed clean image to recover; the aim is to produce a new image consistent with patterns learned during training.

In Algorithm 2, $`T`$ is the total number of steps; $`t`$ counts down towards the final image. $`\mathcal N(0,I)`$ denotes standard Gaussian noise: each component is a random value centred on zero with unit variance, meaning unit average squared distance from zero. Values close to zero are more common than very large positive or negative ones. The update uses a noise prediction, schedule-dependent coefficients, and an additional random term. The schedule fixes the noise amounts across steps. We need to follow these roles, not derive the coefficients.

For intuition, rearranging our one-value corruption rule gives the clean-value estimate $`\hat x_0=(x_t-0.6\hat\epsilon)/0.8`$. With input 1.1 and predicted noise 0.2, the estimate is $`(1.1-0.12)/0.8=1.225`$. Predicting 0.4 gives $`(1.1-0.24)/0.8=1.075`$, closer to the known original 1. This calculation illustrates how a prediction affects a reconstructed value. It is **not** the complete multi-step sampling rule in Algorithm 2.

### Discussion 2 — What changes between training and sampling?

**Passages:** Algorithms 1 and 2. Explain why sampling cannot use the known original value from the example. If we run Algorithm 2 twice, what changes and what stays fixed?

<details>
<summary>Worked answer</summary>

The original value existed because we constructed a training example from data. At generation time there is no supplied original: Algorithm 2 starts from noise. Two runs can use different starting noise and different intermediate random draws, producing different outputs. The learned weights and the chosen sampling schedule stay fixed. This is the distinction from Lesson 1: sampling produces observations; training changes the rule used to produce them.

</details>

## 3. What do the reported results establish?

Return to **Table 1**. CIFAR10 is the image dataset used for this comparison. **FID** compares generated and real image collections through numerical image features from a fixed image classifier; smaller is better by that measure. It is not a percentage of correct images, and does not separately establish coverage, factual correctness or human preference.

The paper also reports a test **NLL upper bound**. An upper bound is a ceiling on the true quantity, not necessarily its exact value. “Bits/dim” expresses loss per image value using base-two logarithms; Lesson 1 used natural logarithms. Compare values within the same column, not FID numerically against NLL.

| Table 1 row | FID | Test NLL upper bound, bits/dim |
|---|---|---|
| Original bound objective, fixed variance | 13.51 | 3.70 |
| Simplified noise-prediction objective | 3.17 | 3.75 |

The simplified objective has lower FID, while its reported NLL ceiling is higher. For example, ceilings of 3.70 and 3.75 could hide true values of 3.60 and 3.50, respectively. The ceilings alone therefore cannot prove the ordering of the true NLLs. Those hidden values are hypothetical, not paper results. The paper's Table 1 FID uses training images as the reference collection, whereas the NLL column refers to held-out test images; the two measures answer different questions.

### Discussion 3 — Is “better generator” a supported conclusion?

**Passage:** §4.1 and Table 1. Replace “the simplified objective makes a better generator” with a claim the table supports. Name one additional comparison needed to investigate a different meaning of “better”.

<details>
<summary>Worked answer</summary>

A bounded claim is: “In the reported CIFAR10 comparison, the simplified objective achieves lower FID than the original-bound row with fixed variance.” This is not evidence of superiority on every criterion. To ask about human judgments, compare equal-sized samples from both models in a blinded evaluation with a stated judging question, such as visible image defects. Keep the sampling procedure and dataset comparison consistent, and repeat with additional samples. That would investigate human judgments, not retroactively turn FID into a human score.

</details>

## Optional further reading

[Extended generative-model notes](further-reading/02-generative-models.md) preserve the VAE/ELBO and reparameterization explanations, full diffusion mathematics, flow matching, additional paper questions and computational demonstrations. They are not required reading or assessed topics for this meeting.

Continue with [Lesson 03 — Learning from preferences](03-preferences-and-evidence.md).
