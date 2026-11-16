# 01 — Objectives, VAEs and attention

Previous: [00 — Get ready](../00-get-ready/README.md) · [Learning path](../README.md) · Next: [02 — Diffusion and flow matching](../02-diffusion-flow-matching/README.md)

**Question:** what does a model learn when we minimize its loss, and how can we check that the code implements the intended mathematics?

**During this lesson:** learn how likelihood, a VAE objective and causal attention work; then work through a derivation, two VAE runs and an attention check. The explanations and examples below come before the practice tasks. These activities take place within Lesson 01; nothing on this page is a pre-lesson assignment.

## Step 1 — Understand what a loss measures

### A. Learn what the loss measures

**Read these three subsections of theory §1, in order:**

1. [From examples to a probability model](../../docs/theory.md#from-examples-to-a-probability-model): what the model assigns probabilities to, and why unlikely observed outcomes incur a larger loss.
2. [Worked example: fitting frequencies](../../docs/theory.md#worked-example-fitting-frequencies): follow the calculation for eight A outcomes and two B outcomes.
3. [Entropy, cross-entropy and the direction of KL](../../docs/theory.md#entropy-cross-entropy-and-the-direction-of-kl): how average log-loss relates to fitting a distribution, including the continuous-density example.

Read from each named heading up to the next heading. **Stop before “Autoregression: generating a joint outcome one part at a time”** and return here. You are learning these concepts now; there is no question to submit before reading them. If a symbol is unfamiliar, use [the probability explanation](../../docs/foundations.md#2-probability-conditioning-and-expectation) or [the log-loss explanation](../../docs/foundations.md#5-logarithms-losses-and-kl-divergence).

The frequency example illustrates the objective: probabilities 0.8 and 0.2 fit the observed eight-to-two ratio better than equal probabilities. This is a statement about those data. A model could memorize training examples and still predict new examples poorly. Keep training fit, held-out fit and generated-sample usefulness separate when interpreting the labs.

### B. Work through a specific three-token sequence

**Now read only [Autoregression: generating a joint outcome one part at a time](../../docs/theory.md#autoregression-generating-a-joint-outcome-one-part-at-a-time)**, including its numerical example. Stop at the heading “2. Latent variables, the ELBO and the VAE”. The worked categorical exercise at the end extends the frequency example you just read; it includes a complete solution and is not an additional assignment at this step.

Use the toy sequence **“the cat sleeps”**, treating each word as one token. We consider exactly three positions and omit start/end tokens for this calculation. Real tokenizers can split words differently.

The chain rule breaks the probability of these three observed tokens into three factors:

```math
p_\theta(\text{the},\text{cat},\text{sleeps})
=p_\theta(\text{the})\,
 p_\theta(\text{cat}\mid\text{the})\,
 p_\theta(\text{sleeps}\mid\text{the},\text{cat}).
```

Each factor asks how probable the next observed token is given the preceding observed tokens. **Teacher forcing** supplies that recorded prefix during training:

| Position | Prefix supplied to the model during training | Target token whose probability is scored |
|---|---|---|
| 1 | Empty prefix in this toy example | `the` |
| 2 | `the` | `cat` |
| 3 | `the cat` | `sleeps` |

The target is used to compute the loss; it must not be included in the input used to predict itself. Even if the model would have sampled `dog` at position 2, the training prediction at position 3 still receives `the cat`. During generation, if it actually samples `the dog`, its next prediction conditions on `the dog` instead. That difference is what the term teacher forcing describes.

If the three target probabilities are 0.5, 0.2 and 0.1, multiplying them gives 0.01 for the three-token sequence. This connects the notation to an actual computation.

### C. Short practice after the explanation

Use the new toy sequence **“I like tea”**, again one word per token. In your notes:

1. Write its three conditional-probability factors, following the worked example above.
2. State the training prefix and target for position 3.
3. With successive target probabilities 0.4, 0.5 and 0.25, compute the joint probability.

**Check your work:**

```math
p_\theta(\text{I},\text{like},\text{tea})
=p_\theta(\text{I})\,
 p_\theta(\text{like}\mid\text{I})\,
 p_\theta(\text{tea}\mid\text{I},\text{like})
=0.4\times0.5\times0.25=0.05.
```

The training prefix is `I like` and the target is `tea`. The purpose is to connect one sequence, its conditioning information and its probability—not to invent a model or derive an unfamiliar result unaided. Continue to Step 2 when these three answers make sense.

## Step 2 — Derive the VAE objective

**Read all four subsections of theory §2**, from [Why introduce an unobserved variable?](../../docs/theory.md#why-introduce-an-unobserved-variable) through **The likelihood determines what reconstruction means**, including the worked Gaussian KL calculation and lab connection. **Stop at “3. Contrasts: GANs and normalizing flows”.** Read them in the following order and use each explanation immediately:

1. **Why introduce an unobserved variable?** Use its distribution table to distinguish prior, decoder, true posterior and encoder. For the two-dimensional data point being reconstructed, identify which distribution can generate a latent value before the point is observed and which uses the observed point.
2. **Why the lower bound exists.** Follow the supplied derivation line by line. Annotate the rewrite as an expectation, the application of Jensen's inequality and the posterior-gap identity. Explain in words why the bound can be loose. You can keep the derivation visible; the task is to understand and then reproduce its reasoning.
3. **A differentiable random sample.** Follow the scalar example with mean 1, standard deviation 2 and noise 0.5. Explain how changing the mean changes the sampled value while the noise draw stays fixed.
4. **The likelihood determines what reconstruction means.** Follow the worked KL for mean `(1, 2)` and zero log variance: the coordinate contributions are 0.5 and 2, so the per-example KL is 2.5 nats. Then calculate two variations: duplicating both latent coordinates gives 5 nats; putting two identical examples in the batch leaves their mean KL at 2.5. Explain why these are different operations.

**Record:** an annotated ELBO derivation, the roles of the four distributions and the two reduction calculations. [Checkpoint 1](../checkpoints.md#1-vae-reductions) explains the same reductions if you need another pass.

Open these files in this order; search for the function or class name rather than reading every file:

1. [objectives.py](../../genai_course/objectives.py): `diagonal_gaussian_kl` and `gaussian_nll` — which dimensions are summed?
2. [models.py](../../genai_course/models.py): `VAE.forward` — where does parameter-free noise enter, and where do gradients flow?
3. [labs.py](../../genai_course/labs.py): the VAE branch of `generative_lab` — where are event losses averaged over a batch?

The toy dataset contains eight Gaussian clusters in **two dimensions**. This experiment is about probability objectives, not image generation. The decoder likelihood has fixed variance `0.1`, and generated observations include that noise.

## Step 3 — Compare two VAE runs

The preceding worked example showed how the training objective combines reconstruction NLL and a weighted KL penalty. Reducing the weight from 1 to 0.1 makes a unit of KL less costly relative to reconstruction. It may allow more information in the latent representation, but does not guarantee better prior samples.

Using that explanation, note which of reconstruction NLL, KL and sample coverage you expect might change. This is a hypothesis to compare with the output, not a result you must obtain. Then run:

```bash
python -m genai_course.labs vae --steps 300 --seed 0 --beta 1
python -m genai_course.labs vae --steps 300 --seed 0 --beta 0.1
```

Use existing runs if their settings match. Open the `metrics.json` in each printed folder; the quantities below are inside its `metrics` object. See [how to inspect outputs](../../labs/outputs.md) if this is your first run.

| Record | $`\beta=1`$ | $`\beta=0.1`$ | Why it matters |
|---|---|---|---|
| Run-folder name | | | Links the claim to its settings and evidence |
| `heldout_reconstruction_nll_mc` | | | Fit of the Gaussian reconstruction distribution |
| `heldout_kl` | | | Posterior-to-prior divergence, summed over latent dimensions |
| `heldout_negative_elbo_mc` | | | Comparable NLL + KL diagnostic for both runs |
| `sample_sliced_w1` | | | Sample-distribution diagnostic; smaller is better for this metric |

**Interpret:** which measurements moved together, and which disagreed? Is the $`\beta=0.1`$ training objective itself a negative ELBO? Why can Gaussian NLL be negative? Write a two-sentence conclusion tied to these runs, not a universal claim about $`\beta`$. Consult [Checkpoint 2](../checkpoints.md#2-interpreting-the-vae-comparison) after answering. Exact recorded values are not pass/fail targets.

## Step 4 — Place the VAE among other model families

**Read both subsections of theory §3**, starting at [Learning by discrimination](../../docs/theory.md#learning-by-discrimination) and continuing through **Learning by invertible transformation**, its volume-change example and comparison table. **Stop at “4. Transformers and scaling”.**

Use the supplied comparison table to explain one difference in your own words for each family: why the VAE uses a bound, why a GAN can sample without evaluating a density, and why an invertible flow needs a Jacobian. Add one limitation per family from the accompanying explanation. You are interpreting the worked comparison; no historical GAN program is required.

## Step 5 — Check whether attention can see the future

**Read only the first two subsections of theory §4:** [What attention computes](../../docs/theory.md#what-attention-computes) and [Worked example: masking changes the distribution](../../docs/theory.md#worked-example-masking-changes-the-distribution). **Stop before “From one head to a language model”; Step 6 covers it.**

Follow the worked calculation: allowed scores 0 and $`\log 3`$ become weights 1/4 and 3/4, and values 2 and 6 produce output 5. The future position has weight zero.

Now apply the shape explanation to this lab: queries and keys have shape `(2, 6, 8)` and values `(2, 6, 4)`. The query–key dot products give scores shaped `(2, 6, 6)`; masking and row-wise softmax produce weights of the same shape. Weighting the values then gives output shape `(2, 6, 4)`. Mark the entries with key position $`j>i`$ as masked for query position $`i`$. Record these expected shapes and compare them with the command output:

```bash
python -m genai_course.labs attention
```

You may reuse the setup result. In [objectives.py](../../genai_course/objectives.py), inspect `scaled_attention`: dot product → scaling → mask → row-wise softmax → weighted values. In [labs.py](../../genai_course/labs.py), read `attention_lab` and identify exactly which values are perturbed and which outputs are compared.

Expected diagnostics: output shape `[2, 6, 4]`; future leakage zero; row-sum error near floating-point precision. The CLI changes the final value vector. The broader test below also checks a changed future key and sensitivity to allowed past values:

```bash
python -m pytest tests/test_objectives.py -k causal_future_invariance_and_past_sensitivity -v
```

Read that test in [test_objectives.py](../../tests/test_objectives.py). Explain why future invariance needs a positive control using a past token, and why a passing mask test does not make a complete autoregressive LM. See [Checkpoint 3](../checkpoints.md#3-causal-attention).

## Step 6 — Explain before moving on

**Read the remaining subsection of theory §4, [From one head to a language model](../../docs/theory.md#from-one-head-to-a-language-model), through its final lab connection. Stop at “5. Diffusion and scores”.** This passage explains what additional components a full transformer needs, the compute tradeoffs and why data leakage matters. Use it to name two things the attention lab did not test. The assessment below follows the explanations and experiments completed in Steps 1–5.

Complete [Class 1's exit ticket](../../docs/assessment.md#a-class-1-exit-ticket--25-points-10-minutes): derive the bound, explain the KL reduction, and describe the causal-mask test and its limits. Use [your learning log](../learning-log.md) for the comparison table and conclusions. Full-paper readings R1 and R4 in the [reading guide](../../docs/readings.md) are optional extensions.

**Move on when you can explain the calculations and diagnostics, not merely reproduce the commands.**

Next: [02 — Diffusion and flow matching](../02-diffusion-flow-matching/README.md).
