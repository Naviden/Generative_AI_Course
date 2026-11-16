# Checkpoint answers and interpretation guidance

Keep these worked answers available while following each lesson. Use them to understand an example, then explain the reasoning in your own words. These are checks of reasoning, not numerical targets for every machine. [Return to the learning path](README.md).

## 1. VAE reductions

For log variance $`\ell`$, the diagonal Gaussian KL is

```math
D_{\mathrm{KL}}(q\Vert p)=\frac12\sum_j\left(\mu_j^2+\exp(\ell_j)-1-\ell_j\right).
```

For $`(1,2)`$ with zero log variance, it is $`\frac12(1+4)=2.5`$ nats. Duplicating the latent coordinates gives `5`: they jointly describe one event, so their KL contributions sum. Duplicating an identical example in a batch leaves the batch mean unchanged. Averaging over latent coordinates would change the KL's weight relative to reconstruction.

The ELBO gap is KL from the approximate posterior to the **true posterior**. It vanishes when those distributions agree almost everywhere under the relevant support/integrability conditions. This is different from the KL to the prior in the training objective.

## 2. Interpreting the VAE comparison

At $`\beta=1`$ the training objective estimates negative ELBO. With $`\beta=0.1`$ the KL penalty is downweighted; the objective is modified. Both runs still report `heldout_negative_elbo_mc` as **unweighted NLL + KL**, allowing that diagnostic to be compared. The subscript `mc` indicates Monte Carlo estimation, not exact log likelihood.

A lower $`\beta`$ may permit more information in the latent representation and improve reconstruction while moving the posterior farther from the prior. Generated samples come from the prior, so improved reconstructions alone do not establish improved generation. Check what actually happened in your runs rather than treating that tendency as a guarantee.

A Gaussian density can exceed one. Its negative log density can therefore be negative; the units and variance matter. This is not the same as assigning a discrete outcome probability greater than one. A good interpretation names the changed quantities and limits the conclusion to the measured data, model and runs.

**Worked interpretation using illustrative measurements.** Suppose reducing $`\beta`$ from 1 to 0.1 changes held-out reconstruction NLL from 0.6 to −0.2, KL from 2.8 to 5.0, and sample distance from 0.30 to 0.41. Negative ELBO changes from $`0.6+2.8=3.4`$ to $`-0.2+5.0=4.8`$. A suitable two-sentence answer is: “The lower KL weight improved reconstruction but increased posterior-to-prior divergence and worsened this sample-distance measurement. These two runs do not establish a universal effect or better unconditional generation.” Use your actual lab values in your own conclusion.

## 3. Causal attention

Weights have shape `(2, 6, 6)` and output has shape `(2, 6, 4)`. Mask key positions `j > i` before softmax for each query position `i`. Weight rows sum to one; a fully masked row is not a valid probability row, although this causal square mask always leaves the diagonal available.

Changing only the final value cannot affect earlier causal outputs. The final position itself is excluded from that comparison because it can attend to its own value. Testing a change to a permitted past value provides a positive control: otherwise an implementation returning a constant could appear to pass a future-invariance check.

The CLI performs the value perturbation. The named automated test also checks future-key changes and past sensitivity. Neither proves that token shifts, embeddings, positional information, padding masks or a full language-model training pipeline are correct.

**Worked perturbation.** For a query at position 2, weights `(1/4, 3/4, 0)` and values `(2, 6, 100)` give output 5. Changing the future value to 200 leaves it at 5; changing the allowed first value to 6 makes it 6. A constant-output implementation could satisfy the first comparison but would fail the second. A separate token-shift test would still be needed to check that an LM predicts the next token rather than exposing it in the input.

## 4. Diffusion variance

The product $`\bar\alpha_2`$ is `0.72`. Thus the conditional mean is $`\sqrt{0.72}\,x_0`$, covariance is $`0.28I`$, and standard deviation per coordinate is $`\sqrt{0.28}`$, approximately `0.5292`.

The two noise-variance contributions are $`\alpha_2(1-\alpha_1)+(1-\alpha_2)=0.08+0.20`$. Adding their standard deviations would be wrong. The conditional score for the sampled noisy observation is $`-\epsilon/\sqrt{1-\bar\alpha_t}`$. A population squared-loss noise predictor learns a conditional expectation, which connects that conditional target to the marginal noisy-data score.

**Worked score.** For a conditional normal with mean 1, variance 1/4 and observation 1.5, the standardized noise is $`(1.5-1)/(1/2)=1`$. The conditional score is $`-1/(1/2)=-2`$, agreeing with the derivative $`-(1.5-1)/(1/4)`$. [Theory §5](../docs/theory.md#two-dynamics-with-the-same-ideal-marginals) also supplies a complete reverse-SDE/ODE table with numerical drifts and the sign of the time step.

## 5. Flow path and sampling

The path is $`x_t=(1-t)x_0+tx_1`$, and its derivative is $`x_1-x_0`$. It starts at noise and ends at data. Squared-error regression learns $`\mathbb{E}[x_1-x_0\mid x_t=x,t]`$ at its population optimum. Different endpoint pairs can pass through the same intermediate region; the network does not memorize a single supplied pair's velocity as a universal field.

Euler uses $`h=1/K`$ and $`x_{k+1}=x_k+h\,v(x_k,t_k)`$ from time 0 to 1. This conditional-path construction is not automatically optimal-transport endpoint coupling, and straight conditional paths do not force every learned marginal trajectory to be straight.

**Worked path and solver.** From noise −1 to data 3, $`x_t=-1+4t`$ and target velocity is 4. At times 0, 1/4 and 1, positions are −1, 0 and 3. With this constant field, four Euler steps of size 1/4 add one each time and reach 3 exactly. For ambiguous target velocities +2 and −2 with equal conditional probability, predicting $`a`$ incurs loss $`a^2+4`$; its optimum is zero, with positive residual 4.

## 6. A controlled flow comparison

Hold the seed, data, architecture, optimizer, training updates and initial sampling noise fixed; change only the sampling budget. Check that configuration and data hashes agree as expected. Equal training histories and held-out velocity MSE are useful consistency checks in the deterministic local setup.

A smaller sample distance is evidence about those generated samples under the chosen metric. More Euler steps can reduce integration error but cannot repair a wrong velocity field. Finite sample variation and limited random projections also affect the metric. The real-vs-real reference is a diagnostic, not a strict lower bound that every generator must exceed.

Reported `seconds` includes training and evaluation. To claim a sampler-only speed advantage, time sampling separately with a defined measurement protocol. Heun uses two velocity evaluations per step; equal steps are not equal cost. Three seeds provide a sensitivity exercise, not a strong confidence statement.

**Worked conclusion.** Suppose valid paired runs have distances `(0.30, 0.26)`, `(0.25, 0.24)` and `(0.20, 0.22)` for `(5, 50)` steps. The differences are −0.04, −0.01 and +0.02, averaging −0.01. Thus the average decreased by 0.01, but one run worsened. Neither universal improvement nor sampler-only speed follows. [Theory §7](../docs/theory.md#paired-comparisons-and-the-unit-of-uncertainty) shows how to resample these paired differences and why three units give weak uncertainty evidence. These figures illustrate an answer; they are not measured repository results.

## 7. DPO at the reference policy

Let $`m`$ be the chosen-minus-rejected policy log-probability difference minus the same reference difference. At the reference policy, $`m=0`$, so the loss is $`-\log\sigma(0)=\log 2`$, approximately `0.6931`.

Its derivative with respect to $`m`$ at zero is $`-\beta/2`$. Gradient descent therefore increases the preference margin. A change to one model logit also changes normalized probabilities of other outputs; the logits are not independent log probabilities.

**Worked scalar update.** If $`\beta=1`$ and a toy optimizer updates the margin itself with step size 0.1, the gradient −1/2 takes $`m`$ from 0 to 0.05. The loss becomes $`\log(1+\exp(-0.05))\approx0.6685`$, below 0.6931. The lab instead updates policy logits, so this scalar arithmetic explains the sign rather than predicts its exact trajectory.

## 8. What the DPO lab establishes

The experiment can show that the implementation fits a repeated preference in a three-outcome policy and moves relative to a fixed reference. The unpaired option is affected by normalization even though it is absent from the pair. It provides no evidence of held-out-context generalization, factuality or reasoning ability.

$`\beta`$ changes gradient scale and the preference objective, while 100 optimization steps need not find the exact reward-objective optimum. Do not infer a universal monotonic relationship between $`\beta`$ and the measured finite-step preferred probability or KL. A real LM experiment also needs completion log probabilities, prompt/padding masks, independent evaluation contexts and a treatment of conflicting preference labels.

**Worked policy update.** For the uniform three-outcome reference, start at logits `(0, 0, 0)`, prefer 0 over 1, set $`\beta=1`$ and take one logit-gradient step of size 0.5. Logits become `(0.25, −0.25, 0)` and probabilities approximately `(0.4192, 0.2543, 0.3265)`. The third logit stayed fixed but its normalized probability decreased. [The full calculation](../docs/theory.md#worked-example-interpreting-the-dpo-margin) explains the cancellation and gradients. Fitting this one comparison cannot tell us which response is factually correct or what happens on a new prompt.

## 9. A research protocol that can fail

A research protocol states what you will compare, what evidence will be collected, and what result would challenge the hypothesis. The following fills **every field requested in the learning log**. It is a proposed experiment; its outcomes have not been measured. Heun's two-evaluation update is explained with numbers in [theory §6](../docs/theory.md#training-error-and-integration-error-are-different).

| Required item | Worked design and why it is chosen |
|---|---|
| Question and hypothesis | Does Heun reduce mean held-out sliced-W1 relative to Euler at 50 velocity evaluations for the repository's eight-cluster 2D flow model? This fixes the model family, outcome and computation being compared. |
| Primary outcome | For each trained model, calculate “Heun distance minus Euler distance”. A negative mean favours Heun. Preselect this metric so a disappointing result cannot be replaced by a more favourable metric after inspection. |
| Independent unit | An independently trained run, identified by its seed. Evaluate both solvers on each run; individual generated points are not independent training replications. |
| Data provenance and splits | Generate synthetic data with the repository's `mixture` distribution. Use disjoint seeds/data for training, development and final evaluation. During development choose the evaluation sample size and check implementation; freeze those choices before generating final evaluation data. Keep hashes and seed assignments. |
| Baseline and fixed budget | Euler with 50 steps versus Heun with 25 steps: both use 50 velocity evaluations. Within each pair keep model weights, initial noise, held-out data and projection directions identical. Equal evaluation counts do not guarantee equal wall time. |
| Ablation | Also compare Euler20 with Heun10 at a budget of 20 evaluations, reporting it as a secondary budget-sensitivity analysis. Keep the 50-evaluation comparison primary. |
| Replications and uncertainty | A pilot is a small development experiment used to plan the final study. Use it to estimate the spread of paired differences and choose the final run count for a predeclared precision goal. The standard error measures the sampling variability of an estimated mean, whereas standard deviation measures the spread of the individual differences. For example, a desired standard error 0.01 and pilot standard deviation 0.04 suggest $`n\approx(0.04/0.01)^2=16`$ independent pairs, because standard error of a mean is approximately $`s/\sqrt n`$. This illustrative calculation is not an actual pilot result or a guarantee of reliably detecting a real effect. Freeze the count before final evaluation; report the mean and a paired bootstrap interval over independent runs, including unfavourable runs. |
| Data/ethical issue | The experiment uses generated points rather than personal data. Record the generator and avoid presenting this as evidence that a real image/text model is private, fair or safe. |
| Practical limitation | One model family, one toy distribution and a finite-projection metric restrict the claim. Inspect cluster coverage as a secondary diagnostic; a smaller scalar distance can hide specific failures. |
| Contradicting outcome | A nonnegative mean fails the claimed direction in the measured sample. An interval spanning zero is inconclusive about a population improvement; it is not evidence of equality. |
| Objection and response | “Equal evaluation counts do not establish speed.” Agree: make no speed claim from the primary comparison. If speed is of interest, add repeated sampler-only timings after warmup on matched hardware; record timing variation separately from training. |

**Worked final reflection with hypothetical results.** Suppose the final mean difference is −0.02 but its uncertainty interval is `(−0.05, +0.01)`. The measurements show a lower sample mean for Heun, with uncertainty spanning no advantage. They do not establish a reliable population improvement, faster sampling or superiority on images. A next measurement could use more preplanned independent replications or test another distribution, depending on which uncertainty the claim needs to address. The interval numbers are illustrative, not an executed result.

The required activity is to design and explain the experiment. Implementing Heun and executing the proposed study remain an optional extension.
