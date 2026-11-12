# Checkpoint answers and interpretation guidance

Attempt each question in its lesson before opening the corresponding answer. These are checks of reasoning, not numerical targets for every machine. [Return to the learning path](README.md).

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

## 3. Causal attention

Weights have shape `(2, 6, 6)` and output has shape `(2, 6, 4)`. Mask key positions `j > i` before softmax for each query position `i`. Weight rows sum to one; a fully masked row is not a valid probability row, although this causal square mask always leaves the diagonal available.

Changing only the final value cannot affect earlier causal outputs. The final position itself is excluded from that comparison because it can attend to its own value. Testing a change to a permitted past value provides a positive control: otherwise an implementation returning a constant could appear to pass a future-invariance check.

The CLI performs the value perturbation. The named automated test also checks future-key changes and past sensitivity. Neither proves that token shifts, embeddings, positional information, padding masks or a full language-model training pipeline are correct.

## 4. Diffusion variance

The product $`\bar\alpha_2`$ is `0.72`. Thus the conditional mean is $`\sqrt{0.72}\,x_0`$, covariance is $`0.28I`$, and standard deviation per coordinate is $`\sqrt{0.28}`$, approximately `0.5292`.

The two noise-variance contributions are $`\alpha_2(1-\alpha_1)+(1-\alpha_2)=0.08+0.20`$. Adding their standard deviations would be wrong. The conditional score for the sampled noisy observation is $`-\epsilon/\sqrt{1-\bar\alpha_t}`$. A population squared-loss noise predictor learns a conditional expectation, which connects that conditional target to the marginal noisy-data score.

## 5. Flow path and sampling

The path is $`x_t=(1-t)x_0+tx_1`$, and its derivative is $`x_1-x_0`$. It starts at noise and ends at data. Squared-error regression learns $`\mathbb{E}[x_1-x_0\mid x_t=x,t]`$ at its population optimum. Different endpoint pairs can pass through the same intermediate region; the network does not memorize a single supplied pair's velocity as a universal field.

Euler uses $`h=1/K`$ and $`x_{k+1}=x_k+h\,v(x_k,t_k)`$ from time 0 to 1. This conditional-path construction is not automatically optimal-transport endpoint coupling, and straight conditional paths do not force every learned marginal trajectory to be straight.

## 6. A controlled flow comparison

Hold the seed, data, architecture, optimizer, training updates and initial sampling noise fixed; change only the sampling budget. Check that configuration and data hashes agree as expected. Equal training histories and held-out velocity MSE are useful consistency checks in the deterministic local setup.

A smaller sample distance is evidence about those generated samples under the chosen metric. More Euler steps can reduce integration error but cannot repair a wrong velocity field. Finite sample variation and limited random projections also affect the metric. The real-vs-real reference is a diagnostic, not a strict lower bound that every generator must exceed.

Reported `seconds` includes training and evaluation. To claim a sampler-only speed advantage, time sampling separately with a defined measurement protocol. Heun uses two velocity evaluations per step; equal steps are not equal cost. Three seeds provide a sensitivity exercise, not a strong confidence statement.

## 7. DPO at the reference policy

Let $`m`$ be the chosen-minus-rejected policy log-probability difference minus the same reference difference. At the reference policy, $`m=0`$, so the loss is $`-\log\sigma(0)=\log 2`$, approximately `0.6931`.

Its derivative with respect to $`m`$ at zero is $`-\beta/2`$. Gradient descent therefore increases the preference margin. A change to one model logit also changes normalized probabilities of other outputs; the logits are not independent log probabilities.

## 8. What the DPO lab establishes

The experiment can show that the implementation fits a repeated preference in a three-outcome policy and moves relative to a fixed reference. The unpaired option is affected by normalization even though it is absent from the pair. It provides no evidence of held-out-context generalization, factuality or reasoning ability.

$`\beta`$ changes gradient scale and the preference objective, while 100 optimization steps need not find the exact reward-objective optimum. Do not infer a universal monotonic relationship between $`\beta`$ and the measured finite-step preferred probability or KL. A real LM experiment also needs completion log probabilities, prompt/padding masks, independent evaluation contexts and a treatment of conflicting preference labels.

## 9. A research protocol that can fail

Example hypothesis: “At a fixed budget of 50 velocity evaluations, Heun sampling reduces mean held-out sliced-W1 relative to Euler for this 2D model family.” Compare Euler with 50 steps to Heun with 25, using the same independently trained checkpoints and paired initial noise. Hold the data distribution and metric directions fixed for the primary comparison.

The primary outcome is the paired difference in sample distance. Independent training seeds represent independent training runs; generated points within a checkpoint do not replace those replications. Specify the number of replications before the final comparison, based on a pilot and the precision needed. Separate development choices from the final held-out evaluation. An uncertainty interval spanning zero would not establish the claimed improvement.

Record configurations, source versions, data hashes and all runs, including unfavorable ones. Limit the claim to the tested model, distribution, budget and metric. A failure to improve is an informative result. This is an optional implementation extension; the required lesson asks for a defensible protocol, not its full execution.
