# 02 — How generative models work

Previous: [01 — Shared foundations](01-foundations.md) · [Reading-group home](README.md) · Next: [03 — Preferences and research evidence](03-preferences-and-evidence.md)

**Central question:** when does the quantity we can optimize tell us what we want to know about a generator?

This page contains the explanation, derivations, worked examples and discussion record. No separate theory or lab page is required. The group can work through the material together; full-paper reading is optional. The computational demonstration near the end is optional too.

The route is **latent-variable inference → denoising → learned transport → testing claims**. In Lesson 01, a loss measured how well a model fitted observations. Here we distinguish a tractable bound, a prediction loss and a sample-distribution diagnostic.

**Notation reminder:** $`x`$ is an observation, $`z`$ an unobserved variable, $`\theta`$ generative-model parameters and $`\phi`$ encoder parameters. Expectation is a probability-weighted average. A Gaussian’s mean gives its centre and its variance its squared spread; a standard normal has mean zero and variance one. KL is an expected log ratio comparing distributions and is generally asymmetric.

## 1. A latent variable makes generation easy and inference hard

**Paper question:** a VAE replaces a hard likelihood calculation with an optimizable lower bound. Which approximation is being made, and what does a better bound establish?

### Why introduce an unobserved variable?

A latent variable represents variation that helps explain an observation but is not directly supplied as a label. For a handwritten digit it might represent aspects of shape or style, although an unsupervised model is not guaranteed to learn human-interpretable factors. For the two-dimensional cluster dataset, latent space provides a distribution that a decoder can transform into a distribution of observations.

A **variational autoencoder (VAE)** learns both a generative distribution and a tractable approximation to its inference problem. Here tractable means that the required sampling and probability calculations can be performed efficiently; an exact definition alone does not make a quantity tractable. “Variational” means optimizing over a chosen family of approximating distributions. A **prior** describes latent values before observing $`x`$; a **posterior** describes them after observing $`x`$. A **decoder** maps latent values to an observation distribution, while an **encoder** maps an observation to an approximate latent distribution. A neural network implements these maps by composing learned linear transformations with nonlinear functions. Training repeatedly evaluates a loss, differentiates it and updates parameters.

The generative story has two steps: draw $`z`$ from a prior $`p(z)`$, then draw $`x`$ from a decoder distribution $`p_\theta(x\mid z)`$. The decoder outputs the parameters of a distribution, not necessarily a deterministic reconstruction. To score an observed $`x`$, however, we must account for every latent value that could have produced it. Summing or integrating out an unobserved variable is called **marginalization**; the resulting observation probability is the **marginal likelihood**, also called **evidence**:

```math
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,\mathrm{d}z.
```

That integral is generally hard for a nonlinear neural decoder. Bayes' rule defines the true posterior $`p_\theta(z\mid x)`$: after seeing the observation, which latent explanations are plausible? Computing it involves the same difficult marginal likelihood. The encoder $`q_\phi(z\mid x)`$ is a tractable approximation to that posterior. Its parameters are shared across observations, so one network learns to perform inference for many inputs; this is **amortized inference**.

| Distribution | Question it answers | Role |
|---|---|---|
| Prior $`p(z)`$ | What latent values can be drawn before seeing an observation? | Used to generate new samples |
| Decoder $`p_\theta(x\mid z)`$ | What observations are plausible for this latent value? | Defines the generative model with the prior |
| True posterior $`p_\theta(z\mid x)`$ | What latent values explain this observation under the model? | Usually intractable inference target |
| Encoder $`q_\phi(z\mid x)`$ | What tractable distribution approximates that inference? | Used in training and reconstruction |

### Why the lower bound exists

The **support** is the region where a distribution can place probability. To divide by the encoder density, it must be positive wherever the joint density contributes to the integral. For continuous variables, “almost everywhere” allows exceptions on sets of probability zero. We also need the expectations used below to exist; writing a formal difference of two infinite quantities is not a valid derivation.

A function is **concave** if its value at an average is at least the average of its values. For log, two equally likely positive values 2 and 8 give $`\log((2+8)/2)=\log5\approx1.6094`$, whereas $`(\log2+\log8)/2=\log4\approx1.3863`$. Jensen's inequality extends this averaging rule to arbitrary distributions.

Multiply and divide the joint density by the encoder density inside the marginalization integral. This rewrites an integral as an expectation under a distribution from which we can sample. For the equality below, the encoder must cover the relevant joint-density support. Jensen's inequality then says that the logarithm of an average is at least the average logarithm, because log is concave. This is the source of the bound; it is not an extra empirical assumption about neural networks.

Insert any $`q_\phi(z\mid x)`$ with suitable support into $`\log p_\theta(x)`$:

```math
\begin{aligned}
\log p_\theta(x)
&=\log\mathbb{E}_{q_\phi(z\mid x)}
\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]\\
&\geq\mathbb{E}_{q_\phi(z\mid x)}
\left[\log p_\theta(x,z)-\log q_\phi(z\mid x)\right].
\end{aligned}
```

The right side is the evidence lower bound $`\mathcal{L}(x)`$. Rearranging gives

```math
\log p_\theta(x)=\mathcal{L}(x)
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p_\theta(z\mid x)\right).
```

For $`p_\theta(x,z)=p_\theta(x\mid z)p(z)`$,

```math
\mathcal{L}(x)
=\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p(z)\right).
```

The gap is nonnegative and vanishes only when the approximate posterior equals the true posterior almost everywhere. Optimizing $`\theta`$ and $`\phi`$ changes both the model and the approximation.

**Fill in the rearrangement.** Bayes' rule gives $`\log p_\theta(z\mid x)=\log p_\theta(x,z)-\log p_\theta(x)`$. Insert this into $`\mathbb{E}_q[\log q-\log p_\theta(z\mid x)]`$. The log evidence is constant with respect to $`z`$, so it comes out of the expectation, leaving $`D_{\mathrm{KL}}(q\Vert p_\theta(z\mid x))=\log p_\theta(x)-\mathcal{L}(x)`$. Likewise, splitting $`\log p_\theta(x,z)`$ into $`\log p_\theta(x\mid z)+\log p(z)`$ produces the reconstruction-minus-prior-KL form.

**Worked bound with every probability specified.** For one observed binary outcome $`x=1`$, let $`z`$ be A or B with equal prior probabilities, and let the decoder assign $`p(x=1\mid A)=0.36`$, $`p(x=1\mid B)=0.04`$. Then:

| Quantity | Calculation | Result |
|---|---|---|
| Joint masses for the observed outcome | $`0.5(0.36),\ 0.5(0.04)`$ | `(0.18, 0.02)` |
| Evidence | $`0.18+0.02`$ | `0.20` |
| True posterior | Divide each joint mass by `0.20` | `(0.9, 0.1)` |
| ELBO with encoder `(0.5, 0.5)` | $`0.5\log(0.18/0.5)+0.5\log(0.02/0.5)`$ | `−2.1203` |
| Posterior gap | $`0.5\log(0.5/0.9)+0.5\log(0.5/0.1)`$ | `0.5108` |
| Gap identity | $`-2.1203+0.5108\approx\log0.2`$ | `−1.6094` |

With encoder `(0.9, 0.1)`, both ratios of joint to encoder mass equal `0.2`, so the ELBO is exactly $`\log0.2`$ and the gap vanishes. This finite example makes the same inference issue visible without a difficult integral. It illustrates the bound, not the continuous reparameterization used in the lab.

The two terms in the practical ELBO have different jobs. The expected log decoder likelihood rewards latent explanations that make the observation probable. A **regularizer** is a penalty favouring certain solutions alongside data fit. The KL to the **prior** acts as a regularizer: it penalizes encoding that observation in a distribution very different from the one used for unconditional generation. Without this pressure, an encoder could put useful codes in regions that prior samples almost never visit: reconstruction could work while generation fails.

The KL in the **gap identity** instead compares the encoder with the **true posterior**. Do not confuse these two KLs. Matching the true posterior makes the bound tight; matching the prior can remove information about the input. For fixed model parameters, suppose the log evidence is −3 nats and the posterior gap is 0.7 nats. Then the ELBO is −3.7. Improving inference so the gap is 0.2 raises the bound to −3.2 without changing the model's evidence at all. Joint training may change both quantities.

### A differentiable random sample

A **Gaussian** is the normal distribution, with mean $`\mu`$ setting its centre and variance $`\sigma^2`$ measuring average squared deviation. A standard normal has mean zero and variance one. Standard deviation is the square root of variance. In several dimensions, the covariance matrix contains variances on its diagonal and cross-coordinate dependence off the diagonal. A diagonal Gaussian has independent coordinates; $`I`$ is the identity matrix with unit diagonal. The notation $`\mathrm{diag}(v)`$ puts the entries of vector $`v`$ on a diagonal. In one dimension the Gaussian density is $`(2\pi\sigma^2)^{-1/2}\exp(-(z-\mu)^2/(2\sigma^2))`$; for independent coordinates the joint density is the product of their scalar densities. These are the objects used in the next formulas.

A diagonal Gaussian encoder predicts a mean vector and one log variance per latent coordinate. Here $`\ell_j=\log\sigma_j^2`$; therefore its standard deviation is $`\exp(\ell_j/2)`$, not $`\exp(\ell_j)`$. Predicting log variance allows an unconstrained network output while keeping the resulting variance positive. The diagonal assumption makes computation simple but excludes posterior correlations in these coordinates.

For $`q=\mathcal{N}(\mu,\mathrm{diag}(\exp\ell))`$ and $`p=\mathcal{N}(0,I)`$:

```math
D_{\mathrm{KL}}(q\Vert p)
=\frac12\sum_j\left[\mu_j^2+\exp(\ell_j)-1-\ell_j\right].
```

The formula follows by taking the expected log ratio of two Gaussian densities, one coordinate at a time. In one coordinate, $`\mathbb{E}_q[(z-\mu)^2]=\sigma^2`$ and $`\mathbb{E}_q[z^2]=\mu^2+\sigma^2`$ give $`D_{\mathrm{KL}}=\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2)`$. Independent-coordinate log densities add, giving the displayed sum after substituting $`\sigma_j^2=\exp\ell_j`$.

A **gradient** collects the derivatives of the loss with respect to parameters. **Gradient descent** updates parameters in the opposite direction to their loss gradient, scaled by a positive learning rate. For loss $`J(w)=(w-3)^2`$ at $`w=0`$, the derivative is −6; a step with learning rate 0.1 gives $`w=0-0.1(-6)=0.6`$ and reduces loss from 9 to 5.76. The chain rule propagates derivatives through a composition by multiplying local derivatives. In the next expression, $`\odot`$ multiplies matching coordinates. The random draw $`\epsilon`$ does not depend on the learned parameters.

Use

```math
z=\mu+\exp(\ell/2)\odot\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I),
```

to move sampling into parameter-free noise and obtain pathwise gradients. This requires differentiable transformations and conditions permitting differentiation under the expectation. The minibatch estimator still has Monte Carlo variance.

**Monte Carlo estimation** replaces an expectation with an average over random draws. For example, if a quantity is 1 or 3 with equal probability, its expectation is 2, but a single draw returns 1 or 3. A one-draw estimator can therefore be unbiased—its repeated-draw average equals the target—while being noisy on any particular update. A minibatch averages over a subset of observations rather than the entire dataset.

Reparameterization means drawing standard noise first, then stretching and shifting it using the encoder outputs. Holding the sampled noise fixed while differentiating gives a differentiable path from the loss back to the mean and scale. For one coordinate with mean 1, standard deviation 2 and sampled noise 0.5, the latent sample is 2. Increasing the mean by 0.1 moves this sample by 0.1; increasing the standard deviation by 0.1 moves it by 0.05. These changes explain how a random sample can still carry a pathwise gradient.

### The likelihood determines what reconstruction means

For a scalar observation with decoder mean $`m`$ and variance $`v>0`$, the Gaussian density is $`(2\pi v)^{-1/2}\exp(-(x-m)^2/(2v))`$. Taking its negative logarithm gives $`(x-m)^2/(2v)+\tfrac12\log(2\pi v)`$. This calculation explains both the squared-error term and the normalizing constant below; $`m_j`$ denotes the decoder mean for observation coordinate $`j`$.

A **Bernoulli** distribution instead describes a binary observation: $`p(x=1)=a`$, $`p(x=0)=1-a`$. Its negative log likelihood is $`-x\log a-(1-x)\log(1-a)`$, called binary cross-entropy. For $`x=1,a=0.8`$ it is $`-\log0.8\approx0.2231`$. A continuous coordinate and a binary event require different likelihood models.

A squared-error reconstruction objective corresponds, up to scaling and constants, to a Gaussian observation model with fixed variance. It encodes a particular notion of error: deviations are penalized quadratically relative to that variance. A Bernoulli observation model instead assigns probabilities to binary outcomes. Choosing a reconstruction loss is therefore a modelling decision about observations, not merely a choice of a convenient tensor operation.

Our lab has a Gaussian decoder with fixed variance $`v=0.1`$. Per-example NLL is

```math
\frac12\sum_j\left[\frac{(x_j-m_j)^2}{v}+\log(2\pi v)\right].
```

We sum over observed coordinates and latent dimensions, then average over the batch. Averaging KL over latent coordinates instead would silently change the relative regularization as latent dimension changes. For binary observations a Bernoulli likelihood is appropriate; binary cross-entropy is not a universal reconstruction loss.

A beta-VAE optimizes $`\text{reconstruction NLL}+\beta D_{\mathrm{KL}}(q\Vert p)`$. At $`\beta=1`$ this is an unbiased one-sample estimator of the negative ELBO; at other weights it is a modified objective. Posterior collapse means $`q`$ is close to the prior and the decoder can ignore z; a low reconstruction loss alone does not diagnose it.

**Worked check:** $`\mu=(1,2)`$, $`\ell=(0,0)`$ yields $`D_{\mathrm{KL}}=2.5`$. Duplicating latent coordinates doubles KL. Averaging two identical examples leaves the batch mean unchanged.

With unit variance, the variance terms in each coordinate cancel, leaving half the squared mean. Thus the two coordinates contribute 0.5 and 2 nats. A per-example reconstruction NLL of 1.2 would give negative ELBO 3.7; a KL weight of 0.1 would instead give training objective 1.45. The reported unweighted negative ELBO remains 3.7. These arithmetic values illustrate why comparing differently weighted training objectives can be misleading.

### Discuss the two different KL terms

**Question:** for fixed model parameters, log evidence is −3 and the encoder-to-true-posterior gap falls from 0.7 to 0.2. What changed? Would matching the encoder to the prior mean the same thing?

**Worked response:** the ELBO rises from −3.7 to −3.2 while evidence stays −3. Inference improved; the generator did not change. Matching the true posterior makes the bound tight. Matching the prior is a different condition and may remove information about the observed input. Support and integrability conditions are needed for the derivation; a symbolic subtraction of infinities is not justified.

## 2. Denoising learns a reverse process

**Paper question:** the injected noise is known during training. Why can’t the predictor always recover that exact draw, and why is a noise-prediction loss not automatically exact likelihood?

### Learn to reverse a controlled corruption process

Destroying structure is easy: repeatedly add random noise to an observation. Generating structure is harder. Diffusion modelling turns generation into learning a sequence of local reverse transitions, using a forward corruption process that we specify ourselves. At low noise levels the model learns fine corrections; at high noise levels it must rely more on the population structure learned from data.

A **denoising diffusion probabilistic model (DDPM)** learns to reverse a prescribed sequence of noisy transitions. Here $`q`$ names the fixed forward corruption distribution; it is not the VAE encoder from the VAE discussion. A **marginal** distribution averages out variables we are not keeping: $`q(x_t\mid x_0)`$ averages over the intermediate states while retaining the clean starting point. The noise **schedule** specifies how much randomness each transition adds.

Here $`x_0`$ denotes clean data, and increasing $`t`$ means **more noise**. This convention will change in the flow-matching section, where time runs from noise to data. The symbol $`\beta_t`$ here denotes a time-dependent noise variance, distinct from the VAE loss weight $`\beta`$. A DDPM forward step scales the current signal by $`\sqrt{\alpha_t}`$ and adds independent Gaussian noise with variance $`\beta_t=1-\alpha_t`$. The product $`\bar\alpha_t`$ measures how much original signal variance remains after repeated steps. If this product becomes very small, the noisy marginal approaches the standard normal under suitable data moment conditions; a finite schedule generally gives an approximation, not exact equality.

Because sums of independent Gaussian noise remain Gaussian, we can sample a noisy observation at an arbitrary training time directly, without simulating all earlier steps. The following marginal gives its mean and covariance.

For a variance-preserving DDPM, define $`\alpha_t=1-\beta_t`$ and $`\bar\alpha_t=\prod_{s=1}^{t}\alpha_s`$. Repeated Gaussian transitions yield

```math
q(x_t\mid x_0)=\mathcal{N}\!\left(
\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I\right),
```

so

```math
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
```

### Worked example: accumulating noise variance

Take $`\alpha_1=0.9`$ and $`\alpha_2=0.8`$. Substituting the first forward step into the second gives

```math
x_2=\sqrt{0.72}\,x_0+\sqrt{0.08}\,\epsilon_1+\sqrt{0.20}\,\epsilon_2.
```

The two noise variables are independent standard normals. Their variances add to 0.28, so the combined standard deviation is $`\sqrt{0.28}\approx0.5292`$. The conditional mean is $`\sqrt{0.72}\,x_0`$. Adding the standard deviations instead would introduce a cross term that independence does not justify. This is the mechanism behind the product in the general marginal.

For training, draw a clean example, a time and a noise vector; construct the noisy observation; then ask a network to predict the injected noise from the observation and time. The time input tells the network which corruption level it is trying to reverse. We know the target noise because we just generated it, so this supplies supervision without manual labels.

An individual noisy point may be compatible with many clean examples and noise draws. The best squared-error predictor therefore estimates an average noise conditional on what it can observe, rather than recovering every particular noise draw exactly.

Write $`\epsilon_\theta(x_t,t)`$ for the network's predicted noise. The squared Euclidean norm $`\|a\|_2^2=\sum_j a_j^2`$ sums squared coordinate errors. For a true noise vector `(1, −1)` and prediction `(0.5, −0.5)`, the per-example squared error is $`(1-0.5)^2+(-1+0.5)^2=0.5`$. Averaging these errors over examples, noise draws and times gives the following objective.

The simple noise-prediction loss is

```math
\mathbb{E}_{x_0,t,\epsilon}
\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2\right].
```

A **surrogate** is a tractable training quantity used in place of the quantity ultimately of interest. This simple loss reweights the noise-prediction terms that arise in a variational likelihood calculation; it is not numerically identical to the full bound. In particular, changing time-dependent weights changes the objective even if the same network predicts noise. Time sampling and loss weights change which noise levels receive emphasis. Predicting $`x_0`$, $`\epsilon`$ or a velocity parameterization requires matching the sampler's conventions.

### A score is a direction, not a quality rating

The **spatial gradient** differentiates with respect to the observation coordinates while keeping time and model parameters fixed; in one dimension it is an ordinary derivative. The notation $`\nabla_x`$ records which variable is being changed. This differs from $`\nabla_\theta`$, the parameter gradient used for learning.

The **score** is the spatial gradient of log density. It tells us how log probability density changes under a small move in the observation space. For a one-dimensional normal with mean $`\mu`$ and variance $`s^2`$, the score is $`-(x-\mu)/s^2`$: it points toward the mean, and its magnitude depends on both distance and spread. In a multimodal distribution, the direction changes across the space; it is not simply “move toward the average data point.”

Differentiating the log of the forward Gaussian marginal gives the expression below. The noise residual points away from its conditional mean; negating and rescaling it therefore gives the conditional score.

The conditional score is

```math
\nabla_{x_t}\log q(x_t\mid x_0)
=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
```

The squared-loss optimum averages this conditional target given $`x_t`$, giving the marginal noisy-data score. Thus a noise predictor implies a score estimate $`-\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}`$.

**Worked score calculation.** Consider one conditional noisy distribution with mean 1 and variance 1/4. Its log density is a constant minus $`(x-1)^2/(2(1/4))`$, so its derivative is $`-(x-1)/(1/4)`$. At $`x=1.5`$, the score is −2. Equivalently, the noise standard deviation is 1/2 and the standardized noise is $`\epsilon=(1.5-1)/(1/2)=1`$, giving $`-\epsilon/(1/2)=-2`$. The negative sign points toward the mean. This is a conditional target for one known clean source; the model averages over plausible sources when only the noisy point and time are supplied.

The conditional score knows the clean example used to create a noisy observation; the sampler does not. Averaging these conditional scores over plausible clean examples given the noisy point produces the marginal score. Squared-error denoising learns the corresponding conditional expectation, which explains why noise prediction can provide the score needed for generation.

This does **not** mean repeatedly moving uphill in density is a complete sampler: plain gradient ascent would tend to concentrate at modes. A sampling dynamics must move probability mass with the correct time dependence and, for a stochastic sampler, the appropriate noise. Starting from a terminal noise distribution, a learned reverse process gradually restores structure.

<details>
<summary>Further derivation: reverse SDE versus probability-flow ODE</summary>

### Two dynamics with the same ideal marginals

An **ordinary differential equation (ODE)** specifies a rate of change; a **stochastic differential equation (SDE)** also includes random increments. The **drift** $`f(x,t)`$ is the systematic velocity, and $`g(t)`$ scales the random part. **Brownian motion** $`W`$ has independent Gaussian increments: over a positive interval of length $`h`$, its increment has mean zero and variance $`h`$ per coordinate. Thus a small forward numerical step adds approximately $`f(x,t)h+g(t)\sqrt{h}\,\xi`$, where $`\xi`$ is standard normal. The square root matters: noise standard deviation does not scale as $`h`$. An ODE adds only the drift after the initial random sample. The reverse SDE retains stochasticity, whereas the probability-flow ODE transports the initial random sample deterministically. Under the stated assumptions, these can have the same distribution at each time without sharing individual trajectories.

For the forward SDE

```math
\mathrm{d}x=f(x,t)\,\mathrm{d}t+g(t)\,\mathrm{d}W
```

and marginal score $`s_t(x)=\nabla_x\log p_t(x)`$, the reverse-time SDE uses drift $`f-g^2s_t`$, with time integrated from $`T`$ down to $`0`$ and the associated reverse Brownian motion. The **probability-flow ODE** uses drift $`f-\tfrac12 g^2s_t`$ and shares marginals with the SDE under regularity and exact-score assumptions. Approximate scores and finite-step solvers break exact equivalence. Time direction and the factor $`\tfrac12`$ are common implementation errors.

**Why the half appears.** In one spatial dimension, the forward SDE evolves its density by the Fokker–Planck equation, which balances transport and noise spreading:

```math
\partial_t p=-\partial_x(fp)+\tfrac12 g^2\partial_x^2p.
```

Here $`\partial_t`$ means derivative in time at fixed position, $`\partial_x`$ derivative in position at fixed time, and $`\partial_x^2`$ a second position derivative. For time-only $`g`$, the identity $`p\,\partial_x\log p=\partial_xp`$ rewrites this as $`\partial_t p=-\partial_x[(f-\tfrac12g^2s)p]`$. That is the density-transport equation for an ODE with velocity $`f-\tfrac12g^2s`$. The reverse SDE still has random spreading and therefore needs the full score correction; simply deleting its noise would not give the same density evolution. This argument assumes differentiable densities, existence of the relevant dynamics, and boundary behaviour permitting conservation of probability.

**Worked comparison — a stationary Gaussian.** Consider the forward SDE with $`f(x,t)=-x/2`$, $`g=1`$, starting from a standard normal. Its distribution remains standard normal, with score $`s(x)=-x`$. Substitution gives:

| Reverse dynamics | Drift | Small step from $`x=2`$ with $`\Delta t=-0.1`$ |
|---|---|---|
| Reverse SDE | $`-x/2-(-x)=x/2`$ | Drift gives 1.9; add a zero-mean normal increment of standard deviation $`\sqrt{0.1}`$. |
| Probability-flow ODE | $`-x/2-(-x)/2=0`$ | Remains at 2; no fresh random increment. |

Both are integrated toward earlier time and, in the exact continuous dynamics, preserve the same standard-normal marginals. The ODE leaves each initial draw fixed while the SDE moves it randomly. This shows why equal distributions need not mean equal trajectories. The displayed finite SDE step is a numerical approximation, not an exact preservation claim.

</details>

### Conditioning and guidance

An unconditional model learns what observations look like overall. A conditional model also receives information such as a text prompt or class label. Classifier-free guidance uses predictions with and without that condition. Their difference identifies how the condition changes the denoising prediction; multiplying that difference strengthens or weakens the conditioning effect. It is an extrapolation rule, not a proof that the resulting samples have higher likelihood or better quality.

Classifier-free guidance combines conditional and unconditional predictions, e.g.

```math
\epsilon_{\mathrm{guided}}
=\epsilon_{\mathrm{uncond}}
+w\left(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}\right).
```

$`w=1`$ is ordinary conditional prediction in this convention. Higher guidance trades sample properties against diversity and is not automatically sampling the original conditional distribution. Latent diffusion first compresses observations with an autoencoder: an encoder produces a smaller representation and a decoder reconstructs from it. This **bottleneck** can discard information; its errors and compression cost must be counted.

**Worked exercise — finish the two-transition derivation.** Use $`x_1=\sqrt{0.9}x_0+\sqrt{0.1}\epsilon_1`$ and $`x_2=\sqrt{0.8}x_1+\sqrt{0.2}\epsilon_2`$, with independent standard-normal noises. Substitution gives signal $`\sqrt{0.72}x_0`$ and noise $`\sqrt{0.08}\epsilon_1+\sqrt{0.20}\epsilon_2`$. Its mean is zero and variance is $`0.08+0.20=0.28`$ because the cross-covariance is zero. Therefore $`q(x_2\mid x_0)=\mathcal{N}(\sqrt{0.72}x_0,0.28I)`$. For scalar $`x_0=2`$, this is mean approximately 1.6971 and standard deviation 0.5292. The experiment in the repository instead fits a flow velocity on 2D points; it measures neither image reconstruction after compression nor a DDPM reverse sampler. Its results cannot establish those systems' performance.

For a scalar prediction with unconditional value 0.4 and conditional value 0.1, guidance weight 1 gives 0.1, while weight 2 gives −0.2. The second result extends past the conditional prediction rather than averaging between the two. This arithmetic shows why changing guidance can substantially change the sampling distribution.

In latent diffusion, an encoder first maps images to a compressed representation and a decoder maps generated representations back to images. Diffusion runs in that representation space, reducing the size of the denoising problem. The decoder can only recover detail its representation supports; compression error is distinct from denoising error.

### Discuss what denoising measures

**Question:** for clean scalar value 2 and the two-transition schedule above, what is the noisy conditional mean and standard deviation? Does a smaller noise-prediction error certify better image generation?

**Worked response:** the mean is approximately 1.6971 and standard deviation 0.5292. The prediction objective measures agreement with injected noise averaged over the sampled training conditions. Sample quality also depends on the learned field, reverse sampler, guidance and representation. Conditional targets are averaged over possible clean sources at the optimum. The repository’s point-flow experiment cannot establish the performance of a DDPM image system.

## 3. Flow matching learns motion without simulating training trajectories

**Paper question:** conditional training paths can be straight even when the learned trajectories are not. What is the network actually predicting?

### Prescribe a path, then learn how to move along its distribution

Flow matching also learns to transform noise into data, but it starts by specifying a family of paths and regressing their velocities. Imagine a cloud of particles initially sampled from a simple noise distribution. A time-dependent velocity field tells each particle which way to move. The objective is to make the cloud's final distribution match data; individual particles need not be assigned a particular observed training point at sampling time.

In this section, $`x_0`$ is **noise**, $`x_1`$ is **data**, and generation runs forward from time zero to one. During training we can choose both endpoints, construct an intermediate position, and compute a target velocity analytically. This removes the need to integrate the learned dynamics just to form a training target. Sampling still requires integration.

A **coupling** specifies how a noise endpoint and a data endpoint are paired while preserving their individual distributions. Independent coupling draws each without reference to the other. Optimal-transport coupling instead seeks a pairing that minimizes an expected movement cost; that is a different construction, not something implied by drawing straight lines. **Regression** here means learning a numerical target from position and time by minimizing squared error. $`\mathrm{Uniform}[0,1]`$ gives equal probability to equal-length subintervals, so no part of the time range is favoured.

The simplest lab path is a straight line between independently sampled endpoints. It is useful because the position and its derivative are explicit.

Let $`x_0\sim\mathcal{N}(0,I)`$ and $`x_1\sim p_{\mathrm{data}}`$ independently, and $`t\sim\mathrm{Uniform}[0,1]`$. Define the path and target

```math
x_t=(1-t)x_0+tx_1,\qquad u_t=x_1-x_0.
```

### Worked example: the target is a velocity

For a one-dimensional noise endpoint −1 and data endpoint 3, the path reaches 0 at time 0.25 and its velocity is 4 at every time. This velocity has units of displacement per unit time. It is neither the destination 3 nor a noise prediction. Training inputs are the intermediate position and time; the target is the endpoint difference.

The network does not receive the endpoint pair. Different pairs can give the same intermediate position and time but require different velocities. Thus the training problem contains target variability even when the path rule is deterministic. What should one prediction represent in that situation?

Train $`v_\theta`$ by minimizing

```math
\mathbb{E}\left[\left\|v_\theta(x_t,t)-u_t\right\|_2^2\right].
```

Conditional expectation minimizes squared loss, so the population optimum is

```math
v^*(x,t)=\mathbb{E}[u_t\mid x_t=x,t].
```

For an illustrative conditional distribution with two equally likely target velocities, 2 and −2, squared error is minimized at velocity 0.

For a proposed prediction $`a`$, expand the loss: $`\tfrac12(a-2)^2+\tfrac12(a+2)^2=a^2+4`$. The smallest value is 4 at $`a=0`$. This is the calculation behind the conditional-mean claim. Predicting 0 gives mean squared error 4, whereas predicting 2 gives 8. A positive irreducible regression error is therefore compatible with an optimal predictor. More generally, expanding the squared error around the conditional mean splits it into a prediction error plus conditional variance; the latter cannot be removed using the available inputs.

The conditional mean velocity combines all the paths passing through a position at a given time. A **probability current** is density multiplied by velocity: it measures how probability moves through space. The current of a mixture is the corresponding weighted average of its component currents. Dividing that current by the mixture density gives the conditional-average velocity. It is not a claim that the network should follow every training line separately.

### Conservation of probability

The continuity equation below says that probability changes in a small region only because probability flows through its boundary. Its divergence term measures net outward flow. If a region loses more probability than it receives, its density decreases. Under regularity and suitable boundary conditions, integrating this equation preserves total probability. This is the link between a local velocity prediction and the evolution of an entire distribution.

Here $`p_t(x)`$ is the density of the whole moving cloud, after averaging over endpoint pairs. For a vector field $`J`$, its **divergence** is $`\nabla\cdot J=\sum_j\partial J_j/\partial x_j`$, the sum of spatial rates of change of its coordinate currents. In one dimension it is simply the derivative of the current. It differs from KL divergence, which compares distributions; the same word denotes two different mathematical objects.

Under suitable smoothness and integrability, this marginal velocity satisfies

```math
\partial_t p_t+\nabla\cdot(p_t v^*)=0.
```

Integrate $`\mathrm{d}x/\mathrm{d}t=v_\theta(x,t)`$ from noise at $`t=0`$ to data at $`t=1`$.

**Worked conservation calculation.** Suppose a fixed one-dimensional interval contains probability 0.20. During a short interval, integrated current carries 0.03 probability in through its left boundary and 0.01 out through its right boundary. Its new probability is $`0.20+0.03-0.01=0.22`$. The surrounding regions lose the net 0.02, so total probability stays one. The continuity equation expresses this bookkeeping locally; a learned velocity must also have the right direction and magnitude to produce the desired final distribution.

The lab's straight conditional paths use an independent endpoint coupling. They are not an implementation of optimal-transport endpoint matching, and straight conditional trajectories do not imply straight trajectories of the marginal learned ODE. A small loss on training pairs does not prove exact distribution recovery.

### Training error and integration error are different

After training, draw fresh noise and repeatedly evaluate the velocity field. A solver approximates the resulting trajectory. Euler's method uses the current velocity as if it stayed constant over one small interval. Making intervals smaller reduces this approximation error for a sufficiently smooth field, but cannot correct a systematically wrong learned velocity.

For example, a constant predicted velocity 4 starting at −1 and step size 1/4 gives positions −1, 0, 1, 2, 3 at times 0, 1/4, 1/2, 3/4, 1. If velocity changes along a step, this constant-velocity approximation is no longer exact. In the formula below, $`K`$ counts integration steps, $`h=1/K`$ is their duration, and $`k`$ indexes the successive states.

Euler updates

```math
x_{k+1}=x_k+h\,v_\theta(x_k,t_k),\qquad h=\frac1K.
```

Even with a perfect model, finite $`K`$ introduces numerical error. Under standard smoothness and stability assumptions, Euler has first-order global error: at fixed final time, halving a sufficiently small step size roughly halves the leading numerical error. This statement concerns solver error for the given field, not a guaranteed halving of a sample-quality metric. Comparing $`K=5`$ with $`K=50`$ at fixed trained weights tests numerical budget; retraining with different seeds confounds the comparison. Our CLI fixes training/evaluation randomness per seed, so rerunning with only `--sampling-steps` changed holds training fixed.

For a numerical illustration, consider the unrelated scalar ODE $`\mathrm{d}x/\mathrm{d}t=x`$ with initial value 1. Its exact solution is $`x(t)=\exp(t)`$: the derivative equals the function itself and $`\exp(0)=1`$ satisfies the initial condition. At time one this is $`e=\exp(1)\approx2.7183`$. One Euler step gives 2; two half-steps give $`1.5^2=2.25`$; ten steps give $`1.1^{10}\approx2.5937`$. The field is exact in this example, so the remaining error is entirely numerical. In a learned flow, field error and integration error both contribute to the final samples.

**Optional solver used in the research-design example.** Heun's method evaluates the velocity once to predict an endpoint and again there to correct the step:

```math
\widetilde{x}=x_k+h\,v(x_k,t_k),\qquad
x_{k+1}=x_k+\frac h2\left[v(x_k,t_k)+v(\widetilde{x},t_k+h)\right].
```

It uses two velocity evaluations per step, so Heun with 25 steps and Euler with 50 have the same count of 50 evaluations. For the scalar example $`v(x,t)=x`$, starting at 1 with $`h=1/2`$, the predictor is 1.5 and the correction gives $`1+(1/4)(1+1.5)=1.625`$. A second step gives $`1.625^2=2.640625`$, nearer $`e`$ than Euler's two-step value 2.25, but at twice the evaluation count. This calculation motivates a matched-cost research design; it does not establish that Heun always improves learned-model sample metrics.

### Discuss the error sources

**Question:** an equally likely pair of target velocities +2 and −2 is predicted as zero. Is residual MSE 4 evidence of a bad optimizer? Would more Euler steps remove it?

**Worked response:** the conditional mean zero is optimal for that condition and its minimum loss is 4. This is target variability not identifiable from the inputs. Additional solver steps change the numerical approximation to a fitted field; they neither remove this conditional variance nor correct a systematically wrong field. A controlled comparison keeps training, initial noise and evaluation data fixed.

## 4. Compare evidence before declaring a winner

### Decide which claim a measurement is meant to support

“Better generation” could mean better density estimation, more realistic individual samples, better coverage, greater task success or lower cost. These are different properties. Choose a primary claim before selecting a metric; otherwise it is easy to report whichever number improved after an experiment.

A loss measures agreement with a training objective. A held-out version asks whether that agreement transfers to new observations. A sample metric asks about outputs from the generation procedure, which includes numerical solvers, decoding or guidance. For example, a VAE's good reconstruction does not establish good prior samples, and a flow's small velocity regression error does not by itself establish coverage of all modes.

On the eight-cluster data, compare two imagined generators: one places every point close to a single real cluster; the other reaches all eight but spreads points too widely. The first has a coverage problem and the second a fidelity problem. A plot may expose both, while a scalar metric compresses them into one number and may obscure which failure occurred.

### Worked example: what a projected sample distance measures

**Wasserstein-1 distance** measures the least average displacement needed to transport one probability distribution into another, using distance as the movement cost. For two equal-size, equal-weight one-dimensional sample sets, pairing points in sorted order attains that minimum. A **projection** onto a unit direction takes each point's dot product with that direction. **Sliced** Wasserstein repeats the one-dimensional calculation on such projected data.

In one dimension, take equal-weight sample sets $`A=(0,2)`$ and $`B=(1,3)`$. Sorting and pairing gives absolute differences 1 and 1, so their empirical Wasserstein-1 distance is 1. In higher dimensions, the lab projects both point sets onto a direction and performs this one-dimensional calculation, then averages over sampled directions. Different directions reveal different discrepancies. A finite set of projections is an approximation, so record the projection settings when comparing results.

**Worked 2D projection.** Let A contain `(0, 0)` and `(2, 0)`, and B contain `(1, 1)` and `(3, 1)`. Projection onto the horizontal unit direction `(1, 0)` gives the one-dimensional sets `(0, 2)` and `(1, 3)`, with distance 1. Projection onto `(0, 1)` gives `(0, 0)` and `(1, 1)`, also distance 1. Averaging these two directions gives 1. This two-direction illustration is not the exact multivariate transport distance; the lab averages 64 fixed random unit directions.

Use held-out likelihood when it is available and its preprocessing is comparable. Report ELBO as a bound, not exact likelihood. Sample diagnostics should examine both quality and coverage. **Fréchet Inception Distance (FID)** represents real and generated images with vectors from an image network, fits their means and covariances, and compares the resulting Gaussian approximations. The network is the feature extractor. Because FID compresses distributions into those summaries, it depends on the extractor, sample count and approximation assumptions; it is not a universal perceptual or memorization test. For the 2D lab, sliced Wasserstein-1 projects equal-sized sample sets onto fixed random directions, sorts each projection, averages absolute paired differences and then averages directions. This is an empirical Monte Carlo diagnostic, not exact multivariate Wasserstein distance.

Compare generated-vs-held-out distance with an independent real-vs-real reference. That reference quantifies finite-sample variability; it is not a strict lower bound. Never compare VAE and flow training losses as if they had identical units or objectives. Plot points, check mode coverage, and inspect multiple seeds.

Even two independent finite samples from the same population usually have a nonzero empirical distance. The real-vs-real comparison reveals this sampling variability; it is not a floor that every generated-vs-real estimate must exceed. A generator can occasionally score below that reference through sampling variation or properties of the metric.

### Paired comparisons and the unit of uncertainty

An **experimental unit** is the unit assigned a condition or independently replicated for the claim at hand. A **baseline** is the reference procedure being compared; a **confound** is another changing factor that could explain a difference. A seed fixes pseudorandom choices for a repeatable run. Using distinct seeds probes run-to-run variability, while matching a seed across methods helps construct a paired comparison.

Pairing asks both methods the same questions under matched conditions. Suppose the distance differences “50-step minus 5-step” for three independent training seeds are −0.04, −0.01 and +0.02. Their mean is −0.01: a small average improvement, with one reversal. Reporting only the best seed would hide that variability. These are hypothetical values to illustrate interpretation, not repository benchmark results.

A **bootstrap** approximates the variability of a statistic by repeatedly sampling the observed independent units **with replacement** and recalculating the statistic. “With replacement” means a unit can be selected more than once. For paired methods, resample whole pairs, equivalently their differences, so the matching is preserved. An uncertainty interval summarizes the spread of that estimated statistic; it does not certify correctness of the model or of the independence assumptions.

**Worked resampling step.** For the three illustrative differences `(-0.04, -0.01, +0.02)`, one bootstrap draw of unit indices `(1, 1, 3)` gives `(-0.04, -0.04, +0.02)`, whose mean is −0.02. Another draw `(2, 3, 3)` gives mean +0.01. Repeating many draws forms a bootstrap distribution; for a percentile interval one sorts those means and takes selected lower and upper percentiles. For example, the 2.5th percentile has approximately 2.5% of the bootstrap means below it; pairing it with the 97.5th percentile retains the middle 95% of the bootstrap distribution. This interval is an approximation under the sampling assumptions, not a guarantee of 95% coverage in every setting. Two draws are not enough to estimate an interval, and three original units give very weak uncertainty evidence. Resampling many times cannot manufacture independent information absent from the original experiment.

If the claim concerns sensitivity to training randomness, the seed-level run is the unit of replication. Generating 10,000 samples from one trained model does not create 10,000 independent training runs. If the claim concerns performance on new users, independent users—not each correlated response from the same user—are the relevant units. Bootstrap intervals must resample at the level that matches the claim; pairing should be retained when the comparison is paired.

For paired evaluation examples, bootstrap the paired difference of scores. The independent unit must match the claim: resample users/documents/problems when responses within them are correlated. Seeds capture optimization variability, not population uncertainty; three seeds are an exploratory illustration, not strong inferential evidence. Record data hashes, splits, model revision, all settings, compute and unsuccessful runs. Choose hyperparameters on validation data and reserve a final test set for the final claim.

An ablation deliberately removes or changes one component to test an explanation. For a solver experiment, fixing model weights and initial noise while changing integration steps tests the contribution of numerical resolution. Retraining one configuration for longer simultaneously would confound that explanation. If one method uses two network evaluations per step, equal step counts also fail to equalize computation.

**Worked interpretation of the three-seed comparison.** Suppose the matching checks pass and the differences are those given above. A valid conclusion is: “With training and evaluation randomness paired within each seed, changing 5 to 50 Euler steps reduced the average sample-distance estimate by 0.01 across these three runs, with one seed worsening by 0.02.” It would be unsupported to say every run improved or sampling became ten times faster. More independent runs would probe stability of the effect; sampler-only timing would address speed. This is a hypothetical worked answer, not a substitute for recording the lab's actual measurements.

### Group discussion record

Record one claim the group accepts, the assumption it needs, one unresolved objection, and the evidence that would resolve that objection. A disagreement is a useful outcome if its reason is explicit. The worked answers illustrate a defensible argument; they are not a required consensus.

| Record | Worked example of the intended precision |
|---|---|
| Mathematical claim | The ELBO is a lower bound under the stated support/integrability conditions; equality requires the true posterior almost everywhere. |
| Implementation claim | Summing two latent KL contributions 0.5 and 2 gives 2.5; averaging coordinates instead changes the loss weighting. |
| Empirical claim | In an illustrative three-seed comparison, 50 rather than 5 Euler steps changes mean sample distance by −0.01, with one seed worsening. |
| Unsupported extrapolation | These toy outcomes do not establish superior image generation, faster sampling or universal superiority of a model family. |
| Next measurement | Repeat matched comparisons across independently trained models; time the sampler separately if claiming speed. |

Complete these five entries for the claim the group finds most interesting. You can use the supplied worked cases without running a computer experiment.

## Optional demonstration — run the comparisons

**How to run an optional Python example:** use the installed repository environment and stay in the repository root. Create `runs/` if needed, save the whole block as `runs/reading_group_example.py`, then run `python runs/reading_group_example.py` in the terminal. Replace the file contents for the next example. Every block supplies its own imports and inputs. Blocks marked `bash` are terminal commands. Installation is needed only for these optional computations; all fixed worked answers and discussion cases are already on this page.

The VAE and flow programs learn synthetic points: eight Gaussian clusters on a circle of radius 2, coordinate-noise standard deviation 0.15, 1,024 training points and 256 independent held-out points. `--steps` counts parameter updates, not epochs. VAE reconstruction uses fixed observation variance 0.1. The encoder’s latent variance is a separate, learned quantity.

A tensor is an array; its axes here are `(examples, coordinates)`. `.sum(-1)` adds coordinates, while the following `.mean()` averages examples. `torch` supplies tensors and automatic differentiation.

**First check a VAE calculation.** Mean `(1, 2)` and unit latent variances give KL 2.5. Duplicating coordinates gives 5; duplicating batch examples leaves mean KL 2.5. Reconstructing `(1, 2)` with mean `(0.8, 1.6)` at observation variance 0.1 gives NLL approximately 0.5353. Expected weighted losses are 3.0353 at beta 1 and 0.7853 at beta 0.1.

```python
import torch
from genai_course.objectives import diagonal_gaussian_kl, gaussian_nll

mu = torch.tensor([[1.0, 2.0]])          # one example, two latent coordinates
logvar = torch.zeros_like(mu)            # log(variance), not standard deviation
kl = diagonal_gaussian_kl(mu, logvar)
print("KL per example:", kl.tolist())
print("Duplicated coordinates:", diagonal_gaussian_kl(mu.repeat(1, 2), logvar.repeat(1, 2)).tolist())
print("Duplicated batch mean:", diagonal_gaussian_kl(mu.repeat(2, 1), logvar.repeat(2, 1)).mean().item())
x = torch.tensor([[1.0, 2.0]])
decoder_mean = torch.tensor([[0.8, 1.6]])
nll = gaussian_nll(x, decoder_mean, variance=0.1)
print("Reconstruction NLL:", nll.item())
print("Loss at beta 1:", (nll + kl).mean().item())
print("Loss at beta 0.1:", (nll + 0.1 * kl).mean().item())
```

**Then compare two fitted VAEs.** Changing beta changes the training objective; record a prediction about reconstruction and KL before running.

```bash
python -m genai_course.labs vae --steps 300 --seed 0 --beta 1
python -m genai_course.labs vae --steps 300 --seed 0 --beta 0.1
```

Each command prints a unique run-folder path. Open that folder’s `metrics.json` in a text editor. JSON stores named fields: `config` records settings, `metrics` records measured quantities, and `seconds` is a separate top-level field. Record the actual path rather than guessing a folder name. Training histories are measured before each update; final held-out diagnostics or final probabilities use the model after training. Timings include training and evaluation, so they are not sampler-only latency. Reuse earlier runs if their settings match.

| Beta | Run folder | `heldout_reconstruction_nll_mc` | `heldout_kl` | `heldout_negative_elbo_mc` | `sample_sliced_w1` |
|---|---|---|---|---|---|
| 1 | | | | | |
| 0.1 | | | | | |

All four measurements are inside `metrics`. The reported negative ELBO is **unweighted NLL + KL** for both runs. `mc` means Monte Carlo: reconstruction uses a random encoder sample. Also compare `reference_sliced_w1`, the distance between two independent real samples, which is not a strict lower bound.

**Worked interpretation with illustrative values:** reconstruction 0.6 → −0.2, KL 2.8 → 5.0 and sample distance 0.30 → 0.41 give negative ELBO 3.4 → 4.8. Lower beta improved reconstruction here but worsened the generated-sample diagnostic. Use your actual measurements; no trend is a pass/fail target.

**Check a known flow before fitting a field.** For endpoints `(-1, 1)` and `(3, -1)`, time 0.25 gives position `(0, 0.5)` and target velocity `(4, -2)`. Prediction `(3, -1)` has summed squared error 2. A constant field equal to the target takes four Euler steps to `(3, -1)` and leaves the input noise unchanged:

```python
import torch
from genai_course.objectives import flow_path
from genai_course.models import euler_sample

noise = torch.tensor([[-1.0, 1.0]])
data = torch.tensor([[3.0, -1.0]])
point, target = flow_path(noise, data, 0.25)
prediction = torch.tensor([[3.0, -1.0]])
print("Point at t=0.25:", point.tolist())
print("Target velocity:", target.tolist())
print("Summed squared error:", (prediction - target).square().sum(-1).mean().item())
# A known constant field verifies the sampling direction; it is not a trained model.
def constant_velocity(x, t):
    return target.expand_as(x)

sample = euler_sample(constant_velocity, noise, steps=4)
print("After four Euler steps:", sample.tolist())
print("Original noise preserved:", noise.tolist())
```

**Compare sampling budgets.** These commands retrain using identical training draws and then use the same sampling noise; only the Euler step count changes.

```bash
python -m genai_course.labs flow --steps 300 --sampling-steps 5 --seed 0
python -m genai_course.labs flow --steps 300 --sampling-steps 50 --seed 0
```

Save the following whole block as `runs/reading_group_example.py`, replace both paths with the printed run folders, and execute it as above. It checks the comparison rather than assuming a shared seed is enough.

```python
from pathlib import Path
import json
import numpy as np

run5 = Path("runs/REPLACE_WITH_5_STEP_RUN")
run50 = Path("runs/REPLACE_WITH_50_STEP_RUN")
a = json.loads((run5 / "metrics.json").read_text())
b = json.loads((run50 / "metrics.json").read_text())
assert a["config"]["lab"] == b["config"]["lab"] == "flow"
assert (a["config"]["sampling_steps"], b["config"]["sampling_steps"]) == (5, 50)
changes = {key: (a["config"][key], b["config"][key])
           for key in a["config"] if a["config"][key] != b["config"][key]}
with np.load(run5 / "arrays.npz") as x, np.load(run50 / "arrays.npz") as y:
    same_history = np.array_equal(x["train_objective"], y["train_objective"])

print("Changed settings:", changes)
print("Same training history:", same_history)
print("Same data:", a["data_sha256"] == b["data_sha256"])
print("Same implementation:", a["source_sha256"] == b["source_sha256"])
print("Same environment:", a["environment"] == b["environment"])
print("Same held-out velocity MSE:",
      a["metrics"]["heldout_velocity_mse_sum"] == b["metrics"]["heldout_velocity_mse_sum"])
print("Sample distance (5, 50):",
      a["metrics"]["sample_sliced_w1"], b["metrics"]["sample_sliced_w1"])
```

For the prescribed commands in the same environment, expect only `{'sampling_steps': (5, 50)}` under changed settings and `True` for all five consistency checks. Different run-folder suffixes and elapsed times are normal. If a check is false, first verify that you selected the correct two runs and used matching settings, code and environment. Do not interpret them as a controlled sampler comparison until the mismatch is understood. Record the sample distances even if 50 steps do not improve them.

| Euler steps | Run folder | `heldout_velocity_mse_sum` | `sample_sliced_w1` | `reference_sliced_w1` | Top-level `seconds` |
|---|---|---|---|---|---|
| 5 | | | | | |
| 50 | | | | | |

The held-out error sums squared velocity error over coordinates and averages examples; it is not likelihood. This single pair does not measure training-run variability. If the group chooses to extend it, repeat both commands with `--seed 1`, then both with `--seed 2`, and rerun the same matching checks separately for each pair. Three seeds still provide weak uncertainty evidence. Keep measured results distinct from the hypothetical examples in the discussion.

## Optional primary sources

The local derivations and worked examples above are sufficient for the group discussion. To inspect the papers themselves, use the following focused questions; no page sends you back here to find a missing exercise.

| Source | Focus for a paper-based discussion |
|---|---|
| [Auto-Encoding Variational Bayes — Kingma and Welling, v11](https://arxiv.org/abs/1312.6114v11) | Identify the inference approximation and why the reparameterized estimator can be differentiated. Keep inference error separate from model fit. |
| [Denoising Diffusion Probabilistic Models — Ho et al., v2](https://arxiv.org/abs/2006.11239v2) | Compare the forward marginal, reverse model and noise-prediction objective. Which weighting choices separate the practical objective from a likelihood bound? |
| [Flow Matching for Generative Modeling — Lipman et al., v2](https://arxiv.org/abs/2210.02747v2) | Distinguish conditional paths, marginal velocity and numerical sampling. The repository uses independent endpoint pairing; it does not replicate every path construction or experiment in the paper. |

The reading-group examples are explanatory calculations and small diagnostics, not reproductions of the papers’ reported large-scale results. Continue with [Lesson 03 — Preferences and research evidence](03-preferences-and-evidence.md).
