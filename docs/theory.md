# Theory notes

These notes explain how a generative model learns a distribution, how that model produces a new observation, and what an experiment can establish about it. Training, generation and evaluation are different operations. Keeping them separate makes the equations easier to interpret: a loss tells us what is optimized; a sampling procedure tells us how outputs are obtained; an evaluation tells us which claims those outputs support.

The recurring example is a cloud of points with eight clusters in a plane, used in the VAE and flow labs. A useful generator must put points near the clusters **and** allocate appropriate probability across them. Always producing the centre of one cluster can look locally plausible while representing the distribution badly. The same distinction appears in language: fluent individual responses do not establish coverage, factuality or reliability.

Use the [learning path](../lessons/README.md) for the sequence of activities. The included [foundations](foundations.md) explain probability, Gaussian variables, tensors, gradients and Python as needed; these are learning resources, not entry requirements. Each section below introduces the problem, interprets the mathematics and gives a worked example or experimental diagnosis. The existing equations and derivations remain part of the explanation; the examples are illustrative calculations, not fabricated lab results.

**Reading the notation.** An observation is one data item; a parameter is a number adjusted during learning. We write $`x`$ for an observation, $`z`$ for an unobserved variable, $`\theta`$ for model parameters and $`\phi`$ for parameters of the inference approximation introduced in §2. A subscript usually identifies a coordinate, time or parameter choice; a superscript $`*`$ denotes an ideal optimum when one is specified.

| Symbol | Meaning and a small example |
|---|---|
| $`\sum_i a_i`$, $`\prod_i a_i`$ | Add or multiply the listed terms: for 2 and 3, the results are 5 and 6. |
| $`p(x\mid y)`$ | Probability mass or density for $`x`$ conditional on knowing $`y`$. The vertical bar means “given”. |
| $`X\sim p`$, $`\mathbb{E}_p[f(X)]`$ | Draw $`X`$ from distribution $`p`$; average $`f(X)`$ with those probabilities. Values 2 and 6 with probabilities 3/4 and 1/4 have expectation 3. |
| $`\log a`$, $`\exp b`$ | Natural logarithm and its inverse: $`\log 1=0`$, $`\exp(0)=1`$. A log-loss in this base is measured in **nats**. |
| $`\int f(x)\,\mathrm{d}x`$ | Accumulate contributions continuously, like summing thin strips of area. A density 2 over an interval of width 1/2 integrates to probability 1. |

Expectations below refer to the population distribution unless an average over a finite dataset is stated. The **event dimensions** are the coordinates of one observation; a **batch** contains several observations. Thus coordinate losses `(1, 2)` and `(3, 4)` give per-observation sums 3 and 7, then batch mean 5. This convention will matter for VAE and flow losses. References use IDs in the [reading list](readings.md). Read sections 1–4 for probability objectives and representation, 5–7 for learned dynamics and evaluation, and 8–10 for adaptation and research design.

## Reading map

1. [Objectives, likelihood and autoregression](#1-what-a-generative-objective-measures)
2. [Latent variables and variational inference](#2-latent-variables-the-elbo-and-the-vae)
3. [Adversarial learning and invertible flows](#3-contrasts-gans-and-normalizing-flows)
4. [Attention and transformers](#4-transformers-and-scaling)
5. [Diffusion and score-based generation](#5-diffusion-and-scores)
6. [Flow matching and numerical sampling](#6-conditional-flow-matching)
7. [Evaluation and reproducibility](#7-evaluation-and-reproducibility)
8. [Adaptation and preference learning](#8-sft-efficient-adaptation-and-dpo)
9. [Reasoning and research evidence](#9-reasoning-and-recent-research)
10. [Retrieval, tools and experimental design](#10-retrieval-tools-and-responsible-experiments)

## 1. What a generative objective measures

### From examples to a probability model

A generative model describes possible observations and their relative plausibility. In a discrete setting, it assigns probability mass to each outcome; in a continuous setting, it assigns a density whose integral over a region gives that region's probability. Sampling draws new outcomes according to this description. Learning changes parameters so that the description agrees with the population that produced the data.

A **discrete** outcome comes from a finite or countable set, such as a word vocabulary; a **continuous** variable can range over an interval. The population is the distribution of possible observations we want to model, while a dataset is a finite sample from it. Sampling means making a random draw according to the assigned probabilities, rather than always choosing the most probable outcome.

A **joint** probability describes outcomes together. Conditional probability is $`p(A\mid B)=p(A,B)/p(B)`$ when $`p(B)>0`$. For 100 records with 30 large items and 20 items that are both large and blue, $`p(\text{blue}\mid\text{large})=20/30`$. Independence means the joint factors into the product of the individual probabilities. The probability chain rule, $`p(A,B)=p(A)p(B\mid A)`$, holds without assuming independence. **Bayes' rule** follows by expressing the joint in both orders: $`p(A\mid B)=p(B\mid A)p(A)/p(B)`$.

Write the unknown population distribution as $`p_{\mathrm{data}}`$ and the model as $`p_\theta`$. We never observe the entire population. We observe a finite dataset and usually replace population averages with sample averages. For independent observations, maximum likelihood chooses parameters that make the observed dataset probable. Taking a logarithm changes a product of probabilities into a sum; negating it turns maximization into minimization. The per-observation quantity $`-\log p_\theta(x)`$ is the negative log likelihood, or NLL. Assigning a tiny probability to an observed outcome creates a large penalty.

The **likelihood** treats the observed data as fixed and asks how their probability changes with the parameters. A **loss** is a scalar we minimize; here it is negative log likelihood. A **held-out** set is kept out of parameter fitting so that it can reveal errors on fresh observations. The [data-split explanation](foundations.md#8-training-validation-and-testing) distinguishes validation used to choose settings from a final test used to assess the chosen procedure.

### Worked example: fitting frequencies

A derivative, written $`J'(a)`$, measures the local rate at which the loss changes when $`a`$ changes. At an interior minimum it must be zero; positive second derivative means the curve bends upward. Here the rules are $`(\log a)'=1/a`$ and $`(\log(1-a))'=-1/(1-a)`$. For a longer explanation of differentiation and parameter updates, see [foundations §6](foundations.md#6-derivatives-gradients-and-learning).

Suppose ten independent observations contain eight A outcomes and two B outcomes. Let $`a=p_\theta(A)`$, so normalization requires $`p_\theta(B)=1-a`$. The average NLL is

```math
J(a)=-0.8\log a-0.2\log(1-a),\qquad
J'(a)=-\frac{0.8}{a}+\frac{0.2}{1-a}.
```

Setting the derivative to zero gives $`a=0.8`$.

The algebra is $`0.2a=0.8(1-a)`$, hence $`a=0.8`$. The second derivative is positive on the interior of the probability interval, so this is the minimum. At that solution the average NLL is about 0.5004 nats, compared with 0.6931 for equal probabilities. Assigning probability one to A would give infinite loss on the two B observations. Likelihood rewards fitting frequencies rather than just picking the most common outcome.

These are fitted **sample** frequencies. With only ten observations they need not equal population frequencies. A model that memorizes an enormous set of training sequences may likewise have excellent training likelihood and poor likelihood on new sequences. Held-out data tests this distinction.

### Entropy, cross-entropy and the direction of KL

Entropy is the average uncertainty of a distribution about its own outcomes. Cross-entropy is the average log-loss incurred when outcomes come from one distribution but probabilities are supplied by another. **Kullback–Leibler (KL) divergence** measures the extra log-loss relative to using the true distribution itself. It is nonnegative, but it is not a distance metric: exchanging its arguments usually changes its value.

In the identity below, expectation means “average over observations drawn from the population.” To derive it, add and subtract the average of $`\log p_{\mathrm{data}}(x)`$ inside the loss. The term involving only the data distribution is entropy; the remaining log ratio is KL. If the model assigns zero probability to a region with positive data probability, this KL can be infinite.

Maximum likelihood minimizes $`\mathbb{E}_{x\sim p_{\mathrm{data}}}[-\log p_\theta(x)]`$. When densities and integrals are well defined,

```math
\mathbb{E}_{x\sim p_{\mathrm{data}}}[-\log p_\theta(x)]
= H(p_{\mathrm{data}})
+ D_{\mathrm{KL}}(p_{\mathrm{data}}\Vert p_\theta).
```

**Worked decomposition.** With population probabilities `(0.8, 0.2)` and model probabilities `(0.5, 0.5)`, entropy is $`-0.8\log0.8-0.2\log0.2\approx0.5004`$. Cross-entropy is $`-0.8\log0.5-0.2\log0.5\approx0.6931`$. KL is $`0.8\log(0.8/0.5)+0.2\log(0.2/0.5)\approx0.1927`$: the additional loss, with $`0.5004+0.1927\approx0.6931`$. In a continuous setting the entropy term is called **differential entropy**; the density and units qualifications below apply.

The entropy term does not depend on $`\theta`$. Thus likelihood fits a distribution; it does not directly optimize a human quality score. For continuous observations, density values depend on units and can exceed one; negative log density can be negative. Comparing densities after rescaling coordinates requires a volume correction: stretching an interval spreads the same probability over more space. The multivariable version is the Jacobian correction, defined and calculated in §3.

A concrete density example makes the units issue visible. A uniform distribution on an interval of width 0.1 has density 10 throughout that interval, yet its total probability is one. Its NLL is $`-\log 10`$, which is negative. If the same variable is expressed in units that stretch the interval to width one, the density becomes one. Nothing about the observations became more predictable; only the reference scale changed. Compare continuous likelihoods only after aligning transformations and units.

### Autoregression: generating a joint outcome one part at a time

A **token** is a unit of a sequence, such as a word, subword or character; a tokenizer specifies that division. A **prefix** is the sequence before the position currently being predicted. A language model assigns probabilities to tokens in context. In the examples below, each displayed word is one token and the sequence has a fixed length of three.

For a sequence, the chain rule expresses its joint probability as successive conditional probabilities. This identity does not assume that only the preceding token matters. A **bigram** approximation conditions only on the immediately preceding token; the transformer model introduced in §4 can instead use a much longer prefix. The notation $`x_{\lt i}`$ means all positions before position $`i`$.

An autoregressive model uses

```math
p_\theta(x)=\prod_i p_\theta(x_i\mid x_{\lt i}).
```

Teacher forcing evaluates observed prefixes during training. At generation time prefixes are sampled. Perplexity is $`\exp(\text{mean token NLL})`$, with natural logarithms, and depends on the tokenizer and evaluation text. Perplexities from different tokenizers are not automatically comparable.

For the three-token sequence “the cat sleeps”, suppose the conditional probabilities of its observed tokens are 0.5, 0.2 and 0.1. Its joint probability is their product, 0.01. Its total NLL is about 4.6052 nats; dividing by three gives 1.5351 nats per token and perplexity about 4.64. Perplexity exponentiates average surprise: a predictor assigning probability one quarter to every observed token has perplexity four. It does not mean that every prediction literally has four equally likely alternatives.

During teacher forcing, the prediction of “sleeps” receives the observed prefix “the cat”, even if the model would itself have sampled a different second token. During generation, an earlier sampled token becomes part of the later input. An early mistake can therefore change every subsequent conditional distribution. This explains why a token-level likelihood measurement and an evaluation of complete generated sequences answer different questions.

**Connection to the labs:** before interpreting any loss, identify its observation unit, its averaging convention and the distribution used for evaluation. The [VAE comparison](../lessons/01-objectives-vae-attention/README.md#step-3--compare-two-vae-runs) applies exactly this discipline to continuous observations.

**Worked exercise — fit a categorical distribution.** A categorical variable chooses one of finitely many labels. Suppose ten observations contain 6 A, 3 B and 1 C. Fit probabilities $`p_A,p_B,p_C`$ that sum to one, and explain what the fitted values do not establish.

For general positive counts $`n_k`$ summing to $`N`$, maximize $`\sum_k n_k\log p_k`$ subject to $`\sum_kp_k=1`$. A **Lagrange multiplier** adds the constraint to the objective so it can be differentiated with the probabilities:

```math
F(p,\lambda)=\sum_k n_k\log p_k+\lambda\left(1-\sum_kp_k\right),
\qquad
\frac{\partial F}{\partial p_k}=\frac{n_k}{p_k}-\lambda=0.
```

Here $`\partial/\partial p_k`$ changes just that probability while the other variables are held fixed. Thus $`p_k=n_k/\lambda`$; summing gives $`\lambda=N`$ and $`p_k=n_k/N`$. The answer is `(0.6, 0.3, 0.1)`, with average NLL about 0.8979 nats. The log objective is concave on positive probabilities, so this stationary point is a maximum. A zero-count category can receive zero mass at a boundary optimum; smoothing would change the estimation problem.

**Interpretation:** these are training frequencies. If new observations instead have frequencies `(0.2, 0.3, 0.5)`, their average NLL under the fitted model is about 1.6146 nats. Good fit to the original counts did not establish generalization. For open-ended data, giving zero mass to every unseen outcome is an even sharper memorization failure.

## 2. Latent variables, the ELBO and the VAE

### Why introduce an unobserved variable?

A latent variable represents variation that helps explain an observation but is not directly supplied as a label. For a handwritten digit it might represent aspects of shape or style, although an unsupervised model is not guaranteed to learn human-interpretable factors. For the two-dimensional cluster dataset, latent space provides a distribution that a decoder can transform into a distribution of observations.

A **variational autoencoder (VAE)** learns both a generative distribution and a tractable approximation to its inference problem. Here tractable means that the required sampling and probability calculations can be performed efficiently; an exact definition alone does not make a quantity tractable. “Variational” means optimizing over a chosen family of approximating distributions. A **prior** describes latent values before observing $`x`$; a **posterior** describes them after observing $`x`$. A **decoder** maps latent values to an observation distribution, while an **encoder** maps an observation to an approximate latent distribution. A neural network implements these maps by composing learned linear transformations with nonlinear functions; [foundations §6](foundations.md#6-derivatives-gradients-and-learning) explains the learning operation.

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

A **Gaussian** is the normal distribution, with a mean setting its centre and a variance measuring average squared deviation. Standard deviation is the square root of variance. In several dimensions, the covariance matrix contains variances on its diagonal and cross-coordinate dependence off the diagonal. A diagonal Gaussian has independent coordinates; $`I`$ is the identity matrix with unit diagonal. The notation $`\mathrm{diag}(v)`$ puts the entries of vector $`v`$ on a diagonal. In one dimension the Gaussian density is $`(2\pi\sigma^2)^{-1/2}\exp(-(z-\mu)^2/(2\sigma^2))`$; for independent coordinates the joint density is the product of their scalar densities. These are the objects used in the next formulas.

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

A beta-VAE optimizes $`\text{reconstruction NLL}+\beta D_{\mathrm{KL}}(q\Vert p)`$. At $`\beta=1`$ this is an unbiased one-sample estimator of the negative ELBO; at other weights it is a modified objective. Posterior collapse means $`q`$ is close to the prior and the decoder can ignore z; a low reconstruction loss alone does not diagnose it. **AEVB [R1].**

**Worked check:** $`\mu=(1,2)`$, $`\ell=(0,0)`$ yields $`D_{\mathrm{KL}}=2.5`$. Duplicating latent coordinates doubles KL. Averaging two identical examples leaves the batch mean unchanged.

With unit variance, the variance terms in each coordinate cancel, leaving half the squared mean. Thus the two coordinates contribute 0.5 and 2 nats. A per-example reconstruction NLL of 1.2 would give negative ELBO 3.7; a KL weight of 0.1 would instead give training objective 1.45. The reported unweighted negative ELBO remains 3.7. These arithmetic values illustrate why comparing differently weighted training objectives can be misleading.

**Connection to the lab:** use [the VAE steps](../lessons/01-objectives-vae-attention/README.md#step-2--derive-the-vae-objective) to locate the mean, log variance, sampling operation and reductions in code. After changing the KL weight, compare reconstruction, posterior-to-prior KL and generated-point coverage separately. Better reconstruction alone does not tell you what prior samples look like.

## 3. Contrasts: GANs and normalizing flows

### Learning by discrimination

A VAE makes likelihood optimization feasible through an inference approximation. A **generative adversarial network (GAN)** takes a different route: it learns a sampler by asking whether generated examples can be distinguished from real ones. The generator $`G`$ maps random noise to an observation. The discriminator $`D`$ receives an observation and estimates whether it came from the real distribution in a balanced real/fake classification task. The generator distribution $`p_g`$ is the distribution induced by applying $`G`$ to noise; it need not have an easily evaluated density.

The discriminator is rewarded for assigning high values to real examples and low values to generated examples. The original minimax generator tries to make the discriminator fail. The two expectations below correspond to these two sources of training examples.

“Minimax” specifies opposing goals: for a fixed generator the discriminator maximizes its classification objective; the generator minimizes that objective after accounting for the discriminator. In practice one alternates a few updates of each network. The ideal nested optimization below does not claim that those finite updates find a global solution.

A GAN's idealized minimax objective is

```math
\min_G\max_D\left\{
\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)]
+\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]
\right\}.
```

For fixed $`p_g`$, the optimal discriminator is the ratio below. To see why, consider a single location with real density $`a`$ and generated density $`b`$. The discriminator maximizes $`a\log D+b\log(1-D)`$ there; setting its derivative to zero gives $`D=a/(a+b)`$. Where the two densities agree, the optimal answer is one half: the observation provides no evidence about its source.

```math
D^*(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}.
```

The **Jensen–Shannon divergence**, written $`D_{\mathrm{JS}}`$, compares each distribution with their equal mixture and averages the two KL values. A **mode** is a peak or concentrated region of a distribution; missing modes means failing to represent some such regions. A saturating discriminator is one whose output is near its limiting value, where derivatives through its output activation may be very small.

Substitution yields $`-\log 4+2D_{\mathrm{JS}}(p_{\mathrm{data}}\Vert p_g)`$, assuming both distributions have densities measured against the same underlying volume or counting convention. Here Jensen–Shannon divergence averages the KL from each distribution to their equally weighted mixture; it vanishes when the distributions agree. This distributional result assumes an optimal unrestricted discriminator, not finite alternating neural training. The commonly used non-saturating generator loss $`-\mathbb{E}_{z\sim p(z)}[\log D(G(z))]`$ changes the generator optimization objective. Instability and missing modes remain practical concerns. **GAN [R2].**

**Worked example: quality without coverage.** Suppose real data has two equally frequent, well-separated clusters, while the generator produces only one of them, with the correct distribution within that cluster. Generated points can individually look realistic. Nevertheless, at the generated cluster the optimal discriminator's probability of “real” is one third: real data places mass 0.5 there, while generated data places mass 1. At the omitted cluster the optimal discriminator gives probability one. Missing a mode is a distributional failure that a few attractive sample images may hide. A limited or poorly optimized discriminator need not detect it reliably.

When the discriminator confidently rejects generated samples, the original minimax generator can receive weak gradients through a saturated discriminator. The non-saturating loss encourages increasing the discriminator's score on generated samples more directly. Its practical usefulness does not turn alternating optimization into guaranteed convergence.

### Learning by invertible transformation

A normalizing flow transports a simple random variable through a differentiable, invertible transformation. To evaluate the density at an observed point, invert the map to find its source and account for how much volume was stretched or compressed. Probability mass is preserved; density changes when the same mass occupies a different volume.

An **inverse** recovers the unique input from an output: $`f^{-1}(f(z))=z`$. The **Jacobian** $`J_f`$ is the matrix of partial derivatives of the output coordinates with respect to the input coordinates. Its determinant describes local signed volume scaling; absolute value discards a possible orientation flip. For $`f(z_1,z_2)=(2z_1,3z_2)`$, the Jacobian is diagonal with entries 2 and 3, so its determinant is 6: a small rectangle becomes six times larger. The inverse determinant is 1/6 and its density correction is $`\log(1/6)`$.

An invertible differentiable map $`x=f_\theta(z)`$ gives

```math
\log p_\theta(x)
=\log p_0(f_\theta^{-1}(x))
+\log\left|\det J_{f_\theta^{-1}}(x)\right|.
```

Exact density evaluation trades architectural restrictions and Jacobian computation for tractability. In affine coupling, an unchanged group of coordinates determines a scale and shift for the other group; the triangular Jacobian makes log determinant a sum of scales. Flow matching, introduced in §6, instead learns a velocity field—a rule assigning a direction and speed to each position and time. Numerical integration follows that rule in finite steps and affects the resulting samples. **Real NVP [R3].**

**Worked comparison task:** for the examples below, distinguish a computed density, a lower bound and a sampler without a tractable density. Use the worked response after the table; then explain the distinction in your own words.

For a one-dimensional example, let $`z`$ be uniform on the unit interval and set $`x=2z`$. The output is uniform on an interval twice as wide, so its density is one half. The inverse derivative is one half, which supplies the required Jacobian factor. Forgetting it would assign total probability two to the output interval. In several dimensions, the determinant plays the role of this volume factor; the absolute value prevents orientation reversals from producing negative density.

An affine coupling layer leaves one group of coordinates unchanged and uses them to determine how to scale and shift another group. Because the unchanged coordinates are still available during inversion, the transformation can be inverted explicitly. Its triangular Jacobian makes the determinant cheap. Alternating the coordinate groups lets both groups change over multiple layers.

**Concrete coupling calculation.** Set $`x_1=z_1`$ and $`x_2=2z_2+z_1`$. For input `(1, 3)`, the output is `(1, 7)`; inversion gives $`z_1=x_1=1`$ and $`z_2=(x_2-x_1)/2=3`$. The Jacobian rows are `(1, 0)` and `(1, 2)`, so it is triangular and has determinant 2. The unchanged first coordinate supplies the shift during inversion. Neural coupling layers learn the scale and shift from the unchanged coordinates.

| Family | How it generates | Training quantity discussed here | Main computational issue |
|---|---|---|---|
| VAE | Prior sample followed by decoder sample | Likelihood lower bound | Approximate posterior and bound gap |
| GAN | Noise passed through a generator | Adversarial discrimination objective | Coupled optimization and coverage |
| Invertible flow | Base sample passed through invertible layers | Tractable change-of-variables likelihood | Invertibility and Jacobian cost |

**Worked response.** The uniform transformation $`x=2z`$ gives an evaluable density 1/2: this is a normalized model density, not proof that it fits real data. The two-state VAE example gave a computable bound −2.1203 on log evidence −1.6094; an inference gap limits what the bound tells us. The one-cluster GAN example can generate points without computing their normalized model density and misses half the real cluster mass. Exact density evaluation, tight inference and coverage are separate properties.

**Connection to the lab:** use the VAE's eight-cluster samples to distinguish a plausible point from adequate coverage. The comparison with GANs and invertible flows is conceptual; [the family comparison activity](../lessons/01-objectives-vae-attention/README.md#step-4--place-the-vae-among-other-model-families) does not require running archived programs.

## 4. Transformers and scaling

### What attention computes

A token initially receives an **embedding**, a learned vector looked up using its identity, together with information about its position. A vector is an ordered list of numbers; it is not yet a probability distribution. To predict in context, that representation needs information from other tokens: a pronoun may need a preceding noun, or a closing bracket may need an earlier opening bracket. Attention computes a content-dependent mixture of information from the available positions.

A **query** is a learned representation used to request information at the current position. A **key** is a representation used to decide how relevant a source position is to that query. A **value** is the information contributed by the source after its weight is chosen. These are three projections of token representations, not labels supplied by a human. The same token can be a query for its own output and a key/value for other outputs.

A dot product multiplies corresponding coordinates and sums them: `(1, 2)` dotted with `(3, 4)` is 11. A linear projection multiplies a vector by a learned matrix to obtain new coordinates; it need not be an orthogonal geometric projection. **Softmax** converts a row of real scores $`s_j`$ into normalized positive weights by exponentiating and dividing by their sum:

```math
w_j=\frac{\exp(s_j)}{\sum_k\exp(s_k)}.
```

Scores 0 and $`\log3`$ exponentiate to 1 and 3, giving weights 1/4 and 3/4. Adding the same constant to every score cancels between numerator and denominator, which is why subtracting the row maximum preserves the weights while preventing excessively large exponentials. A **mask** excludes unavailable positions; adding negative infinity gives an exponential weight of zero.

Let $`X`$ contain one row per position. Query–key dot products form a matrix with one row per requesting position and one column per source position. Row-wise softmax makes the permitted weights positive and sum to one. Multiplying these weights by values forms a weighted average for each query. The mask $`M`$ rules out unavailable sources before this normalization.

The matrices $`W_Q,W_K,W_V`$ are learned projection parameters. The symbol $`K^\top`$ transposes keys, exchanging their position and feature axes so each query can take a dot product with every key. The width $`d_k`$ is the number of coordinates in a key/query vector; $`T`$ is the number of positions. **Causal** self-attention lets a position use itself and earlier positions, excluding future ones. In a next-token predictor, the target is the token after the current input, so allowing this diagonal does not expose the target.

For $`Q=XW_Q`$, $`K=XW_K`$ and $`V=XW_V`$, attention is

```math
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
```

If independent query/key coordinates have zero mean and unit variance, their dot product has variance $`d_k`$; dividing by $`\sqrt{d_k}`$ controls this scale. Real trained vectors need not satisfy those assumptions. Compute softmax after subtracting a row maximum to avoid overflow. For causal self-attention,

```math
M_{ij}=\begin{cases}
-\infty,&j>i,\\
0,&j\leq i.
\end{cases}
```

A padding mask excludes artificial filler positions used to make differently sized sequences fit one batch. Positional information distinguishes the order of real tokens. Neither is supplied by the causal mask itself.

### Worked example: masking changes the distribution

Suppose a query at the second position has scaled scores $`(0,\log 3,10)`$ for three source positions. The third source is in the future. Without masking, its large score would dominate the softmax, leaking information unavailable during generation. A causal mask replaces that score with negative infinity before softmax. The weights become $`(1/4,3/4,0)`$. For scalar values $`(2,6,100)`$, the output is 5, independently of the future value 100.

Explicitly, $`(1/4)2+(3/4)6+0(100)=5`$. Changing the future value from 100 to 200 still gives 5. Changing the allowed first value from 2 to 6 instead gives $`(1/4)6+(3/4)6=6`$. The latter is the **positive control**: it shows that the output responds to permitted information, so a constant-output implementation would not pass both checks.

Masking only after softmax would leave weights on the first two positions summing to less than one unless they were renormalized. That would change the scale and meaning of the output. A mask that excludes every source also needs explicit handling: a softmax over all negative infinities has no valid normalized distribution.

For batch size 2, sequence length 6, query/key width 8 and value width 4, the scores and weights have shape $`(2,6,6)`$ and the output has shape $`(2,6,4)`$. The matrix product mixes **positions**; it does not turn a hidden vector into a vocabulary probability distribution. A later output projection and vocabulary softmax do that.

### From one head to a language model

A **head** is one attention computation with its own query, key and value projections. Several heads compute different mixtures, which are combined by a learned output projection. A **residual connection** adds an update to the input, $`h+F(h)`$, preserving a route for the existing representation. A **feed-forward block** applies learned linear transformations and a nonlinear activation separately at each position. An activation such as $`\max(0,a)`$ can change the form of the mapping beyond a single linear transformation. **Layer normalization** recentres and rescales a position's features, using a small positive stabilizer to avoid division by zero and learned scale/shift parameters. These operations form a block; stacking blocks repeatedly updates the representation. Positional information is needed because content-only attention has no inherent knowledge of token order. These components interact: attention alone is not a complete language model.

**Worked token-shift example.** With a start marker and the phrase “the cat sleeps”, feed input positions `[start, the, cat]` and score targets `[the, cat, sleeps]`. The representation at `cat` can attend to `start`, `the` and itself, then predicts `sleeps`. A vocabulary output head maps that representation to one score, or **logit**, for each possible next token; a final softmax produces token probabilities. In a bidirectional masked-token task, by contrast, input `the [mask] sleeps` can use both neighbours to predict `cat`. This explains the distinction before using the model names below.

The lab tests one attention head, not a complete transformer. A full decoder also needs positional information, multiple blocks, residual paths, normalization, a feed-forward network and a vocabulary output head. Training targets shift one position relative to inputs. BERT's bidirectional masked-token objective differs from a causal LM; hidden states are vectors, not class probabilities. **Attention [R4].**

Dense attention forms $`T\times T`$ interactions per head. **Key/value (KV) caching** stores earlier positions' key and value vectors. When a new token arrives, its query can reuse those vectors instead of recomputing them; stored memory grows with context length. **Sparse attention** evaluates only selected query–key interactions instead of all pairs. Reported speedups depend on kernels (the low-level routines executing tensor operations), hardware, context lengths and quality targets. **NSA [R15]** is a research case, not an automatic replacement for all settings.

An **optimizer** is the rule that updates parameters from gradients. A **compute budget** limits computational work; an empirical scaling fit summarizes measured trends when model size, data or work changes. **Deduplication** removes repeated or near-repeated examples; **contamination** occurs when evaluation material influences training or model selection. For example, splitting paragraphs from the same document across training and test can leak content even if paragraph strings are not identical.

Compute-optimal scaling results are empirical fits over particular training regimes. Parameter count, tokens, optimizer, data quality and inference costs matter; extrapolation beyond the measured regime is uncertain. Hold compute or quality targets fixed before comparing architectures. Tokenization, document-level splits, deduplication and contamination checks precede credible perplexity comparisons. **Chinchilla [R8].**

A fixed training budget creates a concrete allocation question. Spending more on parameters can leave fewer updates or fewer data tokens affordable; using a smaller model can allow more training. Scaling studies estimate tradeoffs under particular data and optimization conditions. They do not establish that increasing one dimension while ignoring the others will improve a given system.

**Connection to the lab:** in [the attention experiment](../lessons/01-objectives-vae-attention/README.md#step-5--check-whether-attention-can-see-the-future), changing a future value must leave earlier outputs unchanged. Also change an allowed past value: an implementation that always returns zero would pass the first check but fail this positive control. State precisely what the test validates—information access in this attention computation—and what still needs separate testing.

## 5. Diffusion and scores

### Learn to reverse a controlled corruption process

Destroying structure is easy: repeatedly add random noise to an observation. Generating structure is harder. Diffusion modelling turns generation into learning a sequence of local reverse transitions, using a forward corruption process that we specify ourselves. At low noise levels the model learns fine corrections; at high noise levels it must rely more on the population structure learned from data.

A **denoising diffusion probabilistic model (DDPM)** learns to reverse a prescribed sequence of noisy transitions. Here $`q`$ names the fixed forward corruption distribution; it is not the VAE encoder from §2. A **marginal** distribution averages out variables we are not keeping: $`q(x_t\mid x_0)`$ averages over the intermediate states while retaining the clean starting point. The noise **schedule** specifies how much randomness each transition adds.

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

A **surrogate** is a tractable training quantity used in place of the quantity ultimately of interest. This simple loss reweights the noise-prediction terms that arise in a variational likelihood calculation; it is not numerically identical to the full bound. In particular, changing time-dependent weights changes the objective even if the same network predicts noise. Time sampling and loss weights change which noise levels receive emphasis. Predicting $`x_0`$, $`\epsilon`$ or a velocity parameterization requires matching the sampler's conventions. **DDPM [R5].**

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

### Two dynamics with the same ideal marginals

An **ordinary differential equation (ODE)** specifies a rate of change; a **stochastic differential equation (SDE)** also includes random increments. The **drift** $`f(x,t)`$ is the systematic velocity, and $`g(t)`$ scales the random part. **Brownian motion** $`W`$ has independent Gaussian increments: over a positive interval of length $`h`$, its increment has mean zero and variance $`h`$ per coordinate. Thus a small forward numerical step adds approximately $`f(x,t)h+g(t)\sqrt{h}\,\xi`$, where $`\xi`$ is standard normal. The square root matters: noise standard deviation does not scale as $`h`$. An ODE adds only the drift after the initial random sample. The reverse SDE retains stochasticity, whereas the probability-flow ODE transports the initial random sample deterministically. Under the stated assumptions, these can have the same distribution at each time without sharing individual trajectories.

For the forward SDE

```math
\mathrm{d}x=f(x,t)\,\mathrm{d}t+g(t)\,\mathrm{d}W
```

and marginal score $`s_t(x)=\nabla_x\log p_t(x)`$, the reverse-time SDE uses drift $`f-g^2s_t`$, with time integrated from $`T`$ down to $`0`$ and the associated reverse Brownian motion. The **probability-flow ODE** uses drift $`f-\tfrac12 g^2s_t`$ and shares marginals with the SDE under regularity and exact-score assumptions. Approximate scores and finite-step solvers break exact equivalence. Time direction and the factor $`\tfrac12`$ are common implementation errors. **Score SDE [R6].**

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

### Conditioning and guidance

An unconditional model learns what observations look like overall. A conditional model also receives information such as a text prompt or class label. Classifier-free guidance uses predictions with and without that condition. Their difference identifies how the condition changes the denoising prediction; multiplying that difference strengthens or weakens the conditioning effect. It is an extrapolation rule, not a proof that the resulting samples have higher likelihood or better quality.

Classifier-free guidance combines conditional and unconditional predictions, e.g.

```math
\epsilon_{\mathrm{guided}}
=\epsilon_{\mathrm{uncond}}
+w\left(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}\right).
```

$`w=1`$ is ordinary conditional prediction in this convention. Higher guidance trades sample properties against diversity and is not automatically sampling the original conditional distribution. Latent diffusion first compresses observations with an autoencoder: an encoder produces a smaller representation and a decoder reconstructs from it. This **bottleneck** can discard information; its errors and compression cost must be counted. **Latent diffusion [R16].**

**Worked exercise — finish the two-transition derivation.** Use $`x_1=\sqrt{0.9}x_0+\sqrt{0.1}\epsilon_1`$ and $`x_2=\sqrt{0.8}x_1+\sqrt{0.2}\epsilon_2`$, with independent standard-normal noises. Substitution gives signal $`\sqrt{0.72}x_0`$ and noise $`\sqrt{0.08}\epsilon_1+\sqrt{0.20}\epsilon_2`$. Its mean is zero and variance is $`0.08+0.20=0.28`$ because the cross-covariance is zero. Therefore $`q(x_2\mid x_0)=\mathcal{N}(\sqrt{0.72}x_0,0.28I)`$. For scalar $`x_0=2`$, this is mean approximately 1.6971 and standard deviation 0.5292. The experiment in the repository instead fits a flow velocity on 2D points; it measures neither image reconstruction after compression nor a DDPM reverse sampler. Its results cannot establish those systems' performance.

For a scalar prediction with unconditional value 0.4 and conditional value 0.1, guidance weight 1 gives 0.1, while weight 2 gives −0.2. The second result extends past the conditional prediction rather than averaging between the two. This arithmetic shows why changing guidance can substantially change the sampling distribution.

In latent diffusion, an encoder first maps images to a compressed representation and a decoder maps generated representations back to images. Diffusion runs in that representation space, reducing the size of the denoising problem. The decoder can only recover detail its representation supports; compression error is distinct from denoising error.

**Connection to the activity:** work through [the diffusion derivation](../lessons/02-diffusion-flow-matching/README.md#step-1--derive-the-noisy-observation), then compare its time direction with the implemented flow lab. This repository's core lab trains a velocity field; it does not train a DDPM or validate an image-generation system.

## 6. Conditional flow matching

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

Integrate $`\mathrm{d}x/\mathrm{d}t=v_\theta(x,t)`$ from noise at $`t=0`$ to data at $`t=1`$. **Flow matching [R7].**

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

For a numerical illustration, consider the unrelated scalar ODE $`\mathrm{d}x/\mathrm{d}t=x`$ with initial value 1. Its exact value at time one is $`e\approx2.7183`$. One Euler step gives 2; two half-steps give $`1.5^2=2.25`$; ten steps give $`1.1^{10}\approx2.5937`$. The field is exact in this example, so the remaining error is entirely numerical. In a learned flow, field error and integration error both contribute to the final samples.

**Optional solver used in the research-design example.** Heun's method evaluates the velocity once to predict an endpoint and again there to correct the step:

```math
\widetilde{x}=x_k+h\,v(x_k,t_k),\qquad
x_{k+1}=x_k+\frac h2\left[v(x_k,t_k)+v(\widetilde{x},t_k+h)\right].
```

It uses two velocity evaluations per step, so Heun with 25 steps and Euler with 50 have the same count of 50 evaluations. For the scalar example $`v(x,t)=x`$, starting at 1 with $`h=1/2`$, the predictor is 1.5 and the correction gives $`1+(1/4)(1+1.5)=1.625`$. A second step gives $`1.625^2=2.640625`$, nearer $`e`$ than Euler's two-step value 2.25, but at twice the evaluation count. This calculation motivates a matched-cost research design; it does not establish that Heun always improves learned-model sample metrics.

**Connection to the lab:** compare [5 and 50 sampling steps](../lessons/02-diffusion-flow-matching/README.md#step-3--change-the-sampling-budget-hold-training-fixed) using the same training settings and seed. Equal training histories and held-out velocity error help confirm that only the solver budget changed. Inspect the generated distribution as well: a better approximation to an imperfect field need not improve every finite-sample quality metric.

## 7. Evaluation and reproducibility

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

**Worked resampling step.** For the three illustrative differences `(-0.04, -0.01, +0.02)`, one bootstrap draw of unit indices `(1, 1, 3)` gives `(-0.04, -0.04, +0.02)`, whose mean is −0.02. Another draw `(2, 3, 3)` gives mean +0.01. Repeating many draws forms a bootstrap distribution; for a percentile interval one takes selected lower and upper quantiles of those means. Two draws are not enough to estimate an interval, and three original units give very weak uncertainty evidence. Resampling many times cannot manufacture independent information absent from the original experiment.

If the claim concerns sensitivity to training randomness, the seed-level run is the unit of replication. Generating 10,000 samples from one trained model does not create 10,000 independent training runs. If the claim concerns performance on new users, independent users—not each correlated response from the same user—are the relevant units. Bootstrap intervals must resample at the level that matches the claim; pairing should be retained when the comparison is paired.

For paired evaluation examples, bootstrap the paired difference of scores. The independent unit must match the claim: resample users/documents/problems when responses within them are correlated. Seeds capture optimization variability, not population uncertainty; three seeds are a classroom illustration, not strong inferential evidence. Record data hashes, splits, model revision, all settings, compute and unsuccessful runs. Choose hyperparameters on validation data and reserve a final test set for the final claim.

An ablation deliberately removes or changes one component to test an explanation. For a solver experiment, fixing model weights and initial noise while changing integration steps tests the contribution of numerical resolution. Retraining one configuration for longer simultaneously would confound that explanation. If one method uses two network evaluations per step, equal step counts also fail to equalize computation.

**Worked interpretation of the three-seed comparison.** Suppose the matching checks pass and the differences are those given above. A valid conclusion is: “With training and evaluation randomness paired within each seed, changing 5 to 50 Euler steps reduced the average sample-distance estimate by 0.01 across these three runs, with one seed worsening by 0.02.” It would be unsupported to say every run improved or sampling became ten times faster. More independent runs would probe stability of the effect; sampler-only timing would address speed. This is a hypothetical worked answer, not a substitute for recording the lab's actual measurements.

**Connection to the lab:** use [the output guide](../labs/outputs.md) to distinguish configuration, training diagnostics and sample metrics. Write a conclusion that identifies the changed variable, what was held fixed, the observed outcome and its scope. Then specify the next measurement that could show your explanation is wrong.

## 8. SFT, efficient adaptation and DPO

### Three different choices: data, objective and parameters

Pretraining learns statistical structure from a broad dataset. Post-training changes a pretrained model toward a desired behaviour using more targeted data and objectives. Supervised fine-tuning (SFT) supplies examples of desired completions: given a prompt, increase the probability of the demonstrated answer. Preference learning instead supplies comparisons: for this prompt, one completion was judged preferable to another. A comparison gives a relative signal, not an absolute quality score or proof of truth.

**Worked SFT example.** For prompt “Name a colour” and demonstrated completion “light blue”, suppose the successive observed completion tokens have probabilities 0.4 and 0.5. Completion NLL is $`-\log0.4-\log0.5=-\log0.2\approx1.6094`$. The prompt supplies context but is not a target in this completion-only loss. A preference example would instead compare two complete responses to that same prompt, rather than declare each demonstrated token the desired target.

Parameter-efficient adaptation answers a separate question: **which model parameters may change?** **Low-rank adaptation (LoRA)** can be used with SFT or a preference objective. Matrix **rank** is the number of independent directions its columns span. Instead of learning every entry of a large update, LoRA learns two smaller factors whose product has rank at most a chosen value. It changes the allowed update to a weight matrix, not the meaning of the training labels. For a square 1,000-by-1,000 weight matrix $`W`$, choose $`B`$ with shape `(1000, 8)` and $`A`$ with shape `(8, 1000)`. Their product $`\Delta W=BA`$ has the required `(1000, 1000)` shape, using $`1000(8)+8(1000)=16{,}000`$ trainable entries instead of one million. This reduces trainable parameters for that matrix. **Frozen** means held fixed during updates; the frozen million-entry base matrix is still needed during computation.

SFT fits demonstrations with conditional token cross-entropy. Prompt tokens and padding should be excluded when the objective is completion likelihood. LoRA uses $`\Delta W=BA`$ with low rank $`r`$, keeping the base weights frozen; trainable parameter savings do not remove base-model memory or all activations. **Quantization** represents numbers with a smaller set of permitted values, usually using fewer bits. Rounding a weight 0.83 to 0.8 illustrates the introduced error, although practical quantizers use more elaborate scales and codes. **QLoRA** combines a quantized frozen base model with trainable low-rank adapters and associated memory/numerical choices. Quantization and low-rank restriction can affect quality; rank and bit width require controlled comparisons. **[R9–R10].**

### Why keep a reference policy?

Write a conditional completion distribution as a policy $`\pi(y\mid x)`$. Here $`x`$ is the prompt or context and $`y`$ is a completion, rather than the latent-variable notation used earlier. If an estimated reward is imperfect, maximizing it without constraints can push the policy toward outputs that exploit the estimate. A KL penalty to a reference policy discourages large distributional changes while allowing improvement according to the reward.

The **reference policy** $`\pi_{\mathrm{ref}}`$ is a fixed distribution used as an anchor; a **reward** $`r(x,y)`$ is a scalar score assigned to completion $`y`$ for prompt $`x`$. Reward is a modelling choice: a score of 1 for passing a test and 0 for failing says nothing about correctness outside that test. The following derivation considers a fixed reward and positive reference probabilities on the candidate responses.

The coefficient $`\beta`$ in the reward objective controls that cost of departure. For a fixed reward, increasing it makes the exact optimum more conservative relative to the reference. That statement about an ideal objective is distinct from predicting the measured KL after a finite number of gradient updates in the DPO lab.

For each context $`x`$, maximize

```math
\mathbb{E}_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta D_{\mathrm{KL}}\!\left(
\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\right),
\qquad\beta>0.
```

**Work through the constrained optimization.** Hold a prompt fixed and abbreviate candidate probabilities by $`\pi_i`$, reference probabilities by $`q_i`$ and rewards by $`r_i`$. As in the categorical example in §1, add a multiplier $`\lambda`$ enforcing that probabilities sum to one:

```math
\begin{aligned}
F(\pi,\lambda)
&=\sum_i\pi_i r_i-\beta\sum_i\pi_i\log(\pi_i/q_i)
  +\lambda\left(\sum_i\pi_i-1\right),\\
0=\frac{\partial F}{\partial\pi_i}
&=r_i-\beta\left(\log(\pi_i/q_i)+1\right)+\lambda,\\
\pi_i&=q_i\exp(r_i/\beta)\exp(\lambda/\beta-1).
\end{aligned}
```

The last factor is shared by all responses. Requiring their sum to be one makes it $`1/Z`$, where the **partition function** is simply the sum of the unnormalized weights: $`Z=\sum_iq_i\exp(r_i/\beta)`$. The reward term is linear and negative KL is concave, so this stationary point is the optimum in a finite candidate space with positive reference probabilities. For an infinite response space the normalizing sum/integral must also be finite.

The result is an exponentially tilted reference: a completion receives more mass if its reward is larger, but only within reference support. For two equally probable reference completions with reward difference $`\log 3`$ and $`\beta=1`$, the optimal probabilities become 0.75 and 0.25.

Explicitly, choose rewards `(log 3, 0)` and reference `(0.5, 0.5)`. The unnormalized weights are `(1.5, 0.5)`, their sum $`Z`$ is 2, and division by 2 gives `(0.75, 0.25)`. As the penalty coefficient grows with rewards held fixed, this tilt weakens.

With normalization enforced by a multiplier, stationary optimization gives

```math
\pi^*(y\mid x)
=\frac{\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)}.
```

Hence

```math
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
```

### From rewards to pairwise classification

The Bradley–Terry model turns a reward difference into a preference probability through the sigmoid $`\sigma(a)=1/(1+\exp(-a))`$. Equal rewards predict a one-half preference probability; a reward difference of $`\log 3`$ predicts three quarters. This is a statistical model of labels, not an assumption that annotators always agree.

Substitute the reward expression above for each completion in the pair. Both completions share the same prompt, so their $`\beta\log Z(x)`$ terms cancel. This cancellation lets the objective be expressed using policy and reference log probabilities without estimating that normalizer or fitting a separate reward network.

In the formulas, $`y_w`$ is the preferred (“winner”) completion, $`y_l`$ the rejected (“loser”) completion, and $`y_w\succ y_l`$ means that the first is preferred to the second. The sigmoid maps any real number to a probability strictly between zero and one. **Direct preference optimization (DPO)** trains the policy using the negative log probability of the observed pair preference. Its adjusted **margin** is the policy's winner-minus-loser log-probability difference minus that same difference under the reference; a positive value means increased relative odds for the observed winner.

**Worked cancellation with the same two responses.** Their policy/reference log ratios are $`\log(0.75/0.5)=\log1.5`$ and $`\log(0.25/0.5)=\log0.5`$. Adding the common $`\log Z=\log2`$ reconstructs rewards $`\log3`$ and 0. Subtracting the rewards cancels the two $`\log2`$ terms and gives $`\log1.5-\log0.5=\log3`$. The implied preference probability is $`\sigma(\log3)=3/4`$, so the observed winner's pair loss is $`-\log(3/4)\approx0.2877`$. Different prompts could have different normalizers; their terms would not cancel this way.

Under a Bradley–Terry preference model

```math
P(y_w\succ y_l\mid x)
=\sigma\!\left(r(x,y_w)-r(x,y_l)\right),
```

the context-dependent $`\log Z(x)`$ cancels. Substituting the parameterized policy yields

```math
\begin{aligned}
\mathcal{L}_{\mathrm{DPO}}
&=-\mathbb{E}_{(x,y_w,y_l)}\log\sigma\!\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right).
\end{aligned}
```

The derivation assumes reference support for candidate responses and a preference model; inconsistent labels and distribution shift can violate the modelling assumptions. Completion log probabilities are sums of token log probabilities. Length normalization changes the objective. Freeze the reference model; changing it while fitting changes the target. DPO learns from preference pairs without explicitly fitting a separate reward model. This does not establish factuality, fairness, or universal alignment. **InstructGPT [R11], DPO [R12].**

The lab fits a single-context, three-action categorical policy. This makes the objective and gradients inspectable, but it is not language-model fine-tuning and has no held-out contexts. At initialization $`\pi_\theta=\pi_{\mathrm{ref}}`$, the margin is $`0`$ and the loss is $`\log 2`$. A negative gradient on the chosen log probability increases the margin under gradient descent. Distinguish the KL coefficient in the reward objective from the observed KL after optimization.

### Worked example: interpreting the DPO margin

Define $`m`$ as the policy's chosen-minus-rejected log-probability difference **minus the same difference under the reference**. The pair loss is $`-\log\sigma(\beta m)`$. A positive margin means the policy has increased the chosen completion's relative odds compared with the reference. It does not require the chosen completion to have the largest absolute probability among all outputs.

To see the update direction, rewrite the pair loss as $`L(m)=\log(1+\exp(-\beta m))`$. Its derivative is $`L'(m)=-\beta/(1+\exp(\beta m))`$, which is negative for finite $`m`$ and positive $`\beta`$. Decreasing this loss therefore favours a larger adjusted margin.

At $`m=0`$, the loss is $`\log 2\approx0.6931`$, and its derivative with respect to the margin is $`-\beta/2`$. Gradient descent therefore pushes the margin upward. For a uniform three-outcome reference and a policy with probabilities $`(0.5,0.25,0.25)`$, preferring outcome zero over outcome one gives margin $`\log 2`$. At $`\beta=1`$, the implied pair preference probability is two thirds and the loss is $`\log(3/2)\approx0.4055`$.

This shows what has improved: the model fits the given relative preference more strongly.

**Worked one-update check, including the unpaired outcome.** Start with three logits `(0, 0, 0)`, hence probabilities `(1/3, 1/3, 1/3)`, and a uniform reference. Prefer outcome 0 to outcome 1 with $`\beta=1`$. In their log-probability difference, the shared softmax normalizer cancels, leaving margin equal to logit 0 minus logit 1. At initialization the loss gradients for the three logits are `(-0.5, +0.5, 0)`. A gradient-descent step of size 0.5 gives logits `(0.25, −0.25, 0)` and probabilities approximately `(0.4192, 0.2543, 0.3265)`. The unpaired logit did not change, but its probability did because the normalizing sum changed. The margin is now 0.5 and the pair loss approximately 0.4741, down from 0.6931. This is one specified update, not the output of the 100-step lab.

It does not demonstrate correctness, broader usefulness or generalization. Because probabilities normalize together, changes to the paired logits also affect the unpaired outcome. Repeating one preference cannot tell us how the policy should behave on new prompts or conflicting labels.

**Connection to the lab:** inspect [the three-outcome preference experiment](../lessons/03-post-training-research/README.md#step-2--inspect-a-preference-experiment-small-enough-to-understand). Record all three probabilities and KL to the reference, alongside the loss. Explain why the finite-step comparison changes both optimization behaviour and the observed fit, rather than treating the coefficient as a direct target for final KL.

## 9. Reasoning and recent research

### What would count as improved reasoning?

A correct final answer is one observable outcome. Robustness asks whether the result survives changes that preserve the problem, such as renaming entities or reordering irrelevant information. Process validity asks whether the stated intermediate steps support the answer. These measurements can disagree: a system may guess correctly, present an invalid derivation, or fail after a harmless wording change. A useful research question names which property is claimed and how it is measured.

A **verifier** is a procedure that checks a proposed answer against a specified criterion. Outcome-based training can use a verifier to score completed attempts. A code task might run tests; a mathematical task might compare a canonical final answer. This reduces reliance on manual preference labels for those particular outcomes, but transfers responsibility to the verifier. Passing incomplete tests is evidence about those tests, not all intended program behaviour. Intermediate reasoning can remain wrong even when the final answer passes.

An **advantage** is a reward expressed relative to a baseline, indicating which sampled actions did better or worse than that reference. A policy probability ratio compares a completion's probability under the updated and previous policy; **clipping** limits how far such a ratio contributes to an update. A regularization penalty, such as KL to a reference, discourages excessive change. These terms describe ingredients of some policy-training rules, not a new loss assigned in the core lab.

Group-relative updates compare several sampled attempts for the same problem to obtain a relative learning signal. For rewards 0, 0 and 1, subtracting the group mean makes the successful attempt positive relative to the group and the others negative. This is intuition for a centred reward signal, not a complete implementation: normalization, policy ratios, clipping and regularization depend on the algorithm. A group with identical rewards supplies no within-group preference through simple centring.

For the rewards `(0, 0, 1)`, the mean is 1/3 and the centred values are `(−1/3, −1/3, +2/3)`, whose sum is zero. For `(1, 1, 1)`, centring gives three zeros. This explains both the comparison signal and why a uniformly successful or uniformly unsuccessful group contains no relative ranking under this rule.

Verifiable-reward training uses checkable outcomes, such as executable tests or a mathematical answer verifier. The verifier can be incomplete, exploitable or misaligned with the intended task. Group-relative methods estimate advantages relative to sampled alternatives and require careful variance, reward and KL handling. The DeepSeek-R1 report is a case study in reasoning post-training, not evidence that one small experiment reproduces a frontier training pipeline. **[R13].**

**Inference** here means using an already trained model to generate or score an answer. **Pass@k** asks whether at least one of $`k`$ attempts is correct; it differs from the accuracy of one answer selected for a user. A **verifier** is the procedure used to recognize correctness. A **chain of thought** is a sequence of intermediate reasoning statements generated as text; its appearance alone does not establish that it describes the internal cause of the answer.

Inference-time computation includes sampling multiple candidates, verification and search. Compare both success and total cost: tokens, latency, tool calls and verifier cost. A $`\mathrm{pass}@k`$ improvement can simply purchase more attempts. Distinguish final-answer correctness, robustness under meaning-preserving perturbations and process validity. Written chains of thought need not faithfully expose a model's internal causal reasoning.

### Worked example: more attempts versus a better attempt

Suppose a fixed model succeeds with probability 0.2 on each independent attempt for one task. The probability of at least one success in five attempts is $`1-(1-0.2)^5\approx0.6723`$. The model's per-attempt probability has not improved. The increased success rate was purchased with more attempts. Real attempts may be correlated, and task difficulty varies, so this calculation is an illustration rather than a formula for every aggregate benchmark.

Moreover, “one of the attempts was correct” assumes a way to recognize that success. A deployable system must select an output using an available checker or selection rule. An oracle pass@k measurement can be much better than the accuracy of the answer a user actually receives. Report attempts, token usage, selection procedure and verifier cost alongside success.

### Reading a research claim critically

**Worked control for a reasoning claim.** Imagine method A answers 60 of 100 problems correctly with one attempt per problem, while B reaches 75 with five attempts and a checker. The 15-point gain does not isolate a better per-attempt model. A concrete control gives A the same five-attempt budget and checker on the same problems. If it then reaches 74, the remaining observed difference is one problem; if it stays at 60, the budget-only explanation is less plausible. Either outcome still needs paired problem-level uncertainty and scrutiny of the checker. These are hypothetical numbers for reasoning about controls, not results from R18 or R19.

A paper's headline result is a starting point for a causal question: what changed between the compared systems? Different data, larger inference budgets, selection by a stronger verifier and contamination can each explain an apparent gain. Identify the experimental unit and matched baseline before interpreting a percentage. A benchmark's scoring rule is itself a measurement instrument whose validity needs evidence.

Use **RRB [R18]** and **ReasoningMath-Plus [R19]** as 2026 preprint critique cases. A **perturbation** is a deliberate input change; preserving semantics means preserving the mathematical task and its valid answer. A **refusal** is a response declining to attempt the task, which should be distinguished from an attempted wrong answer. Ask whether perturbations preserve semantics, whether refusals are separately categorized, how problems were selected, whether samples are independent, and whether the score rewards a particular expression of reasoning. Read the version stated in the bibliography; later revisions may change conclusions. An abstract-level summary cannot establish independent replication.

Diffusion language models such as **LLaDA [R14]** provide a non-autoregressive research comparison. Masking schedules, conditioning, number of forward passes, token budgets and baseline training data all affect fair evaluation. Do not infer universal superiority from an isolated benchmark.

Autoregressive language generation commits to a sequence of left-to-right conditional predictions. A masked diffusion language model instead learns to recover missing token content under a corruption process and uses an iterative generation procedure. This can allow different patterns of parallel prediction and revision, but introduces choices about which positions are updated, how many passes are used and what conditioning is supplied. Comparing the approaches requires aligning training data and accounting for actual inference work; one iteration need not cost the same amount in each system.

**Connection to the activity:** in [the research critique](../lessons/03-post-training-research/README.md#step-4--critique-one-paper-with-a-concrete-question), select one claim and propose a control that could distinguish an algorithmic improvement from a budget, data or measurement explanation. You are evaluating evidence, not reproducing a frontier model with the toy labs.

## 10. Retrieval, tools and responsible experiments

### Retrieval supplies evidence at answer time

A model's parameters store learned statistical regularities; they are not an editable database of guaranteed facts. Retrieval augments a request with external material selected at answer time. A typical pipeline turns a query into a search, ranks candidate passages, supplies selected passages to the generator and asks for an answer grounded in them. Updating the source collection can update available evidence without retraining all model weights, but only if the relevant material is actually retrieved and used correctly.

There are at least two distinct failures: the retriever does not supply the needed evidence, or the generator misuses evidence that is present. A fluent wrong answer alone does not identify which component failed.

A **retrieval-augmented generation (RAG)** system combines this search with answer generation. **Retrieval recall** measures how much of the required relevant evidence is retrieved; if each question has one sufficient passage, it is the fraction of questions for which that passage is found. **Citation support** asks whether a cited passage actually justifies a generated claim. An **oracle-evidence baseline** manually supplies sufficient evidence to test the generator with retrieval failure removed; a **no-retrieval baseline** supplies no external evidence. “Oracle” describes an evaluation intervention, not a deployable component.

A retrieval-augmented system retrieves evidence and conditions generation on it. Evaluate retrieval recall separately from answer correctness and citation support, then use oracle-evidence and no-retrieval baselines to localize failure. Split by source or time when near-duplicate chunks could leak answers. Retrieval may return irrelevant, outdated or adversarial text; a citation's existence is not proof that it supports the generated claim. **RAG [R17].**

### Worked diagnosis: isolate retrieval from evidence use

Imagine a public handbook contains the correct deadline. Under no retrieval, a system answers 40 of 100 questions correctly; with ordinary retrieval, 55; with a manually supplied sufficient passage, 85. These hypothetical outcomes suggest room for improving retrieval or evidence selection, while the remaining 15 errors show that supplying evidence does not guarantee correct use. The 30-point gap is a diagnostic under this intervention, not a universal decomposition of causal effects.

**Worked measurement for the lesson task.** In a separate hypothetical inspection of 100 questions, suppose ordinary retrieval supplies the one sufficient passage for 70 questions and the generator answers 56 of those correctly. Evidence recall is $`70/100=70\%`$. Failure to answer correctly despite sufficient evidence is $`(70-56)/70=20\%`$ within that subset. The other 30 questions have missing evidence, but their answer accuracy is not given. These denominators distinguish the two failure types; the earlier aggregate scores alone cannot recover them.

To make the comparison interpretable, use the same questions, generator and decoding budget, and control passage length and formatting as far as practical. Check whether the ordinary retriever actually included the necessary passage. Add an irrelevant but plausible distractor to otherwise sufficient evidence: a performance drop tests susceptibility to distraction rather than missing knowledge. Check citation support directly; a real document can be cited for a claim it never makes.

### Tools change what a model can do

A tool-using system can request an operation, receive an observation and choose a later action from the updated context. For example, a calculator can resolve arithmetic that a text-only response gets wrong. The complete system includes the model, tool specification, execution environment, observation handling and stopping rule. An action's syntactic validity does not establish that it was useful or that the overall task was completed.

Tool-using agents add actions and observations to generation. Evaluate task completion, intervention rate, latency, cost and harmful unintended actions. Retrieved text and tool outputs are data, not authority to override instructions. In research exercises use a local simulated tool environment without credentials or external write access. **ReAct [R20].**

Consider a simulated agent asked to locate a record and update one field.

For a concrete case, record `17` initially contains `{"status": "pending", "owner": "Ana"}`. The request is to set record 17's status to `approved`. A successful final state is `{"status": "approved", "owner": "Ana"}` for record 17, with all other records unchanged, followed by an accurate report of that change. Finding record 17 but leaving `pending`, modifying record 18, or overwriting `owner` fails the task. This is a worked evaluation example; it does not ask you to modify an external system.

Successful search is an intermediate result. Full success requires finding the intended record, making the correct change, preserving unrelated fields and reporting the result accurately. An evaluation that counts only successful tool calls can reward a system that never achieves the requested final state. Evaluate both intermediate failures and the end condition, including unnecessary actions and cost.

Retrieved passages and tool responses may contain text that looks like instructions. Such text is evidence from an external source, not authority to change the task. This distinction is also experimentally testable: insert a distractor instruction into a simulated retrieved document and measure whether the system follows the original request.

**Provenance** records where data came from and how they were collected or transformed. **Memorization risk** concerns reproducing training content rather than learning transferable structure. An **affected group** is a population whose outcomes could change under the system's use. **Release constraints** identify which data, code or model artefacts may be shared. For example, a retrieval study using a private handbook may report aggregate accuracy without permission to release the handbook itself; synthetic toy data avoid that particular restriction but do not establish safety on private documents.

A **hypothesis** is a testable statement about what will happen under specified conditions. It is **falsifiable** when some possible outcome would count against it. For example, “Heun reduces mean sample distance relative to Euler at 50 evaluations on these models” specifies a metric, comparison and budget; observing no reduction counts against the directional claim. The [worked protocol](../lessons/checkpoints.md#9-a-research-protocol-that-can-fail) turns this statement into a complete design with a baseline, data split, uncertainty calculation and possible objection.

Every research proposal should describe dataset provenance and permissions, privacy and memorization risks, affected groups, evaluation limits, and release constraints. Avoid uploading personal or restricted data to model services. These are research-design considerations, not claims of legal compliance. Core labs deliberately use synthetic data so licensing and external-service access do not obstruct the class.

These design choices affect scientific validity as well as responsible use. If near-duplicate documents occur in training and test sets, the apparent generalization result may be retrieval or memorization. If only convenient user groups are represented, the measured accuracy may not support the intended deployment claim. Record the population and conditions to which the conclusion applies.

**Connection to the activity:** use [the research-protocol task](../lessons/03-post-training-research/README.md#step-5--turn-an-observation-into-a-research-protocol) to specify a hypothesis, the manipulated component, a matched baseline, the independent unit, an outcome and a result that would contradict the hypothesis. The [worked protocol](../lessons/checkpoints.md#9-a-research-protocol-that-can-fail) illustrates the required precision. A well-designed negative result still teaches us which explanation the evidence does not support.
