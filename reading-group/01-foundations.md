# 01 — Shared foundations for generative AI

[Reading-group home](README.md) · Next: [02 — How generative models work](02-generative-models.md)

**No advance reading, installation or prior machine-learning course is assumed for this meeting.** Read the explanations together and work through the examples on this page. All questions have a worked answer here. A notebook or paper for notes is enough; nobody needs to run code.

**Central question:** what does it mean to learn a distribution, and how is that different from producing a plausible example?

The route is **probability → learning from data → representations and sequences → interpreting evidence**. Keep this page open throughout. The group can take turns explaining a calculation, questioning an assumption and recording the conclusion; no roles need to be assigned beforehand.

## 1. What a generative model describes

An **observation** is one item of data: a point, image or sequence of tokens. A **variable** represents a quantity that can take different values. A model uses **parameters**, usually written $`\theta`$, to determine its predictions. Learning adjusts those parameters using observations.

A discriminative model predicts a target from an input, such as a label given an image. A generative model describes a distribution of observations, possibly conditional on an input, and can produce new observations by sampling. Our running example is a cloud of points with two coordinates, drawn from eight clusters. The generative task is to learn the distribution of points, not to classify their cluster labels.

### Notation as it appears

We write $`x`$ for an observation and $`\theta`$ for adjustable model parameters. A subscript usually names a coordinate or a parameter choice. $`p(x\mid y)`$ means probability or density for $`x`$ **given** $`y`$. $`X\sim p`$ means draw a random value using distribution $`p`$.

$`\sum`$ adds terms, while $`\prod`$ multiplies them: for values 2 and 3 their results are 5 and 6. The natural logarithm $`\log`$ and exponential $`\exp`$ are inverses; $`\log1=0`$ and $`\exp(0)=1`$. An integral accumulates continuously, like summing density times the widths of many thin intervals. These symbols describe operations, not extra assumptions about the data.

A generative model describes possible observations and their relative plausibility. In a discrete setting, it assigns probability mass to each outcome; in a continuous setting, it assigns a density whose integral over a region gives that region's probability. Sampling draws new outcomes according to this description. Learning changes parameters so that the description agrees with the population that produced the data.

A **discrete** outcome comes from a finite or countable set, such as a word vocabulary; a **continuous** variable can range over an interval. The population is the distribution of possible observations we want to model, while a dataset is a finite sample from it. Sampling means making a random draw according to the assigned probabilities, rather than always choosing the most probable outcome.

A **joint** probability describes outcomes together. Conditional probability is $`p(A\mid B)=p(A,B)/p(B)`$ when $`p(B)>0`$. For 100 records with 30 large items and 20 items that are both large and blue, $`p(\text{blue}\mid\text{large})=20/30`$. Independence means the joint factors into the product of the individual probabilities. The probability chain rule, $`p(A,B)=p(A)p(B\mid A)`$, holds without assuming independence. **Bayes' rule** follows by expressing the joint in both orders: $`p(A\mid B)=p(B\mid A)p(A)/p(B)`$.

Write the unknown population distribution as $`p_{\mathrm{data}}`$ and the model as $`p_\theta`$. We never observe the entire population. We observe a finite dataset and usually replace population averages with sample averages. For independent observations, maximum likelihood chooses parameters that make the observed dataset probable. Taking a logarithm changes a product of probabilities into a sum; negating it turns maximization into minimization. The per-observation quantity $`-\log p_\theta(x)`$ is the negative log likelihood, or NLL. Assigning a tiny probability to an observed outcome creates a large penalty.

The **likelihood** treats the observed data as fixed and asks how their probability changes with the parameters. A **loss** is a scalar we minimize; here it is negative log likelihood. A **held-out** set is kept out of parameter fitting so that it can reveal errors on fresh observations. Training data fit parameters; validation data choose settings; a final test assesses the chosen procedure after those choices are fixed.

**Try it together:** among 100 records, 40 are blue, 30 are large and 20 are both. What is the probability of blue given large, and is it the same as large given blue?

**Worked answer:** restrict the denominator to the known group. The first is `20/30 = 2/3`; the second is `20/40 = 1/2`. Conditioning changes the question. Knowing either answer does not show that size causes colour.

### Averages and variability

An **expectation** is a probability-weighted average. If a variable takes value 2 with probability 0.75 and value 6 with probability 0.25, its expectation is $`0.75(2)+0.25(6)=3`$. The notation $`\mathbb{E}[f(X)]`$ means to average the quantity $`f(X)`$, rather than necessarily averaging $`X`$ itself. A dataset mean approximates a population expectation using observed examples.

**Variance** measures average squared deviation from the mean. In this example it is $`0.75(2-3)^2+0.25(6-3)^2=3`$. Standard deviation is the square root of variance. Independent noise variances add; their standard deviations generally do not.

### A continuous example: the Gaussian

For a continuous variable, a **density** gives probability per unit of the variable. Probability over a region is the integral of density over that region. An integral is the continuous analogue of adding contributions: approximate it by summing density times the width of many small intervals. A density value itself is not a probability and can exceed one. A uniform variable on an interval of width 0.1 has density 10 but total probability one.

A Gaussian or normal distribution is a bell-shaped density, written $`\mathcal{N}(\mu,\sigma^2)`$ in one dimension. Its mean $`\mu`$ sets the centre and its standard deviation $`\sigma`$ sets the spread. A standard normal has mean zero and variance one. If $`\epsilon\sim\mathcal{N}(0,1)`$, then $`z=\mu+\sigma\epsilon`$ has the desired mean and variance. With mean 1, standard deviation 2 and noise draw 0.5, the sample is 2.

For several coordinates, a **covariance matrix** records their variances on the diagonal and cross-coordinate dependence off the diagonal. An identity covariance $`I`$ gives each coordinate unit variance and zero covariance with the others. For a multivariate Gaussian, diagonal covariance also implies independent coordinates. The notation $`\mathrm{diag}(\sigma^2)`$ places the coordinate variances on the diagonal and zeros elsewhere.

**Try it together:** a normal variable has mean 1 and standard deviation 2. A standard-normal draw is 0.5. What sample does shifting and scaling produce?

**Worked answer:** `1 + 2 × 0.5 = 2`. Holding that draw fixed, increasing the mean to 1.1 gives 2.1; increasing the standard deviation to 2.1 instead gives 2.05. The random draw and the distribution parameters have different roles.

## 2. How data changes a model

### Why unlikely observations cost more

A negative log likelihood turns an assigned probability into a penalty. An observed outcome assigned probability 0.5 costs $`-\log0.5\approx0.6931`$; at probability 0.1 it costs $`-\log0.1\approx2.3026`$. Smaller assigned probability gives larger loss. Natural-log units are called **nats**. Because $`\log(ab)=\log a+\log b`$, multiplying probabilities of independent observations becomes adding their log probabilities.

### Worked example: fitting frequencies

A derivative, written $`J'(a)`$, measures the local rate at which the loss changes when $`a`$ changes. At an interior minimum it must be zero; positive second derivative means the curve bends upward. Here the rules are $`(\log a)'=1/a`$ and $`(\log(1-a))'=-1/(1-a)`$.

Suppose ten independent observations contain eight A outcomes and two B outcomes. Let $`a=p_\theta(A)`$, so normalization requires $`p_\theta(B)=1-a`$. The average NLL is

```math
J(a)=-0.8\log a-0.2\log(1-a),\qquad
J'(a)=-\frac{0.8}{a}+\frac{0.2}{1-a}.
```

Setting the derivative to zero gives $`a=0.8`$.

The algebra is $`0.2a=0.8(1-a)`$, hence $`a=0.8`$. The second derivative is positive on the interior of the probability interval, so this is the minimum. At that solution the average NLL is about 0.5004 nats, compared with 0.6931 for equal probabilities. Assigning probability one to A would give infinite loss on the two B observations. Likelihood rewards fitting frequencies rather than just picking the most common outcome.

These are fitted **sample** frequencies. With only ten observations they need not equal population frequencies. A model that memorizes an enormous set of training sequences may likewise have excellent training likelihood and poor likelihood on new sequences. Held-out data tests this distinction.

### What a parameter update does

A **derivative** measures local rate of change. For $`f(w)=w^2`$, its derivative is $`2w`$: near $`w=3`$, a small increase in $`w`$ changes the output by approximately six times that increase. A **partial derivative** changes one input while holding the others fixed. A **gradient** collects all partial derivatives of a scalar function into a vector. It points in the direction of greatest local increase under the usual Euclidean geometry.

Gradient descent adjusts parameters in the opposite direction to reduce a loss:

```math
\theta_{\mathrm{new}}=\theta-\eta\nabla_\theta J(\theta).
```

Here $`J`$ is the loss and $`\eta`$ is the positive step size, or learning rate. For $`J(w)=(w-3)^2`$, starting at $`w=0`$ gives derivative −6. With learning rate 0.1 the next value is 0.6, and the loss falls from 9 to 5.76. An overly large step can overshoot; a gradient step is not a guarantee of finding a global minimum.

The **chain rule** differentiates a composition by multiplying local derivatives. If $`u=2w`$ and $`J=u^2`$, then $`\mathrm{d}J/\mathrm{d}w=(2u)(2)=8w`$. Backpropagation applies this rule through a computation graph. Automatic differentiation records those operations and computes their derivatives; it does not estimate them by trying random parameter changes.

A neural network composes simple parameterized operations. In one scalar operation, `w × x + b` multiplies input `x` by a learned **weight** `w` and adds a learned **bias** `b`. An **activation** applies a nonlinear function, such as replacing a negative result with zero. For `x = 2`, `w = 0.5` and `b = −0.2`, the result is 0.8 and this activation leaves it at 0.8; for `x = 0`, the result −0.2 becomes zero. Composing many such operations allows a model to represent relationships beyond a single straight line. The weights and biases are parameters adjusted by training. Training computes predictions, evaluates a loss, differentiates it and updates parameters. Minibatch training estimates gradients using a subset of observations at a time. An **epoch** traverses a dataset; the lab's `--steps` flag instead counts parameter updates.

### Entropy, cross-entropy and the direction of KL

Entropy is the average uncertainty of a distribution about its own outcomes. Cross-entropy is the average log-loss incurred when outcomes come from one distribution but probabilities are supplied by another. **Kullback–Leibler (KL) divergence** measures the extra log-loss relative to using the true distribution itself. It is nonnegative, but it is not a distance metric: exchanging its arguments usually changes its value.

For a finite set of outcomes indexed by $`k`$, let $`p_k`$ be the population probabilities and $`q_k`$ the model probabilities. The definitions are

```math
\begin{aligned}
H(p)&=-\sum_k p_k\log p_k,\\
C(p,q)&=-\sum_k p_k\log q_k,\\
D_{\mathrm{KL}}(p\Vert q)&=\sum_k p_k\log(p_k/q_k)=C(p,q)-H(p).
\end{aligned}
```

Here $`C(p,q)`$ names cross-entropy. Subtracting the first sum from the second creates $`\log p_k-\log q_k=\log(p_k/q_k)`$ term by term. Terms with zero $`p_k`$ contribute zero by continuity; positive $`p_k`$ with zero $`q_k`$ gives infinite loss. A distribution’s support is the region where it places probability; assigning zero model probability to a possible observed event cannot give finite log-loss.

In the identity below, expectation means “average over observations drawn from the population.” To derive it, add and subtract the average of $`\log p_{\mathrm{data}}(x)`$ inside the loss. The term involving only the data distribution is entropy; the remaining log ratio is KL. If the model assigns zero probability to a region with positive data probability, this KL can be infinite.

Maximum likelihood minimizes $`\mathbb{E}_{x\sim p_{\mathrm{data}}}[-\log p_\theta(x)]`$. When densities and integrals are well defined,

```math
\mathbb{E}_{x\sim p_{\mathrm{data}}}[-\log p_\theta(x)]
= H(p_{\mathrm{data}})
+ D_{\mathrm{KL}}(p_{\mathrm{data}}\Vert p_\theta).
```

**Worked decomposition.** With population probabilities `(0.8, 0.2)` and model probabilities `(0.5, 0.5)`, entropy is $`-0.8\log0.8-0.2\log0.2\approx0.5004`$. Cross-entropy is $`-0.8\log0.5-0.2\log0.5\approx0.6931`$. KL is $`0.8\log(0.8/0.5)+0.2\log(0.2/0.5)\approx0.1927`$: the additional loss, with $`0.5004+0.1927\approx0.6931`$. In a continuous setting the entropy term is called **differential entropy**; the density and units qualifications below apply.

The entropy term does not depend on $`\theta`$. Thus likelihood fits a distribution; it does not directly optimize a human quality score. For continuous observations, density values depend on units and can exceed one; negative log density can be negative. Comparing densities after rescaling coordinates requires a volume correction: stretching an interval spreads the same probability over more space. In several dimensions, stretching two coordinates by factors 2 and 3 enlarges area by a factor of 6, so density must be divided by 6 to preserve probability. This local volume correction is called a Jacobian correction.

A concrete density example makes the units issue visible. A uniform distribution on an interval of width 0.1 has density 10 throughout that interval, yet its total probability is one. Its NLL is $`-\log 10`$, which is negative. If the same variable is expressed in units that stretch the interval to width one, the density becomes one. Nothing about the observations became more predictable; only the reference scale changed. Compare continuous likelihoods only after aligning transformations and units.

### Training fit is not a generalization result

Training observations determine parameter updates. Validation observations help select settings such as learning rate or model size. A final test set is reserved to assess the selected procedure on observations not used for these decisions. Repeatedly choosing settings based on a test set turns it into part of the selection process.

**Overfitting** means fitting peculiarities of the training data that do not transfer to the target population. A model might memorize training examples, lower its training loss and still perform poorly on new observations. Generalization asks whether performance transfers. Neither a lower training loss nor a visually attractive generated example establishes that transfer.

**Try it together:** a model’s training NLL falls from 2.0 to 1.0 while NLL on the same held-out set rises from 2.1 to 2.4. Has learning improved?

**Worked answer:** fit to training data improved; the held-out evidence worsened. This does not support improved generalization. Check data splits and model-selection choices before attributing the difference to a particular mechanism. For a concrete split example, fit candidate settings on 60 observations, choose between them on 20 validation observations, and assess the selected procedure on 20 untouched test observations. These counts illustrate the roles, not an adequate sample size for every study.

## 3. Representations and sequences

A **vector** is an ordered list of numbers. A **matrix** is a rectangular table. A **tensor** extends this idea to any number of axes. Shape tells you the length of each axis, not what the numbers mean. In the labs, an array with shape `(256, 2)` represents 256 observations with two coordinates each.

A dot product multiplies corresponding coordinates and sums them: the dot product of `(1, 2)` and `(3, 4)` is 11. Matrix multiplication takes a dot product between every row of the first matrix and every column of the second. Thus an $`m\times n`$ matrix times an $`n\times k`$ matrix produces an $`m\times k`$ matrix. The shared inner dimension must agree. Transposing a matrix exchanges rows and columns.

For attention, $`Q`$ and $`K`$ each have shape $`(B,T,d)`$: batch size, token count and feature width. Transpose the last two axes of $`K`$ and multiply. The result $`QK^\top`$ has shape $`(B,T,T)`$, with one score for each query–key pair in each example. A batch is a group of independent examples processed together; it is not another coordinate of one observation.

A **reduction** combines values along an axis. For per-coordinate losses `(1, 2)` and `(3, 4)` in a batch of two observations, summing coordinates gives per-example losses 3 and 7. Averaging examples then gives 5. Averaging all four entries would give 2.5, which changes the scale of the objective. This is why the lessons distinguish event sums from batch means.

### Predict one token from the observed prefix

A **token** is a unit of a sequence, such as a word, subword or character; a tokenizer specifies that division. A **prefix** is the sequence before the position currently being predicted. A language model assigns probabilities to tokens in context. In the examples below, each displayed word is one token and the sequence has a fixed length of three.

For a sequence, the chain rule expresses its joint probability as successive conditional probabilities. This identity does not assume that only the preceding token matters. A **bigram** approximation conditions only on the immediately preceding token; the transformer model introduced in the attention discussion can instead use a much longer prefix. The notation $`x_{\lt i}`$ means all positions before position $`i`$.

An autoregressive model uses

```math
p_\theta(x)=\prod_i p_\theta(x_i\mid x_{\lt i}).
```

Teacher forcing evaluates observed prefixes during training. At generation time prefixes are sampled. Perplexity is $`\exp(\text{mean token NLL})`$, with natural logarithms, and depends on the tokenizer and evaluation text. Perplexities from different tokenizers are not automatically comparable.

For the three-token sequence “the cat sleeps”, suppose the conditional probabilities of its observed tokens are 0.5, 0.2 and 0.1. Its joint probability is their product, 0.01. Its total NLL is about 4.6052 nats; dividing by three gives 1.5351 nats per token and perplexity about 4.64. Perplexity exponentiates average surprise: a predictor assigning probability one quarter to every observed token has perplexity four. It does not mean that every prediction literally has four equally likely alternatives.

During teacher forcing, the prediction of “sleeps” receives the observed prefix “the cat”, even if the model would itself have sampled a different second token. During generation, an earlier sampled token becomes part of the later input. An early mistake can therefore change every subsequent conditional distribution. This explains why a token-level likelihood measurement and an evaluation of complete generated sequences answer different questions.

**Try it together:** treat “I like tea” as exactly three tokens, with observed-token probabilities 0.4, 0.5 and 0.25. Write the factorization, compute the probability, and name the prefix used to predict the last token during training.

**Worked answer:**

```math
p(\text{I},\text{like},\text{tea})
=p(\text{I})p(\text{like}\mid\text{I})p(\text{tea}\mid\text{I},\text{like})
=0.4\times0.5\times0.25=0.05.
```

The last training prefix is `I like` and its target is `tea`. The target is scored but must not be part of the input used to predict itself. A model-generated prefix may differ at generation time.

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

### What this says about a language model

An **embedding** maps a token to a learned vector. Attention mixes information from allowed positions; it does not yet assign probabilities to vocabulary items. A later learned output map produces one real-valued score, or **logit**, for each candidate next token, and softmax turns these scores into probabilities. A full transformer combines several attention computations with additional learned transformations, positional information and stabilization operations.

For input `[start, the, cat]`, score targets `[the, cat, sleeps]`. At input `cat`, attention may use `start`, `the` and `cat` itself; the target `sleeps` is not present in that prefix. This is why allowing the causal-mask diagonal does not leak the next-token target.

**Discuss:** why is “future changes leave earlier outputs unchanged” insufficient by itself?

**Worked answer:** an implementation returning zero for every input passes that test. In the example above, changing an allowed past value must change output 5 to 6. This positive control shows sensitivity to available information. Neither check establishes a complete or useful language model.

## 4. What can we conclude from an example?

A generator must do more than produce an attractive individual output. On the eight-cluster point distribution, a system that always produces points in one cluster may look locally realistic but miss seven clusters. That is a **coverage** failure. A system that reaches every cluster but scatters points far from them has a **fidelity** failure. A single picture or loss number may not distinguish these failures.

The next meeting develops different ways to generate a distribution:

| Approach | Intuition | Question to carry forward |
|---|---|---|
| Autoregressive model | Generate successive parts conditional on earlier parts | Which information is available at each prediction? |
| Variational autoencoder (VAE) | Draw a hidden variable, then draw an observation conditional on it | How can we train when many hidden explanations could produce the same observation? |
| Generative adversarial network (GAN) | Train a generator against a classifier distinguishing real and generated data | Can realistic-looking samples still miss part of the distribution? |
| Invertible flow | Transform simple random values through a reversible map | How does stretching space change density while preserving probability? |
| Diffusion | Learn to undo a specified corruption process | What can be learned from a noisy observation when its clean source is unknown? |
| Flow matching | Learn a velocity that moves a noise cloud toward the data distribution | How do learned-velocity errors differ from numerical-solver errors? |

These are organizing ideas, not claims that one family is always best.

**Closing discussion, with worked responses**

| Question | Worked response |
|---|---|
| Does sampling mean choosing the most probable outcome? | No. Sampling draws according to the probabilities; repeatedly choosing the maximum is a different rule and changes the output distribution. |
| Can a probability be 10? | A discrete outcome probability cannot. A continuous density can be 10 over an interval of width 0.1 and still integrate to total probability 1. |
| What changed in the eight-A/two-B example? | The fitted probability of A became 0.8; this fits observed frequencies, not necessarily the unknown population. |
| What does a smaller training loss establish? | Better fit to that objective on those training observations. Fresh-data performance and output usefulness require separate evidence. |
| What should a research comparison hold fixed? | Factors that could otherwise explain the difference: data, measurement procedure, computation and available information. The appropriate controls depend on the claim. |

Record one claim the group accepts, the assumption it needs, one unresolved objection, and the evidence that would resolve that objection. A disagreement is a useful outcome if its reason is explicit. The worked answers illustrate a defensible argument; they are not a required consensus.

**First-meeting record:** write one sentence each defining a distribution, a loss, a sample and held-out evidence. Then record one concept the group wants clarified at the beginning of Lesson 02. This is the whole first-meeting activity; there is no installation exercise or advance-reading requirement.

## Optional sources after the meeting

The page above contains the material needed for the discussion. These links are for tracing ideas to their sources after the meeting, not preparation required to attend.

- [Attention Is All You Need — Vaswani et al., version 7](https://arxiv.org/abs/1706.03762v7): source for scaled attention and transformer architecture. Our scalar masking example is an explanatory calculation, not a replication of the paper’s translation results.
- [Existing foundation reference](../docs/foundations.md): another presentation of the same vocabulary, available if useful.

Continue with [Lesson 02 — How generative models work](02-generative-models.md).
