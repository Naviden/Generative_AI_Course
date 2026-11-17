# Further reading — mathematics and sequence models

[Back to Lesson 01](../01-foundations.md)

These optional notes preserve the detailed explanations and worked examples beyond the four introductory topics. They can be consulted when a later paper needs the relevant mathematics; they are not preparation for the introductory meeting.

The running record example has 100 observations: 20 blue and large, 20 blue and not large, 10 not blue and large, and 50 neither. Its fitted probabilities are `(0.20, 0.20, 0.10, 0.50)`. An observation is written $`x`$ and the model's parameters $`\theta`$.

## 1. Probability identities and continuous models

The **probability chain rule**, $`p(A,B)=p(A)p(B\mid A)`$, also gives a way to generate a complete record in stages. First sample its colour, then sample its size conditional on that colour. For the blue-and-large row, $`0.40\times0.50=0.20`$ recovers the joint probability. Section 3 applies this same rule to successive tokens in a sequence.

**Independence** would mean that learning the colour does not change the size probabilities, so the joint factors as $`p(A,B)=p(A)p(B)`$. Our fitted table is not independent: blue and large have joint probability 0.20, whereas $`0.40\times0.30=0.12`$. Generating the two attributes independently would therefore reproduce their separate totals but get their combinations wrong. **Bayes' rule** reverses a conditional using the same joint distribution: $`p(A\mid B)=p(B\mid A)p(A)/p(B)`$. Here $`p(\text{blue}\mid\text{large})=0.50\times0.40/0.30=2/3`$. Later, a latent-variable model uses this reversal to infer hidden variables from observed outputs.

A **discriminative** model typically learns to predict a designated target from an input, such as size given colour. Our joint generative model describes complete records and can also answer conditional questions. A conditional generative model instead starts with a specified context and models the distribution of outputs, such as whole images given a prompt; the presence of a condition alone does not make a model discriminative.

### How likelihood connects the examples to learning

Why choose the fitted probabilities above instead of assigning 0.25 to every row? We need a rule for comparing candidate generators using the observed records. **Likelihood** supplies one such rule: hold the dataset fixed and ask how probable it is under each parameter choice.

Assume the 100 records are independent draws from the same model distribution. If their four probabilities are $`a,b,c,d`$ in table order, the probability of the particular observed sequence is

```math
L(a,b,c,d)=a^{20}b^{20}c^{10}d^{50}.
```

There are 20 factors of $`a`$ because 20 observed records belong to the first row, and similarly for the other rows. A power abbreviates repeated multiplication. The parameters must remain nonnegative and sum to one. **Maximum likelihood** chooses parameters that make this observed sequence as probable as possible within the model family. For this unrestricted four-row table, that gives the sample frequencies above. The records stay fixed while we compare probability assignments; this is why likelihood is a function of the parameters.

### Why the training objective uses a logarithm and a minus sign

The likelihood multiplies many small numbers. We use natural logarithms throughout. A logarithm turns multiplication into addition: $`\log(ab)=\log a+\log b`$. It is increasing, so the parameter choice with the largest likelihood also has the largest log likelihood. A **loss** is a single number that training tries to minimize. Negating log likelihood turns that maximization into minimization, and dividing by the number of records gives the average **negative log likelihood**, or NLL:

```math
J(a,b,c,d)
=-\frac{20\log a+20\log b+10\log c+50\log d}{100}.
```

**Worked comparison:** the equal-probability model has average NLL $`-\log0.25\approx1.3863`$. The frequency-fitted model has average NLL

```math
-0.20\log0.20-0.20\log0.20-0.10\log0.10-0.50\log0.50
\approx1.2206.
```

The smaller value means that the fitted model assigns higher likelihood to these training observations. It also changes generation: the probability of producing a large record becomes $`0.20+0.10=0.30`$, compared with $`0.25+0.25=0.50`$ in the equal-probability model. This is the connection between an improved training objective and a changed generator.

For one observation the contribution is $`-\log p_\theta(x)`$. Assigning a very small probability to something that actually occurred gives a large penalty. But increasing one probability leaves less probability for other outcomes: the model must fit the collection of examples, not make every observed outcome have probability one. Section 2 shows how a parameter update can reduce this loss.

Likelihood is one training principle, not a requirement that every generative model can evaluate exact observation probabilities efficiently. Later models use bounds or other training objectives because direct likelihood calculations can be difficult. They still need an explicit account of how training changes the distribution of generated observations.

### Averages and variability

A generator produces a distribution of possible outputs, so one draw cannot tell us its typical behaviour. An **expectation** is a probability-weighted average. Suppose another simple generator outputs the numerical value 2 with probability 0.75 or 6 with probability 0.25. Its expected output is $`0.75(2)+0.25(6)=3`$. The mean is 3 even though the generator never outputs 3: describing a distribution by its mean loses information about its samples.

The notation $`\mathbb{E}[f(X)]`$ means average a quantity computed from the random output, rather than necessarily averaging the output itself. For generative-model training, the quantity can be the loss $`f(x)=-\log p_\theta(x)`$. Averaging that loss over observed records estimates how the model scores fresh observations from the same population. The symbol $`\sum`$ abbreviates adding contributions, while $`\prod`$ abbreviates multiplying them.

**Variance** measures average squared deviation from the mean. In the numerical generator above it is $`0.75(2-3)^2+0.25(6-3)^2=3`$. Standard deviation is the square root of variance. These quantities describe how spread out generated values are. A model that always outputs 3 has the same mean but zero variance and the wrong distribution. Independent noise variances add; their standard deviations generally do not. This matters when we later construct a generator by adding or removing noise.

### A continuous example: the Gaussian

The record table has a finite set of possibilities. Such outcomes are **discrete**; more generally a discrete set can be countably infinite. A **continuous** variable can take values throughout an interval. The repository's point-generation labs use two continuous coordinates, with real observations concentrated around eight clusters. Their task is to generate a similar cloud of points, not predict a cluster label.

For continuous values, a model assigns a **density**, or probability per unit of the variable. A density at one exact value is not that value's probability; probability comes from a region. An **integral** accumulates density over that region, like adding density times the widths of many thin intervals. A uniform distribution on an interval of width 0.1 has density 10 and total probability $`10\times0.1=1`$. This is why a continuous generator can have density values greater than one without violating probability rules.

A **Gaussian**, or normal distribution, is a simple continuous generative model with a bell-shaped density, written $`\mathcal{N}(\mu,\sigma^2)`$. The mean $`\mu`$ sets its centre and the standard deviation $`\sigma`$ sets its spread. These can be learned parameters. To generate from it, first draw standard-normal noise $`\epsilon`$ with mean zero and variance one, then compute $`z=\mu+\sigma\epsilon`$. Shifting and scaling the noise turns a fixed source of randomness into the chosen output distribution. With mean 1, standard deviation 2 and noise draw 0.5, the sample is 2.

The exponential $`\exp`$ is the inverse of the natural logarithm: $`\log1=0`$ and $`\exp(0)=1`$. Later an encoder predicts log variance, a number that can be positive or negative, and exponentiates it to obtain a positive variance. This lets a neural network control the spread of a sampling distribution without ever producing an invalid negative variance.

For multiple coordinates, a **covariance matrix** is a table containing coordinate variances on its diagonal and their joint linear variability off the diagonal. It determines the shape and orientation of a Gaussian cloud. An identity covariance $`I`$ gives unit variance in every coordinate and zero covariance between different coordinates. For a multivariate Gaussian, diagonal covariance also implies independent coordinates. The notation $`\mathrm{diag}(\sigma^2)`$ places the coordinate variances on the diagonal and zeros elsewhere. A single Gaussian cannot represent arbitrary separated clusters; more expressive models learn transformations or combinations of simple distributions.

**Exercise:** a normal variable has mean 1 and standard deviation 2. A standard-normal draw is 0.5. What sample does shifting and scaling produce?

**Worked answer:** `1 + 2 × 0.5 = 2`. Holding that draw fixed, increasing the mean to 1.1 gives 2.1; increasing the standard deviation to 2.1 instead gives 2.05. The random draw and the distribution parameters have different roles. When the mean or scale changes during learning, the distribution of future generated samples changes with it.

## 2. How data changes a model

The introductory lesson compared two candidate probability tables. Training needs a procedure for moving from the current table toward a better one. The connection is **parameters → probabilities of observed examples → loss → parameter update → a changed sampling distribution**. A derivative tells us which small parameter change would lower the loss; a gradient collects that information when there are many parameters. The purpose of the calculations below is to change what the generator will produce, not merely to make a number smaller.

### Why unlikely observations cost more

For a generative model trained by likelihood, each observed example asks the model to reserve probability for that kind of outcome. A negative log likelihood turns the assigned probability into a penalty. An observed outcome assigned probability 0.5 costs $`-\log0.5\approx0.6931`$; at probability 0.1 it costs $`-\log0.1\approx2.3026`$. Smaller assigned probability gives larger loss. Natural-log units are called **nats**. Because $`\log(ab)=\log a+\log b`$, multiplying probabilities of independent observations becomes adding their log probabilities.

### Worked example: fitting frequencies

A derivative, written $`J'(a)`$, measures the local rate at which the loss changes when $`a`$ changes. At an interior minimum it must be zero; positive second derivative means the curve bends upward. Here the rules are $`(\log a)'=1/a`$ and $`(\log(1-a))'=-1/(1-a)`$.

To calculate a complete learning step by hand, simplify the four-row generator to a generator of just two possible outputs, A and B. A sample from this model is one of those two outputs. Suppose ten independent observations contain eight A outcomes and two B outcomes. Let $`a=p_\theta(A)`$, so normalization requires $`p_\theta(B)=1-a`$. The average NLL is

```math
J(a)=-0.8\log a-0.2\log(1-a),\qquad
J'(a)=-\frac{0.8}{a}+\frac{0.2}{1-a}.
```

Setting the derivative to zero gives $`a=0.8`$. The fitted generator will sample A with probability 0.8 and B with probability 0.2; fitting has changed the frequencies of its future outputs.

The algebra is $`0.2a=0.8(1-a)`$, hence $`a=0.8`$. The second derivative is positive on the interior of the probability interval, so this is the minimum. At that solution the average NLL is about 0.5004 nats, compared with 0.6931 for equal probabilities. Assigning probability one to A would give infinite loss on the two B observations. Likelihood rewards fitting frequencies rather than just picking the most common outcome.

These are fitted **sample** frequencies. With only ten observations they need not equal population frequencies. A model that memorizes an enormous set of training sequences may likewise have excellent training likelihood and poor likelihood on new sequences. Held-out data tests this distinction.

### What a parameter update does

The two-outcome example has a closed-form optimum, but a neural generator usually has too many interacting parameters to solve for its optimum directly. Instead we evaluate the current loss and repeatedly take small steps that reduce it locally. We can see what such a step means for generation before introducing the general notation.

**Worked update of the generator:** start with $`a=0.5`$ in the eight-A/two-B example. The loss derivative is $`J'(0.5)=-0.8/0.5+0.2/0.5=-1.2`$. With step size 0.1, subtract this derivative times the step size: $`a_{\mathrm{new}}=0.5-0.1(-1.2)=0.62`$. The probability of B becomes 0.38 so that the probabilities still sum to one. Average NLL falls from about 0.6931 to 0.5759. The next generated output is now more likely to be A. This update changes both a measurable loss and an interpretable sampling behaviour.

This particular step stays within the valid probability interval. Arbitrary direct updates need not do so; neural models usually produce unconstrained scores and turn them into valid probabilities with a normalization such as the softmax introduced in Section 3.

A **derivative** measures local rate of change. For $`f(w)=w^2`$, its derivative is $`2w`$: near $`w=3`$, a small increase in $`w`$ changes the output by approximately six times that increase. A **partial derivative** changes one input while holding the others fixed. A **gradient** collects all partial derivatives of a scalar function into a vector—an ordered list of numbers. It points in the direction of greatest local increase under the usual Euclidean geometry.

Gradient descent adjusts parameters in the opposite direction to reduce a loss:

```math
\theta_{\mathrm{new}}=\theta-\eta\nabla_\theta J(\theta).
```

Here $`J`$ is the loss and $`\eta`$ is the positive step size, or learning rate. For a continuous generator, let $`w`$ be a Gaussian mean, fix its variance to 1/2 and observe the value 3. Its NLL, apart from a constant independent of $`w`$, is $`J(w)=(w-3)^2`$. Starting at $`w=0`$ gives derivative −6. With learning rate 0.1 the next value is 0.6, and the loss falls from 9 to 5.76. Generated values are now centred at 0.6 instead of zero, closer to the observed value 3. An overly large step can overshoot; a gradient step is not a guarantee of finding a global minimum.

In a neural generator, a weight affects an intermediate computation, which affects predicted probabilities, which affect the loss. We need to carry the loss sensitivity back through that chain. The **chain rule** differentiates a composition by multiplying local derivatives. If $`u=2w`$ and $`J=u^2`$, then $`\mathrm{d}J/\mathrm{d}w=(2u)(2)=8w`$. Backpropagation applies this rule through a computation graph. Automatic differentiation records those operations and computes their derivatives; it does not estimate them by trying random parameter changes.

A neural network composes simple parameterized operations. In one scalar operation, `w × x + b` multiplies input `x` by a learned **weight** `w` and adds a learned **bias** `b`. An **activation** applies a nonlinear function, such as replacing a negative result with zero. For `x = 2`, `w = 0.5` and `b = −0.2`, the result is 0.8 and this activation leaves it at 0.8; for `x = 0`, the result −0.2 becomes zero. Composing many such operations allows a model to represent relationships beyond a single straight line. The weights and biases are parameters adjusted by training. In a generative model these network outputs determine quantities such as next-token probabilities, a Gaussian mean or a velocity used to move noise toward data. Training computes those outputs, evaluates the relevant loss, differentiates it and updates parameters. Sampling later uses the updated outputs with the weights held fixed. Minibatch training estimates gradients using a subset of observations at a time. An **epoch** traverses a dataset; the lab's `--steps` flag instead counts parameter updates.

### Entropy, cross-entropy and the direction of KL

The fitted A/B generator still has average loss about 0.5004. This nonzero loss is expected: a model that correctly represents random A/B outcomes cannot assign probability one to whichever outcome happens to appear next. We need to separate uncertainty inherent in the source from the extra loss caused by wrong model probabilities. Entropy and KL make that distinction precise; cross-entropy is the total expected NLL we have been minimizing.

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

If the true A/B probabilities were `(0.8, 0.2)`, changing the generator from `(0.5, 0.5)` to `(0.8, 0.2)` would remove the extra 0.1927 nats of loss, while preserving the source’s genuine randomness. It should still sometimes generate B. Always generating A would give B probability zero and infinite NLL whenever B occurs. This explains the direction $`D_{\mathrm{KL}}(p_{\mathrm{data}}\Vert p_\theta)`$: outcomes that occur under the data distribution must be accounted for by the model. Exchanging the two distributions changes which outcomes receive weight.

The entropy term does not depend on $`\theta`$. Thus likelihood fits a distribution; it does not directly optimize a human quality score. For continuous observations, density values depend on units and can exceed one; negative log density can be negative. Comparing densities after rescaling coordinates requires a volume correction: stretching an interval spreads the same probability over more space. In several dimensions, stretching two coordinates by factors 2 and 3 enlarges area by a factor of 6, so density must be divided by 6 to preserve probability. This local volume correction is called a Jacobian correction.

A concrete density example makes the units issue visible. A uniform distribution on an interval of width 0.1 has density 10 throughout that interval, yet its total probability is one. Its NLL is $`-\log 10`$, which is negative. If the same variable is expressed in units that stretch the interval to width one, the density becomes one. Nothing about the observations became more predictable; only the reference scale changed. Compare continuous likelihoods only after aligning transformations and units.

## 3. Representations and sequences

A four-row table can store the probability of every complete record. A table of complete sentences quickly becomes impractical: even fixed three-token sequences over a vocabulary of 1,000 tokens have a billion possibilities. A language generator instead computes a distribution for the next token from the available prefix and repeats that operation. The chain rule connects those smaller distributions to the probability of a complete sequence. Vectors and attention are tools for computing these conditional probabilities from shared learned parameters.

A **vector** is an ordered list of numbers. A **matrix** is a rectangular table. A **tensor** extends this idea to any number of axes. Shape tells you the length of each axis, not what the numbers mean. In the labs, an array with shape `(256, 2)` represents 256 observations with two coordinates each.

A dot product multiplies corresponding coordinates and sums them: the dot product of `(1, 2)` and `(3, 4)` is 11. Matrix multiplication takes a dot product between every row of the first matrix and every column of the second. Thus an $`m\times n`$ matrix times an $`n\times k`$ matrix produces an $`m\times k`$ matrix. The shared inner dimension must agree. Transposing a matrix exchanges rows and columns.

A **reduction** combines values along an axis. For per-coordinate losses `(1, 2)` and `(3, 4)` in a batch of two observations, summing coordinates gives per-example losses 3 and 7. Averaging examples then gives 5. Averaging all four entries would give 2.5, which changes the scale of the objective. For independent coordinates under a chosen likelihood, the complete-observation log probability adds coordinate log probabilities; this is why summing their losses can be appropriate. Averaging across examples then estimates the loss for one observation. Changing that reduction changes the training scale and can change the balance between reconstruction and regularization. This is why the lessons distinguish event sums from batch means.

### Predict one token from the observed prefix

A **token** is a unit of a sequence, such as a word, subword or character; a tokenizer specifies that division. A **prefix** is the sequence before the position currently being predicted. A language model assigns probabilities to tokens in context. In the examples below, each displayed word is one token and the sequence has a fixed length of three.

For a sequence, the chain rule expresses its joint probability as successive conditional probabilities. This identity does not assume that only the preceding token matters. A **bigram** approximation conditions only on the immediately preceding token; the transformer model introduced in the attention discussion can instead use a much longer prefix. The notation $`x_{\lt i}`$ means all positions before position $`i`$.

An autoregressive model uses

```math
p_\theta(x)=\prod_i p_\theta(x_i\mid x_{\lt i}).
```

Training uses the NLL from Section 2, now applied to the conditional probability of each observed token. Taking the negative logarithm of the product above adds these token losses. **Teacher forcing** supplies the observed prefix when scoring each training token, allowing the model to learn from the actual sequence. Generation instead starts from a supplied or empty prefix, samples a token, appends it, and uses the enlarged prefix to predict the next one. Perplexity is $`\exp(\text{mean token NLL})`$, with natural logarithms, and depends on the tokenizer and evaluation text. Perplexities from different tokenizers are not automatically comparable.

For the three-token sequence “the cat sleeps”, suppose the conditional probabilities of its observed tokens are 0.5, 0.2 and 0.1. Its joint probability is their product, 0.01. Its total NLL is about 4.6052 nats; dividing by three gives 1.5351 nats per token and perplexity about 4.64. Perplexity exponentiates average surprise: a predictor assigning probability one quarter to every observed token has perplexity four. It does not mean that every prediction literally has four equally likely alternatives.

During teacher forcing, the prediction of “sleeps” receives the observed prefix “the cat”, even if the model would itself have sampled a different second token. During generation, an earlier sampled token becomes part of the later input. An early mistake can therefore change every subsequent conditional distribution. This explains why a token-level likelihood measurement and an evaluation of complete generated sequences answer different questions.

**Exercise:** treat “I like tea” as exactly three tokens, with observed-token probabilities 0.4, 0.5 and 0.25. Write the factorization, compute the probability, and name the prefix used to predict the last token during training.

**Worked answer:**

```math
p(\text{I},\text{like},\text{tea})
=p(\text{I})p(\text{like}\mid\text{I})p(\text{tea}\mid\text{I},\text{like})
=0.4\times0.5\times0.25=0.05.
```

The last training prefix is `I like` and its target is `tea`. The target is scored but must not be part of the input used to predict itself. A model-generated prefix may differ at generation time.

### What attention computes

To predict “sleeps” after “the cat”, the network needs a numerical representation of that prefix. Attention helps build it by selecting and combining information from permitted positions. Its output is an intermediate representation, not a generated token. A later output map converts that representation into token probabilities, and NLL sends a learning signal back through both operations.

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

For attention, $`Q`$ and $`K`$ each have shape $`(B,T,d)`$: batch size, token count and feature width. Transpose the last two axes of $`K`$ and multiply. The result $`QK^\top`$ has shape $`(B,T,T)`$, with one score for each query–key pair in each example. A batch is a group of independent examples processed together; it is not another coordinate of one observation.

### Worked example: masking changes the distribution

Suppose a query at the second position has scaled scores $`(0,\log 3,10)`$ for three source positions. The third source is in the future. Without masking, its large score would dominate the softmax, leaking information unavailable during generation. A causal mask replaces that score with negative infinity before softmax. The weights become $`(1/4,3/4,0)`$. For scalar values $`(2,6,100)`$, the output is 5, independently of the future value 100.

Explicitly, $`(1/4)2+(3/4)6+0(100)=5`$. Changing the future value from 100 to 200 still gives 5. Changing the allowed first value from 2 to 6 instead gives $`(1/4)6+(3/4)6=6`$. The latter is the **positive control**: it shows that the output responds to permitted information, so a constant-output implementation would not pass both checks.

Masking only after softmax would leave weights on the first two positions summing to less than one unless they were renormalized. That would change the scale and meaning of the output. A mask that excludes every source also needs explicit handling: a softmax over all negative infinities has no valid normalized distribution.

For batch size 2, sequence length 6, query/key width 8 and value width 4, the scores and weights have shape $`(2,6,6)`$ and the output has shape $`(2,6,4)`$. The matrix product mixes **positions**; it does not turn a hidden vector into a vocabulary probability distribution. A later output projection and vocabulary softmax do that.

### What this says about a language model

An **embedding** maps a token to a learned vector. Attention mixes information from allowed positions; it does not yet assign probabilities to vocabulary items. A later learned output map produces one real-valued score, or **logit**, for each candidate next token, and softmax turns these scores into probabilities. A full transformer combines several attention computations with additional learned transformations, positional information and stabilization operations.

**Worked connection from attention to a generated token:** suppose the attention calculation above supplies the scalar representation 5. In a deliberately tiny two-token output vocabulary, let an output map give “sleeps” the score $`(\log3/5)\times5=\log3`$ and “runs” the score 0. Vocabulary softmax produces probabilities 0.75 and 0.25. If the observed next token is “sleeps”, its training loss is $`-\log0.75\approx0.2877`$. At generation time, a draw below 0.75 selects “sleeps”; otherwise it selects “runs”. The value 5 was a representation, 0.75 is an output probability, and the selected word is a sample. These are three different stages of one generator.

In a real model, the output map and the attention projections are learned together. Backpropagation carries the next-token loss through the output map into the representation and its attention computations. This is how attention participates in learning the generative distribution.

For input `[start, the, cat]`, score targets `[the, cat, sleeps]`. At input `cat`, attention may use `start`, `the` and `cat` itself; the target `sleeps` is not present in that prefix. This is why allowing the causal-mask diagonal does not leak the next-token target.

**Discuss:** why is “future changes leave earlier outputs unchanged” insufficient by itself?

**Worked answer:** an implementation returning zero for every input passes that test. In the example above, changing an allowed past value must change output 5 to 6. This positive control shows sensitivity to available information. Neither check establishes a complete or useful language model.

## 4. Comparing model families

Generative approaches differ in how they construct and learn a distribution:

| Approach | Intuition | Main question |
|---|---|---|
| Autoregressive model | Generate successive parts conditional on earlier parts | Which information is available at each prediction? |
| Variational autoencoder (VAE) | Draw a hidden variable, then draw an observation conditional on it | How can we train when many hidden explanations could produce the same observation? |
| Generative adversarial network (GAN) | Train a generator against a classifier distinguishing real and generated data | Can realistic-looking samples still miss part of the distribution? |
| Invertible flow | Transform simple random values through a reversible map | How does stretching space change density while preserving probability? |
| Diffusion | Learn to undo a specified corruption process | What can be learned from a noisy observation when its clean source is unknown? |
| Flow matching | Learn a velocity that moves a noise cloud toward the data distribution | How do learned-velocity errors differ from numerical-solver errors? |

These are organizing ideas, not claims that one family is always best.

**Questions and worked responses**

| Question | Worked response |
|---|---|
| Does sampling mean choosing the most probable outcome? | No. Sampling draws according to the probabilities; repeatedly choosing the maximum is a different rule and changes the output distribution. |
| Can a probability be 10? | A discrete outcome probability cannot. A continuous density can be 10 over an interval of width 0.1 and still integrate to total probability 1. |
| What changed in the eight-A/two-B example? | The fitted probability of A became 0.8; this fits observed frequencies, not necessarily the unknown population. |
| What does a smaller training loss establish? | Better fit to that objective on those training observations. Fresh-data performance and output usefulness require separate evidence. |
| What should a research comparison hold fixed? | Factors that could otherwise explain the difference: data, measurement procedure, computation and available information. The appropriate controls depend on the claim. |

## Sources

- [Attention Is All You Need — Vaswani et al., version 7](https://arxiv.org/abs/1706.03762v7): the source for scaled attention and transformer architecture. The scalar calculations here illustrate the mechanism; they do not reproduce the paper's experiments.
- [Additional foundation reference](../../docs/foundations.md).

[Back to Lesson 01](../01-foundations.md)
