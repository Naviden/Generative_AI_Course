# Foundations used in the lessons

This is part of the learning material, not an entry test. The concepts needed for the core lessons are introduced here and developed further in the [theory notes](theory.md). You can read this page in order or return to a topic when it appears in a lesson. No separate machine-learning course or external refresher is assigned.

For installation, use [the setup guide](setup.md). The labs run on a CPU without a GPU, paid account or API key. You can read the explanations before installing anything.

## 1. Observations, variables and models

An **observation** is one item of data: a point, image or sequence of tokens. A **variable** represents a quantity that can take different values. A model uses **parameters**, usually written $`\theta`$, to determine its predictions. Learning adjusts those parameters using observations.

A discriminative model predicts a target from an input, such as a label given an image. A generative model describes a distribution of observations, possibly conditional on an input, and can produce new observations by sampling. The core VAE and flow labs use points with two coordinates, drawn from eight clusters. Their task is to learn the distribution of points, not to classify cluster labels.

## 2. Probability, conditioning and expectation

For discrete outcomes, probabilities are nonnegative and sum to one. A coin model might assign 0.6 to heads and 0.4 to tails. A **sample** is a random draw according to these probabilities, not necessarily the most probable outcome.

A joint probability describes events together. A conditional probability describes one event after another is known:

```math
p(A\mid B)=\frac{p(A,B)}{p(B)},\qquad p(B)>0.
```

**Worked example:** among 100 records, 40 are blue, 30 are large and 20 are both. The probability of blue given large is $`20/30=2/3`$. The probability of large given blue is $`20/40=1/2`$. Reversing the conditioning changes the question.

Bayes' rule reverses a conditional using the marginal probabilities:

```math
p(A\mid B)=\frac{p(B\mid A)p(A)}{p(B)}.
```

The same count example gives $`(1/2)(40/100)/(30/100)=2/3`$. In a latent-variable model, the **prior** describes latent values before seeing an observation; the **posterior** describes them after conditioning on that observation. [Theory §2](theory.md#2-latent-variables-the-elbo-and-the-vae) explains why the posterior is often difficult to calculate.

The chain rule repeatedly applies conditional probability:

```math
p(x_1,x_2,x_3)=p(x_1)p(x_2\mid x_1)p(x_3\mid x_1,x_2).
```

This is an identity. Independence is an additional assumption: independent variables satisfy $`p(A,B)=p(A)p(B)`$. A bigram language model assumes the next token depends only on its immediate predecessor, replacing the last conditional above with $`p(x_3\mid x_2)`$.

An **expectation** is a probability-weighted average. If a variable takes value 2 with probability 0.75 and value 6 with probability 0.25, its expectation is $`0.75(2)+0.25(6)=3`$. The notation $`\mathbb{E}[f(X)]`$ means to average the quantity $`f(X)`$, rather than necessarily averaging $`X`$ itself. A dataset mean approximates a population expectation using observed examples.

**Variance** measures average squared deviation from the mean. In this example it is $`0.75(2-3)^2+0.25(6-3)^2=3`$. Standard deviation is the square root of variance. Independent noise variances add; their standard deviations generally do not.

## 3. Continuous variables and Gaussians

For a continuous variable, a **density** gives probability per unit of the variable. Probability over a region is the integral of density over that region. An integral is the continuous analogue of adding contributions: approximate it by summing density times the width of many small intervals. A density value itself is not a probability and can exceed one. A uniform variable on an interval of width 0.1 has density 10 but total probability one.

A Gaussian or normal distribution is a bell-shaped density, written $`\mathcal{N}(\mu,\sigma^2)`$ in one dimension. Its mean $`\mu`$ sets the centre and its standard deviation $`\sigma`$ sets the spread. A standard normal has mean zero and variance one. If $`\epsilon\sim\mathcal{N}(0,1)`$, then $`z=\mu+\sigma\epsilon`$ has the desired mean and variance. With mean 1, standard deviation 2 and noise draw 0.5, the sample is 2.

For several coordinates, a **covariance matrix** records their variances on the diagonal and cross-coordinate dependence off the diagonal. An identity covariance $`I`$ gives each coordinate unit variance and zero covariance with the others. For a multivariate Gaussian, diagonal covariance also implies independent coordinates. The notation $`\mathrm{diag}(\sigma^2)`$ places the coordinate variances on the diagonal and zeros elsewhere.

A VAE predicts log variance $`\ell=\log\sigma^2`$. Exponentiating gives variance; exponentiating half the log variance gives standard deviation. The sampling rule is

```math
z=\mu+\exp(\ell/2)\odot\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I).
```

The symbol $`\odot`$ means coordinate-by-coordinate multiplication. [Theory §2](theory.md#a-differentiable-random-sample) connects this rule to learning; [theory §5](theory.md#worked-example-accumulating-noise-variance) shows how independent Gaussian noise accumulates.

## 4. Vectors, matrices and tensor shapes

A **vector** is an ordered list of numbers. A **matrix** is a rectangular table. A **tensor** extends this idea to any number of axes. Shape tells you the length of each axis, not what the numbers mean. In the labs, an array with shape `(256, 2)` represents 256 observations with two coordinates each.

A dot product multiplies corresponding coordinates and sums them: the dot product of `(1, 2)` and `(3, 4)` is 11. Matrix multiplication takes a dot product between every row of the first matrix and every column of the second. Thus an $`m\times n`$ matrix times an $`n\times k`$ matrix produces an $`m\times k`$ matrix. The shared inner dimension must agree. Transposing a matrix exchanges rows and columns.

For attention, $`Q`$ and $`K`$ each have shape $`(B,T,d)`$: batch size, token count and feature width. Transpose the last two axes of $`K`$ and multiply. The result $`QK^\top`$ has shape $`(B,T,T)`$, with one score for each query–key pair in each example. A batch is a group of independent examples processed together; it is not another coordinate of one observation.

A **reduction** combines values along an axis. For per-coordinate losses `(1, 2)` and `(3, 4)` in a batch of two observations, summing coordinates gives per-example losses 3 and 7. Averaging examples then gives 5. Averaging all four entries would give 2.5, which changes the scale of the objective. This is why the lessons distinguish event sums from batch means.

## 5. Logarithms, losses and KL divergence

The natural logarithm $`\log`$ is the inverse of $`\exp`$. It turns multiplication into addition: $`\log(ab)=\log a+\log b`$ for positive inputs. Consequently, the negative log likelihood of independent observations can be computed by adding their individual negative log probabilities. All logarithms in the theory use this natural base; log-loss is measured in **nats**.

If the model assigns an observed discrete outcome probability 0.5, its negative log likelihood is about 0.6931. Assigning probability 0.1 gives about 2.3026. A low assigned probability produces a larger penalty. Maximum likelihood minimizes the average of this penalty; [theory §1](theory.md#worked-example-fitting-frequencies) solves a complete two-outcome example.

KL divergence compares distributions through an expected log ratio:

```math
D_{\mathrm{KL}}(q\Vert p)=\mathbb{E}_{q}\left[\log\frac{q}{p}\right].
```

It measures the extra expected log-loss from using $`p`$ for outcomes drawn from $`q`$. It is nonnegative and generally changes when its arguments are reversed. It can be infinite if $`p`$ assigns zero probability where $`q`$ assigns positive probability. It is not the difference between two individual observations.

**Worked example:** for two Gaussians with identity covariance, the KL from mean $`\mu=(1,2)`$ to mean zero is half the sum of squared mean coordinates, $`(1^2+2^2)/2=2.5`$. [Theory §2](theory.md#2-latent-variables-the-elbo-and-the-vae) provides the more general diagonal-Gaussian formula and explains which distributions it compares in a VAE.

## 6. Derivatives, gradients and learning

A **derivative** measures local rate of change. For $`f(w)=w^2`$, its derivative is $`2w`$: near $`w=3`$, a small increase in $`w`$ changes the output by approximately six times that increase. A **partial derivative** changes one input while holding the others fixed. A **gradient** collects all partial derivatives of a scalar function into a vector. It points in the direction of greatest local increase under the usual Euclidean geometry.

Gradient descent adjusts parameters in the opposite direction to reduce a loss:

```math
\theta_{\mathrm{new}}=\theta-\eta\nabla_\theta J(\theta).
```

Here $`J`$ is the loss and $`\eta`$ is the positive step size, or learning rate. For $`J(w)=(w-3)^2`$, starting at $`w=0`$ gives derivative −6. With learning rate 0.1 the next value is 0.6, and the loss falls from 9 to 5.76. An overly large step can overshoot; a gradient step is not a guarantee of finding a global minimum.

The **chain rule** differentiates a composition by multiplying local derivatives. If $`u=2w`$ and $`J=u^2`$, then $`\mathrm{d}J/\mathrm{d}w=(2u)(2)=8w`$. Backpropagation applies this rule through a computation graph. Automatic differentiation records those operations and computes their derivatives; it does not estimate them by trying random parameter changes.

A neural network composes learned linear transformations with nonlinear functions. Weights and biases are parameters; an activation changes the representation nonlinearly. Training computes predictions, evaluates a loss, differentiates it and updates parameters. Minibatch training estimates gradients using a subset of observations at a time. An **epoch** traverses a dataset; the lab's `--steps` flag instead counts parameter updates.

Two additional calculus ideas appear later. Jensen's inequality says that applying a concave function such as log after averaging gives a value at least as large as averaging after applying the function; this produces the ELBO in [theory §2](theory.md#why-the-lower-bound-exists). An ordinary differential equation specifies a rate of change, while a numerical solver approximates the resulting path using finite steps; [theory §6](theory.md#training-error-and-integration-error-are-different) works through an Euler example. Their role and assumptions are explained where used.

## 7. Reading the Python lab code

Terminal commands launch programs; Python statements belong inside Python. The setup guide marks terminal blocks with `bash`. For the following example, activate the installed environment, run `python` in the terminal, then enter the statements at the Python prompt. Use `exit()` to leave it.

Python uses `=` to assign a value to a name, `#` for a comment, brackets for lists and indexing, and indentation to group function or loop bodies. Indexing starts at zero. `import` makes a library available; a function call supplies inputs in parentheses and may return a result. PyTorch, imported as `torch`, supplies tensors and automatic differentiation.

```python
import torch

# Two observations, each with two coordinate losses.
losses = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(losses.shape)                 # torch.Size([2, 2])
print(losses[0])                    # tensor([1., 2.])
print(losses.sum(dim=1).mean())      # tensor(5.)

# Differentiate the scalar loss from the previous section.
w = torch.tensor(0.0, requires_grad=True)
loss = (w - 3.0) ** 2
loss.backward()
print(w.grad.item())                # -6.0
```

Here `dim=1` selects the coordinate axis, `.mean()` averages the remaining examples, `** 2` squares a value, and `.item()` converts a one-element tensor to a Python number. `requires_grad=True` requests gradient tracking and `.backward()` computes derivatives of the scalar loss. This example computes a gradient but does not update `w`.

In the implementation files, `def` introduces a function and `class` groups data and methods into an object. A model's `forward` method defines its prediction computation. You do not need to read every file before the lessons: each lesson names the function to inspect and the question to answer. [The output guide](../labs/outputs.md) explains dictionaries, saved metrics and arrays in the context of an actual run.

## 8. Training, validation and testing

Training observations determine parameter updates. Validation observations help select settings such as learning rate or model size. A final test set is reserved to assess the selected procedure on observations not used for these decisions. Repeatedly choosing settings based on a test set turns it into part of the selection process.

**Overfitting** means fitting peculiarities of the training data that do not transfer to the target population. A model might memorize training examples, lower its training loss and still perform poorly on new observations. Generalization asks whether performance transfers. Neither a lower training loss nor a visually attractive generated example establishes that transfer.

A seed controls a pseudorandom generator so a computation can be repeated under matched conditions. Repeating one seed checks reproducibility; using several independent seeds probes variability across runs. [Theory §7](theory.md#7-evaluation-and-reproducibility) explains which units to compare, how sampling variability enters and why more generated points do not replace independent training runs.

## Practice with worked answers

Use these questions to consolidate the explanations. There is no pass mark or admission threshold; follow the linked explanation whenever an answer is unclear.

1. **Factor three-token probability.** The chain rule gives $`p(x_1)p(x_2\mid x_1)p(x_3\mid x_1,x_2)`$. A bigram model restricts the last factor to conditioning on $`x_2`$ only; see section 2.
2. **Sample from a diagonal Gaussian.** Draw standard Gaussian noise, then compute $`z=\mu+\sigma\odot\epsilon`$. Holding noise fixed, gradients pass through the mean and scale; see sections 3 and 6.
3. **Find attention-score shape.** Two inputs shaped $`(B,T,d)`$ produce $`(B,T,T)`$ scores after transposing the last two key axes. A causal mask excludes $`j>i`$ because future tokens are unavailable when predicting from a prefix; see section 4 and [theory §4](theory.md#4-transformers-and-scaling).
4. **Interpret a falling training loss.** It shows improved fit to that training objective, but could reflect memorization. Measure the relevant held-out outcome before claiming better generation; see section 8.
5. **Compute the Gaussian KL.** With mean `(1, 2)` and unit covariance against a standard-normal prior, the two contributions are 0.5 and 2, giving 2.5 nats; see section 5.
6. **Distinguish validation from testing.** Use validation to choose settings and reserve the final test for the resulting claim; see section 8.

Continue with [Lesson 01](../lessons/01-objectives-vae-attention/README.md). Keep this page available as a reference while working through the theory and labs.
