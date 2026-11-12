# Entry requirements and preparation

The intended audience is doctoral students or researchers familiar with a graduate machine-learning course. Required knowledge: multivariate calculus and chain rule; matrix multiplication and eigenvalues; conditional probability and Bayes' rule; expectation, Gaussian distributions and maximum likelihood; gradient-based optimization, train/validation/test splits and basic neural networks; readable Python and basic tensor operations.

No prior diffusion, RLHF or large-model engineering experience is required. A laptop capable of running Python 3.12 with about 4 GB of available RAM is sufficient for the small labs; this is a planning estimate, not a measured minimum. No GPU or paid account is needed.

## Before class (suggested 60–90 minutes, outside contact time)

Install the environment using [setup](setup.md), run the attention lab and tests, and attempt the diagnostic below. The instructor should collect anonymous confidence ratings, not grades. If several topics are unfamiliar, complete a probability/deep-learning refresher before enrolling; the lessons assume these prerequisites.

## Diagnostic (15 minutes)

1. Write the factorization of $`p(x_1,x_2,x_3)`$. What independence assumption does a bigram model add?
2. For $`z\sim\mathcal{N}(\mu,\mathrm{diag}(\sigma^2))`$, write a reparameterized sample using standard Gaussian noise. Which parameters receive gradients?
3. If $`Q`$ has shape $`(B,T,d)`$ and $`K`$ has shape $`(B,T,d)`$, what shape does $`QK^\top`$ have? Why must an autoregressive mask remove $`j>i`$?
4. Explain why a lower training loss does not establish better generation on a new population.
5. Compute $`D_{\mathrm{KL}}(\mathcal{N}(\mu,I)\Vert\mathcal{N}(0,I))`$ when $`\mu=(1,2)`$.
6. Distinguish a validation set used to select hyperparameters from a final test set.

Self-check:

1. $`p(x_1)p(x_2\mid x_1)p(x_3\mid x_1,x_2)`$; bigrams use only the preceding token.
2. $`z=\mu+\sigma\odot\epsilon`$, $`\epsilon\sim\mathcal{N}(0,I)`$; gradients pass through $`\mu`$ and $`\sigma`$.
3. $`(B,T,T)`$; future labels must not enter predictions.
4. Generalization, overfitting and the objective/metric mismatch must be examined.
5. $`\frac12(1^2+2^2)=2.5`$.
6. A repeatedly consulted test set becomes part of model selection.

Aim to answer at least four questions and explain the reasoning. Use the first 15 minutes of class to address gaps; offer optional refresher material instead of silently slowing the entire course.
