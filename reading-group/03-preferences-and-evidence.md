# 03 — Preferences and research evidence

Previous: [02 — How generative models work](02-generative-models.md) · [Reading-group home](README.md)

**Central question:** what does fitting a preference establish, and what would make a broader research claim credible?

The route is **adaptation objectives → the DPO derivation → evaluating a system → a testable research proposal**. Every required explanation, example and discussion prompt is on this page. Paper links and computer experiments are optional extensions, not prerequisites.

**Vocabulary reminder:** a model assigns a normalized distribution to outcomes. A loss is minimized during learning. A gradient gives local derivatives, and gradient descent moves opposite that gradient. Training observations fit parameters; validation selects settings; final test observations assess the chosen procedure. A seed fixes pseudorandom choices. A held-out result and a training result answer different questions.

## 1. What changes during post-training?

### Three different choices: data, objective and parameters

Pretraining learns statistical structure from a broad dataset. Post-training changes a pretrained model toward a desired behaviour using more targeted data and objectives. Supervised fine-tuning (SFT) supplies examples of desired completions: given a prompt, increase the probability of the demonstrated answer. Preference learning instead supplies comparisons: for this prompt, one completion was judged preferable to another. A comparison gives a relative signal, not an absolute quality score or proof of truth.

**Worked SFT example.** For prompt “Name a colour” and demonstrated completion “light blue”, suppose the successive observed completion tokens have probabilities 0.4 and 0.5. Completion NLL is $`-\log0.4-\log0.5=-\log0.2\approx1.6094`$. The prompt supplies context but is not a target in this completion-only loss. A preference example would instead compare two complete responses to that same prompt, rather than declare each demonstrated token the desired target.

Parameter-efficient adaptation answers a separate question: **which model parameters may change?** **Low-rank adaptation (LoRA)** can be used with SFT or a preference objective. Matrix **rank** is the number of independent directions its columns span. Instead of learning every entry of a large update, LoRA learns two smaller factors whose product has rank at most a chosen value. It changes the allowed update to a weight matrix, not the meaning of the training labels. For a square 1,000-by-1,000 weight matrix $`W`$, choose $`B`$ with shape `(1000, 8)` and $`A`$ with shape `(8, 1000)`. Their product $`\Delta W=BA`$ has the required `(1000, 1000)` shape, using $`1000(8)+8(1000)=16{,}000`$ trainable entries instead of one million. This reduces trainable parameters for that matrix. **Frozen** means held fixed during updates; the frozen million-entry base matrix is still needed during computation.

SFT fits demonstrations with conditional token cross-entropy. Prompt tokens and padding should be excluded when the objective is completion likelihood. LoRA uses $`\Delta W=BA`$ with low rank $`r`$, keeping the base weights frozen; trainable parameter savings do not remove base-model memory or all activations. **Quantization** represents numbers with a smaller set of permitted values, usually using fewer bits. Rounding a weight 0.83 to 0.8 illustrates the introduced error, although practical quantizers use more elaborate scales and codes. **QLoRA** combines a quantized frozen base model with trainable low-rank adapters and associated memory/numerical choices. Quantization and low-rank restriction can affect quality; rank and bit width require controlled comparisons.

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

**Work through the constrained optimization.** Hold a prompt fixed and abbreviate candidate probabilities by $`\pi_i`$, reference probabilities by $`q_i`$ and rewards by $`r_i`$. A Lagrange multiplier adds a normalization constraint to an objective before differentiation. Add a multiplier $`\lambda`$ enforcing that probabilities sum to one:

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

The last factor is shared by all responses. Requiring their sum to be one makes it $`1/Z`$, where the **partition function** is simply the sum of the unnormalized weights: $`Z=\sum_iq_i\exp(r_i/\beta)`$. A concave function evaluated at an average is at least as large as the corresponding average of its values. The reward term is linear and negative KL is concave, so this stationary point is the optimum in a finite candidate space with positive reference probabilities. For an infinite response space the normalizing sum/integral must also be finite.

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

The derivation assumes reference support for candidate responses and a preference model; inconsistent labels and distribution shift can violate the modelling assumptions. Completion log probabilities are sums of token log probabilities. Length normalization changes the objective. Freeze the reference model; changing it while fitting changes the target. DPO learns from preference pairs without explicitly fitting a separate reward model. This does not establish factuality, fairness, or universal alignment.

The lab fits a single-context, three-action categorical policy. This makes the objective and gradients inspectable, but it is not language-model fine-tuning and has no held-out contexts. At initialization $`\pi_\theta=\pi_{\mathrm{ref}}`$, the margin is $`0`$ and the loss is $`\log 2`$. A negative gradient on the chosen log probability increases the margin under gradient descent. Distinguish the KL coefficient in the reward objective from the observed KL after optimization.

### Worked example: interpreting the DPO margin

Define $`m`$ as the policy's chosen-minus-rejected log-probability difference **minus the same difference under the reference**. The pair loss is $`-\log\sigma(\beta m)`$. A positive margin means the policy has increased the chosen completion's relative odds compared with the reference. It does not require the chosen completion to have the largest absolute probability among all outputs.

To see the update direction, rewrite the pair loss as $`L(m)=\log(1+\exp(-\beta m))`$. Its derivative is $`L'(m)=-\beta/(1+\exp(\beta m))`$, which is negative for finite $`m`$ and positive $`\beta`$. Decreasing this loss therefore favours a larger adjusted margin.

At $`m=0`$, the loss is $`\log 2\approx0.6931`$, and its derivative with respect to the margin is $`-\beta/2`$. Gradient descent therefore pushes the margin upward. For a uniform three-outcome reference and a policy with probabilities $`(0.5,0.25,0.25)`$, preferring outcome zero over outcome one gives margin $`\log 2`$. At $`\beta=1`$, the implied pair preference probability is two thirds and the loss is $`\log(3/2)\approx0.4055`$.

This shows what has improved: the model fits the given relative preference more strongly.

**Worked one-update check, including the unpaired outcome.** Start with three logits `(0, 0, 0)`, hence probabilities `(1/3, 1/3, 1/3)`, and a uniform reference. Prefer outcome 0 to outcome 1 with $`\beta=1`$. In their log-probability difference, the shared softmax normalizer cancels, leaving margin equal to logit 0 minus logit 1. At initialization the loss gradients for the three logits are `(-0.5, +0.5, 0)`. A gradient-descent step of size 0.5 gives logits `(0.25, −0.25, 0)` and probabilities approximately `(0.4192, 0.2543, 0.3265)`. The unpaired logit did not change, but its probability did because the normalizing sum changed. The margin is now 0.5 and the pair loss approximately 0.4741, down from 0.6931. This is one specified update, not the output of the 100-step lab.

It does not demonstrate correctness, broader usefulness or generalization. Because probabilities normalize together, changes to the paired logits also affect the unpaired outcome. Repeating one preference cannot tell us how the policy should behave on new prompts or conflicting labels.

### Discuss what the derivation guarantees

**Paper question:** why can the two normalizers cancel, and does the resulting preference loss measure truth?

**Worked response:** both responses belong to the same prompt, so their prompt-dependent $`\beta\log Z(x)`$ terms are identical. Responses to different prompts would not share that normalizer. The derivation uses a supported reference distribution and a statistical model of preference labels. Better pair fitting establishes agreement with those labels; truth, fairness and generalization to other prompts need separate measurements.

**Concrete counterexample:** the one-step calculation changes the unpaired outcome’s probability from 1/3 to about 0.3265 even though its logit gradient is zero. Normalization couples probabilities. A preference on one pair does not specify a desirable probability for every other outcome.

## 2. Evaluate the claim and the complete system

### Decide which claim a measurement is meant to support

“Better generation” could mean better density estimation, more realistic individual samples, better coverage, greater task success or lower cost. These are different properties. Choose a primary claim before selecting a metric; otherwise it is easy to report whichever number improved after an experiment.

A loss measures agreement with a training objective. A held-out version asks whether that agreement transfers to new observations. A sample metric asks about outputs from the generation procedure, which includes numerical solvers, decoding or guidance. For example, a VAE's good reconstruction does not establish good prior samples, and a flow's small velocity regression error does not by itself establish coverage of all modes.

On the eight-cluster data, compare two imagined generators: one places every point close to a single real cluster; the other reaches all eight but spreads points too widely. The first has a coverage problem and the second a fidelity problem. A plot may expose both, while a scalar metric compresses them into one number and may obscure which failure occurred.

### Reasoning, attempts and answer selection

A correct final answer is one observable outcome. Robustness asks whether the result survives changes that preserve the problem, such as renaming entities or reordering irrelevant information. Process validity asks whether the stated intermediate steps support the answer. These measurements can disagree: a system may guess correctly, present an invalid derivation, or fail after a harmless wording change. A useful research question names which property is claimed and how it is measured.

A **verifier** is a procedure that checks a proposed answer against a specified criterion. Outcome-based training can use a verifier to score completed attempts. A code task might run tests; a mathematical task might compare a canonical final answer. This reduces reliance on manual preference labels for those particular outcomes, but transfers responsibility to the verifier. Passing incomplete tests is evidence about those tests, not all intended program behaviour. Intermediate reasoning can remain wrong even when the final answer passes.

**Inference** here means using an already trained model to generate or score an answer. **Pass@k** asks whether at least one of $`k`$ attempts is correct; it differs from the accuracy of one answer selected for a user. A **verifier** is the procedure used to recognize correctness. A **chain of thought** is a sequence of intermediate reasoning statements generated as text; its appearance alone does not establish that it describes the internal cause of the answer.

Inference-time computation includes sampling multiple candidates, verification and search. Compare both success and total cost: tokens, latency, tool calls and verifier cost. A $`\mathrm{pass}@k`$ improvement can simply purchase more attempts. Distinguish final-answer correctness, robustness under meaning-preserving perturbations and process validity. Written chains of thought need not faithfully expose a model's internal causal reasoning.

### Worked example: more attempts versus a better attempt

Suppose a fixed model succeeds with probability 0.2 on each independent attempt for one task. The probability of at least one success in five attempts is $`1-(1-0.2)^5\approx0.6723`$. The model's per-attempt probability has not improved. The increased success rate was purchased with more attempts. Real attempts may be correlated, and task difficulty varies, so this calculation is an illustration rather than a formula for every aggregate benchmark.

Moreover, “one of the attempts was correct” assumes a way to recognize that success. A deployable system must select an output using an available checker or selection rule. An oracle pass@k measurement can be much better than the accuracy of the answer a user actually receives. Report attempts, token usage, selection procedure and verifier cost alongside success.

### Retrieval supplies evidence at answer time

A model's parameters store learned statistical regularities; they are not an editable database of guaranteed facts. Retrieval augments a request with external material selected at answer time. A typical pipeline turns a query into a search, ranks candidate passages, supplies selected passages to the generator and asks for an answer grounded in them. Updating the source collection can update available evidence without retraining all model weights, but only if the relevant material is actually retrieved and used correctly.

There are at least two distinct failures: the retriever does not supply the needed evidence, or the generator misuses evidence that is present. A fluent wrong answer alone does not identify which component failed.

A **retrieval-augmented generation (RAG)** system combines this search with answer generation. **Retrieval recall** measures how much of the required relevant evidence is retrieved; if each question has one sufficient passage, it is the fraction of questions for which that passage is found. **Citation support** asks whether a cited passage actually justifies a generated claim. An **oracle-evidence baseline** manually supplies sufficient evidence to test the generator with retrieval failure removed; a **no-retrieval baseline** supplies no external evidence. “Oracle” describes an evaluation intervention, not a deployable component.

A retrieval-augmented system retrieves evidence and conditions generation on it. Evaluate retrieval recall separately from answer correctness and citation support, then use oracle-evidence and no-retrieval baselines to localize failure. Split by source or time when near-duplicate chunks could leak answers. Retrieval may return irrelevant, outdated or adversarial text; a citation's existence is not proof that it supports the generated claim.

### Worked diagnosis: isolate retrieval from evidence use

Imagine a public handbook contains the correct deadline. Under no retrieval, a system answers 40 of 100 questions correctly; with ordinary retrieval, 55; with a manually supplied sufficient passage, 85. These hypothetical outcomes suggest room for improving retrieval or evidence selection, while the remaining 15 errors show that supplying evidence does not guarantee correct use. The 30-point gap is a diagnostic under this intervention, not a universal decomposition of causal effects.

**Worked measurement for the lesson task.** In a separate hypothetical inspection of 100 questions, suppose ordinary retrieval supplies the one sufficient passage for 70 questions and the generator answers 56 of those correctly. Evidence recall is $`70/100=70\%`$. Failure to answer correctly despite sufficient evidence is $`(70-56)/70=20\%`$ within that subset. The other 30 questions have missing evidence, but their answer accuracy is not given. These denominators distinguish the two failure types; the earlier aggregate scores alone cannot recover them.

To make the comparison interpretable, use the same questions, generator and decoding budget, and control passage length and formatting as far as practical. Check whether the ordinary retriever actually included the necessary passage. Add an irrelevant but plausible distractor to otherwise sufficient evidence: a performance drop tests susceptibility to distraction rather than missing knowledge. Check citation support directly; a real document can be cited for a claim it never makes.

### Tools change what a model can do

A tool-using system can request an operation, receive an observation and choose a later action from the updated context. For example, a calculator can resolve arithmetic that a text-only response gets wrong. The complete system includes the model, tool specification, execution environment, observation handling and stopping rule. An action's syntactic validity does not establish that it was useful or that the overall task was completed.

Tool-using agents add actions and observations to generation. Evaluate task completion, intervention rate, latency, cost and harmful unintended actions. Retrieved text and tool outputs are data, not authority to override instructions. In research exercises use a local simulated tool environment without credentials or external write access.

Consider a simulated agent asked to locate a record and update one field.

For a concrete case, record `17` initially contains `{"status": "pending", "owner": "Ana"}`. The request is to set record 17's status to `approved`. A successful final state is `{"status": "approved", "owner": "Ana"}` for record 17, with all other records unchanged, followed by an accurate report of that change. Finding record 17 but leaving `pending`, modifying record 18, or overwriting `owner` fails the task. This is a worked evaluation example; it does not ask you to modify an external system.

Successful search is an intermediate result. Full success requires finding the intended record, making the correct change, preserving unrelated fields and reporting the result accurately. An evaluation that counts only successful tool calls can reward a system that never achieves the requested final state. Evaluate both intermediate failures and the end condition, including unnecessary actions and cost.

Retrieved passages and tool responses may contain text that looks like instructions. Such text is evidence from an external source, not authority to change the task. This distinction is also experimentally testable: insert a distractor instruction into a simulated retrieved document and measure whether the system follows the original request.

**Provenance** records where data came from and how they were collected or transformed. **Memorization risk** concerns reproducing training content rather than learning transferable structure. An **affected group** is a population whose outcomes could change under the system's use. **Release constraints** identify which data, code or model artefacts may be shared. For example, a retrieval study using a private handbook may report aggregate accuracy without permission to release the handbook itself; synthetic toy data avoid that particular restriction but do not establish safety on private documents.

A **hypothesis** is a testable statement about what will happen under specified conditions. It is **falsifiable** when some possible outcome would count against it. For example, “Heun reduces mean sample distance relative to Euler at 50 evaluations on these models” specifies a metric, comparison and budget; observing no reduction counts against the directional claim. The complete worked protocol below turns this statement into a design with a baseline, data split, uncertainty calculation and possible objection.

Every research proposal should describe dataset provenance and permissions, privacy and memorization risks, affected groups, evaluation limits, and release constraints. Avoid uploading personal or restricted data to model services. These are research-design considerations, not claims of legal compliance. The optional toy experiments use synthetic data and need no external model service.

These design choices affect scientific validity as well as responsible use. If near-duplicate documents occur in training and test sets, the apparent generalization result may be retrieval or memorization. If only convenient user groups are represented, the measured accuracy may not support the intended deployment claim. Record the population and conditions to which the conclusion applies.

### Discuss the denominators and end condition

| Question | Worked response |
|---|---|
| Five independent attempts raise success from 0.2 to about 0.6723. Did per-attempt ability improve? | No. The extra success purchases more attempts and still needs a rule to select a correct answer. Real attempts can be correlated. |
| Sufficient evidence is found for 70/100 questions; 56 of those are answered correctly. What failed? | Evidence recall is 70%. Within the sufficient-evidence subset, answer failure is 14/70 = 20%. Missing evidence and misuse of available evidence have different denominators. |
| A tool call finds record 17. Is the task complete? | Only if the requested status change occurs on record 17, unrelated fields/records stay unchanged, and the system reports accurately. A successful intermediate call is not the final task condition. |

Write one additional control for one row. For example, give the same generator sufficient evidence plus a plausible irrelevant passage: a decline tests distraction under supplied evidence rather than failure to retrieve it.

## 3. Critique a claim without inventing evidence

**A claim to examine:** “Method B is a better reasoning model because it solves 75 of 100 problems, while A solves 60.” Assume A uses one attempt per problem and B uses five attempts plus a checker. These numbers are hypothetical and fully specified for discussion; they are not attributed to a paper.

| Question for the group | Worked critique |
|---|---|
| What is observed? | B’s complete five-attempt system has higher measured success on these problems. |
| What remains unidentified? | Whether its underlying model has higher per-attempt success under equal conditions. |
| What control distinguishes the explanations? | Give A the same five-attempt budget and checker on the same problems, with matched token and scoring budgets. |
| What result would change the interpretation? | If A then reaches 74, most of the original gap can be explained by the changed procedure in this example. If it remains at 60, that explanation is less persuasive; other differences still require control. |
| What is the replication unit? | The mathematical problem, keeping its variants and paired model outcomes together. Correlated variants are not independent problems. |
| What should be reported besides success? | Attempts, tokens, checker quality/cost, refusals, attempted wrong answers, uncertainty and the answer-selection rule. |

**Apply the same reasoning to DPO:** imagine pair loss decreases on repeatedly labelled training prompts. The supported statement is improved fit to those comparisons. Propose evaluating on separate prompts and separately scoring the property of interest, such as task correctness, rather than naming the loss decrease “alignment”. If using a real paper, mark a baseline or dataset detail as “not yet checked” until you locate it; the mathematical derivation is not a substitute for its experimental evidence.

Record five notes: claim, available evidence, baseline, possible confound and counter-test. The table above supplies a complete worked model. Agreement with the authors is not the goal; an argument tied to inspected evidence is.

## 4. Turn one question into a research proposal

### Paired comparisons and the unit of uncertainty

An **experimental unit** is the unit assigned a condition or independently replicated for the claim at hand. A **baseline** is the reference procedure being compared; a **confound** is another changing factor that could explain a difference. A seed fixes pseudorandom choices for a repeatable run. Using distinct seeds probes run-to-run variability, while matching a seed across methods helps construct a paired comparison.

Pairing asks both methods the same questions under matched conditions. Suppose the distance differences “50-step minus 5-step” for three independent training seeds are −0.04, −0.01 and +0.02. Their mean is −0.01: a small average improvement, with one reversal. Reporting only the best seed would hide that variability. These are hypothetical values to illustrate interpretation, not repository benchmark results.

A **bootstrap** approximates the variability of a statistic by repeatedly sampling the observed independent units **with replacement** and recalculating the statistic. “With replacement” means a unit can be selected more than once. For paired methods, resample whole pairs, equivalently their differences, so the matching is preserved. An uncertainty interval summarizes the spread of that estimated statistic; it does not certify correctness of the model or of the independence assumptions.

**Worked resampling step.** For the three illustrative differences `(-0.04, -0.01, +0.02)`, one bootstrap draw of unit indices `(1, 1, 3)` gives `(-0.04, -0.04, +0.02)`, whose mean is −0.02. Another draw `(2, 3, 3)` gives mean +0.01. Repeating many draws forms a bootstrap distribution; for a percentile interval one sorts those means and takes selected lower and upper percentiles. For example, the 2.5th percentile has approximately 2.5% of the bootstrap means below it; pairing it with the 97.5th percentile retains the middle 95% of the bootstrap distribution. This interval is an approximation under the sampling assumptions, not a guarantee of 95% coverage in every setting. Two draws are not enough to estimate an interval, and three original units give very weak uncertainty evidence. Resampling many times cannot manufacture independent information absent from the original experiment.

If the claim concerns sensitivity to training randomness, the seed-level run is the unit of replication. Generating 10,000 samples from one trained model does not create 10,000 independent training runs. If the claim concerns performance on new users, independent users—not each correlated response from the same user—are the relevant units. Bootstrap intervals must resample at the level that matches the claim; pairing should be retained when the comparison is paired.

For paired evaluation examples, bootstrap the paired difference of scores. The independent unit must match the claim: resample users/documents/problems when responses within them are correlated. Seeds capture optimization variability, not population uncertainty; three seeds are an exploratory illustration, not strong inferential evidence. Record data hashes, splits, model revision, all settings, compute and unsuccessful runs. Choose hyperparameters on validation data and reserve a final test set for the final claim.

An ablation deliberately removes or changes one component to test an explanation. For a solver experiment, fixing model weights and initial noise while changing integration steps tests the contribution of numerical resolution. Retraining one configuration for longer simultaneously would confound that explanation. If one method uses two network evaluations per step, equal step counts also fail to equalize computation.

**Worked interpretation of the three-seed comparison.** Suppose the matching checks pass and the differences are those given above. A valid conclusion is: “With training and evaluation randomness paired within each seed, changing 5 to 50 Euler steps reduced the average sample-distance estimate by 0.01 across these three runs, with one seed worsening by 0.02.” It would be unsupported to say every run improved or sampling became ten times faster. More independent runs would probe stability of the effect; sampler-only timing would address speed. This is a hypothetical worked answer, not a substitute for recording the lab's actual measurements.

### Worked proposal: compare solvers at matched cost

A research protocol states what you will compare, what evidence will be collected, and what result would challenge the hypothesis. The following fills every field in the discussion record below. It is a proposed experiment; its outcomes have not been measured. Heun predicts an endpoint with one velocity evaluation, evaluates velocity there, and advances using the average of the starting and predicted-endpoint velocities. It uses two network calls per step; Euler uses one.

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

### Your group’s discussion record

Choose one concrete question: fixed-compute flow sampling, preference fitting with noisy labels, robustness to meaning-preserving wording changes, or retrieval with distracting evidence. Fill the record below together. Designing the test is the activity; running a larger study is optional.

| Item | Your group’s proposal |
|---|---|
| Falsifiable hypothesis and a contradicting result | |
| Primary outcome and why it measures the claim | |
| Independent experimental unit | |
| Data provenance and train/validation/final-test separation | |
| Baseline, controlled budget and one ablation | |
| Replication count and uncertainty method | |
| One data/ethical issue and a practical limitation | |
| One objection and a measurement that would address it | |
| Supported conclusion, unsupported conclusion and next measurement | |

Record one claim the group accepts, the assumption it needs, one unresolved objection, and the evidence that would resolve that objection. A disagreement is a useful outcome if its reason is explicit. The worked answers illustrate a defensible argument; they are not a required consensus.

## Optional demonstration — inspect preference updates

**How to run an optional Python example:** use the installed repository environment and stay in the repository root. Create `runs/` if needed, save the whole block as `runs/reading_group_example.py`, then run `python runs/reading_group_example.py` in the terminal. Replace the file contents for the next example. Every block supplies its own imports and inputs. Blocks marked `bash` are terminal commands. Installation is needed only for these optional computations; all fixed worked answers and discussion cases are already on this page.

**One update before a full run.** Use one context and three outcomes, with fixed uniform reference and a preference for outcome 0 over 1. Outcome 2 is unpaired. A logit is a real-valued score before softmax. At beta 1 and zero logits, expect loss 0.6931 and gradients `(-0.5, +0.5, 0)`. A step of size 0.5 gives logits `(0.25, -0.25, 0)`, probabilities approximately `(0.4192, 0.2543, 0.3265)` and new loss 0.4741. `.backward()` computes gradients; `no_grad` applies the explicit update without building another derivative graph.

```python
import math
import torch
from genai_course.objectives import dpo_loss

logits = torch.zeros(3, requires_grad=True)
reference = torch.full((3,), -math.log(3.0))  # fixed log probabilities
log_probs = logits.log_softmax(dim=-1)
loss = dpo_loss(log_probs[0:1], log_probs[1:2], reference[0:1], reference[1:2], beta=1.0)
loss.backward()
print("Initial loss:", loss.item())
print("Logit gradients:", logits.grad.tolist())
with torch.no_grad():
    logits -= 0.5 * logits.grad

updated = logits.log_softmax(dim=-1)
after = dpo_loss(updated[0:1], updated[1:2], reference[0:1], reference[1:2], beta=1.0)
print("Updated logits:", logits.detach().tolist())
print("Updated probabilities:", logits.detach().softmax(dim=-1).tolist())
print("Loss after update:", after.item())
```

**Optional longer comparison:**

```bash
python -m genai_course.labs dpo --steps 100 --beta 0.1
python -m genai_course.labs dpo --steps 100 --beta 1
```

Each command prints a unique run-folder path. Open that folder’s `metrics.json` in a text editor. JSON stores named fields: `config` records settings, `metrics` records measured quantities, and `seconds` is a separate top-level field. Record the actual path rather than guessing a folder name. Training histories are measured before each update; final held-out diagnostics or final probabilities use the model after training. Timings include training and evaluation, so they are not sampler-only latency. Reuse earlier runs if their settings match.

| Beta | Run folder | `loss_first` / `loss_last` | `final_probabilities` | `kl_policy_reference` |
|---|---|---|---|---|
| 0.1 | | | | |
| 1 | | | | |

Read these fields inside `metrics`. `loss_last` is evaluated before the final update; `final_probabilities` and KL describe the policy after it. Both runs start at loss about 0.6931. The toy problem has no held-out contexts or correctness labels. Finite-step comparisons of beta change optimization dynamics and need not follow the ideal fixed-reward optimum’s trend.

**Worked interpretation:** after the single update above, the pair loss fell and the preferred outcome became more probable; the unpaired probability also changed. This says nothing about factual correctness or another prompt. For the 100-step comparison, report the measured probabilities and KL rather than assuming which beta must produce the larger divergence.

## Optional primary sources

- [Direct Preference Optimization — Rafailov et al., v3](https://arxiv.org/abs/2305.18290v3): follow the connection between the regularized reward optimum and the preference objective. In an experimental result, distinguish the measured task from broader claims about alignment.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — Lewis et al., v4](https://arxiv.org/abs/2005.11401v4): inspect what information retrieval adds and which evidence would separate retrieval quality from generation quality. The handbook example here is an explanatory diagnostic, not a reported result from that paper.

The required group record and worked response are above; no external worksheet or assessment page is needed. Return to the [reading-group home](README.md).
