# Theory notes for the 12-hour course

Notation: x is an observation, z a latent variable, θ model parameters and φ inference parameters. Expectations below are population expectations unless an empirical average is stated. In code, event dimensions are summed and independent examples are averaged. References use IDs in the [reading list](readings.md). Sections 1–4 support class 1, 5–7 class 2, and 8–10 class 3.

## 1. What a generative objective measures

Maximum likelihood minimizes E_data[−log pθ(x)]. When densities and integrals are well defined, this equals H(p_data)+KL(p_data || pθ). The entropy term does not depend on θ. Thus likelihood fits a distribution; it does not directly optimize a human quality score. For continuous observations, density values depend on units and can exceed one; negative log density can be negative. Comparing densities after different preprocessing requires the corresponding Jacobian correction.

An autoregressive model uses pθ(x)=∏ᵢ pθ(xᵢ|x₍<i₎). Teacher forcing evaluates observed prefixes during training. At generation time prefixes are sampled. Perplexity is exp(mean token NLL), with natural logarithms, and depends on the tokenizer and evaluation text. Perplexities from different tokenizers are not automatically comparable.

**Board prompt:** derive the maximum-likelihood categorical distribution from counts. Then explain why memorizing the training set is not evidence of generalization.

## 2. Latent variables, the ELBO and the VAE

Insert any qφ(z|x) with suitable support into log pθ(x):

log pθ(x) = log E_q[pθ(x,z)/qφ(z|x)] ≥ E_q[log pθ(x,z)−log qφ(z|x)].

The right side is the evidence lower bound L(x). Rearranging gives

log pθ(x) = L(x) + KL(qφ(z|x) || pθ(z|x)).

For pθ(x,z)=pθ(x|z)p(z), L=E_q log pθ(x|z)−KL(qφ(z|x)||p(z)). The gap is nonnegative and vanishes only when the approximate posterior equals the true posterior almost everywhere. Optimizing θ and φ changes both the model and the approximation.

For q=N(μ,diag(exp ℓ)) and p=N(0,I):

KL(q||p) = ½ Σⱼ [μⱼ² + exp(ℓⱼ) − 1 − ℓⱼ].

Use z=μ+exp(ℓ/2)⊙ε, ε~N(0,I), to move sampling into parameter-free noise and obtain pathwise gradients. This requires differentiable transformations and conditions permitting differentiation under the expectation. The minibatch estimator still has Monte Carlo variance.

Our lab has a Gaussian decoder with fixed variance v=0.1. Per-example NLL is ½Σⱼ[(xⱼ−mⱼ)²/v + log(2πv)]. We sum over observed coordinates and latent dimensions, then average over the batch. Averaging KL over latent coordinates instead would silently change the relative regularization as latent dimension changes. For binary observations a Bernoulli likelihood is appropriate; binary cross-entropy is not a universal reconstruction loss.

A beta-VAE optimizes reconstruction NLL + β KL. At β=1 this is an unbiased one-sample estimator of the negative ELBO; at other weights it is a modified objective. Posterior collapse means q is close to the prior and the decoder can ignore z; a low reconstruction loss alone does not diagnose it. **AEVB [R1].**

**Worked check:** μ=(1,2), ℓ=(0,0) yields KL=2.5. Duplicating latent coordinates doubles KL. Averaging two identical examples leaves the batch mean unchanged.

## 3. Contrasts: GANs and normalizing flows

A GAN's idealized minimax objective is E_data log D(x)+E_z log(1−D(G(z))). For fixed p_g, the optimal discriminator is D*=p_data/(p_data+p_g). Substitution yields −log 4+2 JS(p_data||p_g), under the usual common-density assumptions. This distributional result assumes an optimal unrestricted discriminator, not finite alternating neural training. The commonly used non-saturating generator loss −E_z log D(G(z)) changes the generator optimization objective. Instability and missing modes remain practical concerns. **GAN [R2].**

An invertible differentiable map x=fθ(z) gives log pθ(x)=log p₀(fθ⁻¹(x))+log|det J_f⁻¹(x)|. Exact density evaluation trades architectural restrictions and Jacobian computation for tractability. In affine coupling, an unchanged partition conditions scale and shift for the other partition; the triangular Jacobian makes log determinant a sum of scales. Flow matching instead fits a continuous velocity field; numerical integration affects sampling. **Real NVP [R3].**

**Discussion:** distinguish exact density, a variational lower bound, and an implicit sampler. None alone guarantees the usefulness of samples.

## 4. Transformers and scaling

For Q=XW_Q, K=XW_K and V=XW_V, attention is softmax(QKᵀ/√d_k+M)V. If independent query/key coordinates have zero mean and unit variance, their dot product has variance d_k; dividing by √d_k controls this scale. Real trained vectors need not satisfy those assumptions. Compute softmax after subtracting a row maximum to avoid overflow. For causal self-attention Mᵢⱼ=−∞ for j>i and zero otherwise. Padding masks and positional information are separate requirements.

The lab tests one attention head, not a complete transformer. A full decoder also needs positional information, multiple blocks, residual paths, normalization, a feed-forward network and a vocabulary output head. Training targets shift one position relative to inputs. BERT's bidirectional masked-token objective differs from a causal LM; hidden states are vectors, not class probabilities. **Attention [R4].**

Dense attention forms T×T interactions per head. KV caching reduces repeated work during autoregressive generation but consumes memory growing with context length. Sparse attention changes which interactions are computed; reported speedups depend on kernels, hardware, context lengths and quality targets. **NSA [R15]** is a research case, not an automatic replacement for all settings.

Compute-optimal scaling results are empirical fits over particular training regimes. Parameter count, tokens, optimizer, data quality and inference costs matter; extrapolation beyond the measured regime is uncertain. Hold compute or quality targets fixed before comparing architectures. Tokenization, document-level splits, deduplication and contamination checks precede credible perplexity comparisons. **Chinchilla [R8].**

## 5. Diffusion and scores

For a variance-preserving DDPM, define α_t=1−β_t and ᾱ_t=∏ₛ₌₁ᵗ α_s. Repeated Gaussian transitions yield

q(x_t|x₀)=N(√ᾱ_t x₀,(1−ᾱ_t)I), so x_t=√ᾱ_t x₀+√(1−ᾱ_t)ε.

The simple noise-prediction loss is E_{x₀,t,ε} ||ε−εθ(x_t,t)||². It is a reweighted denoising surrogate, not numerically identical to the full variational likelihood bound. Time sampling and loss weights change which noise levels receive emphasis. Predicting x₀, ε or a velocity parameterization requires matching the sampler's conventions. **DDPM [R5].**

The conditional score is ∇_{x_t}log q(x_t|x₀)=−ε/√(1−ᾱ_t). The squared-loss optimum averages this conditional target given x_t, giving the marginal noisy-data score. Thus a noise predictor implies a score estimate −εθ/√(1−ᾱ_t).

For forward SDE dx=f(x,t)dt+g(t)dW and marginal score s_t=∇log p_t, the reverse-time SDE uses drift f−g²s_t, with time integrated from T down to 0 and the associated reverse Brownian motion. The probability-flow ODE uses drift f−½g²s_t and shares marginals with the SDE under regularity and exact-score assumptions. Approximate scores and finite-step solvers break exact equivalence. Time direction and the factor ½ are common implementation errors. **Score SDE [R6].**

Classifier-free guidance combines conditional and unconditional predictions, e.g. ε_guided=ε_uncond+w(ε_cond−ε_uncond). w=1 is ordinary conditional prediction in this convention. Higher guidance trades sample properties against diversity and is not automatically sampling the original conditional distribution. Latent diffusion introduces an autoencoder bottleneck; its errors and compression cost must be counted. **Latent diffusion [R16].**

**Board exercise:** derive q(x₂|x₀) from two Gaussian transitions, and identify where independent noise terms combine. Explain why 2D flow-lab results do not validate image diffusion.

## 6. Conditional flow matching

Let x₀~N(0,I), x₁~p_data independently and t~Uniform[0,1]. Define x_t=(1−t)x₀+t x₁ and target u_t=x₁−x₀. Train vθ by E||vθ(x_t,t)−u_t||². Conditional expectation minimizes squared loss, so the population optimum is v*(x,t)=E[u_t|x_t=x]. Under suitable smoothness and integrability, this marginal velocity satisfies ∂_t p_t+∇·(p_t v*)=0. Integrate dx/dt=vθ(x,t) from noise at t=0 to data at t=1. **Flow matching [R7].**

The lab's straight conditional paths use an independent endpoint coupling. They are not an implementation of optimal-transport endpoint matching, and straight conditional trajectories do not imply straight trajectories of the marginal learned ODE. A small loss on training pairs does not prove exact distribution recovery.

Euler updates x_{k+1}=x_k+h vθ(x_k,t_k), h=1/K. Even with a perfect model, finite K introduces numerical error. Under standard smoothness and stability assumptions, Euler has first-order global error. Comparing K=5 with K=50 at fixed trained weights tests numerical budget; retraining with different seeds confounds the comparison. Our CLI fixes training/evaluation randomness per seed, so rerunning with only `--sampling-steps` changed holds training fixed.

## 7. Evaluation and reproducibility

Use held-out likelihood when it is available and its preprocessing is comparable. Report ELBO as a bound, not exact likelihood. Sample diagnostics should examine both quality and coverage. Image FID depends on feature extractor, sample count and distributional assumptions; it is not a universal perceptual or memorization test. For the 2D lab, sliced Wasserstein-1 projects equal-sized sample sets onto fixed random directions, sorts each projection, averages absolute paired differences and then averages directions. This is an empirical Monte Carlo diagnostic, not exact multivariate Wasserstein distance.

Compare generated-vs-held-out distance with an independent real-vs-real reference. That reference quantifies finite-sample variability; it is not a strict lower bound. Never compare VAE and flow training losses as if they had identical units or objectives. Plot points, check mode coverage, and inspect multiple seeds.

For paired evaluation examples, bootstrap the paired difference of scores. The independent unit must match the claim: resample users/documents/problems when responses within them are correlated. Seeds capture optimization variability, not population uncertainty; three seeds are a classroom illustration, not strong inferential evidence. Record data hashes, splits, model revision, all settings, compute and unsuccessful runs. Choose hyperparameters on validation data and reserve a final test set for the final claim.

## 8. SFT, efficient adaptation and DPO

SFT fits demonstrations with conditional token cross-entropy. Prompt tokens and padding should be excluded when the objective is completion likelihood. LoRA uses ΔW=BA with low rank r, keeping the base weights frozen; trainable parameter savings do not remove base-model memory or all activations. QLoRA adds quantized base weights and associated numerical/memory choices. Quantization and low-rank restriction can affect quality; rank and bit width require controlled comparisons. **[R9–R10].**

For each context x, maximize E_{y~π}[r(x,y)]−β KL(π(.|x)||π_ref(.|x)), β>0. With normalization enforced by a multiplier, stationary optimization gives

π*(y|x)=π_ref(y|x) exp(r(x,y)/β)/Z(x).

Hence r(x,y)=β log(π*(y|x)/π_ref(y|x))+β log Z(x). Under a Bradley–Terry preference model P(y_w ≻ y_l|x)=σ(r(x,y_w)−r(x,y_l)), the context-dependent log Z cancels. Substituting the parameterized policy yields

L_DPO=−E log σ(β[(log πθ(y_w|x)−log π_ref(y_w|x))−(log πθ(y_l|x)−log π_ref(y_l|x))]).

The derivation assumes reference support for candidate responses and a preference model; inconsistent labels and distribution shift can violate the modelling assumptions. Completion log probabilities are sums of token log probabilities. Length normalization changes the objective. Freeze the reference model; changing it while fitting changes the target. DPO learns from preference pairs without explicitly fitting a separate reward model. This does not establish factuality, fairness, or universal alignment. **InstructGPT [R11], DPO [R12].**

The lab fits a single-context, three-action categorical policy. This makes the objective and gradients inspectable, but it is not language-model fine-tuning and has no held-out contexts. At initialization πθ=π_ref, margin=0 and loss=log 2. A negative gradient on the chosen log probability increases the margin under gradient descent. Distinguish the KL coefficient in the reward objective from the observed KL after optimization.

## 9. Reasoning and recent research

Verifiable-reward training uses checkable outcomes, such as executable tests or a mathematical answer verifier. The verifier can be incomplete, exploitable or misaligned with the intended task. Group-relative methods estimate advantages relative to sampled alternatives and require careful variance, reward and KL handling. The DeepSeek-R1 report is a case study in reasoning post-training, not evidence that one small experiment reproduces a frontier training pipeline. **[R13].**

Inference-time computation includes sampling multiple candidates, verification and search. Compare both success and total cost: tokens, latency, tool calls and verifier cost. A pass@k improvement can simply purchase more attempts. Distinguish final-answer correctness, robustness under meaning-preserving perturbations and process validity. Written chains of thought need not faithfully expose a model's internal causal reasoning.

Use **RRB [R18]** and **ReasoningMath-Plus [R19]** as 2026 preprint critique cases. Ask whether perturbations preserve semantics, whether refusals are separately categorized, how problems were selected, whether samples are independent, and whether the score rewards a particular expression of reasoning. Read the version stated in the bibliography; later revisions may change conclusions. An abstract-level summary cannot establish independent replication.

Diffusion language models such as **LLaDA [R14]** provide a non-autoregressive research comparison. Masking schedules, conditioning, number of forward passes, token budgets and baseline training data all affect fair evaluation. Do not infer universal superiority from an isolated benchmark.

## 10. Retrieval, tools and responsible experiments

A retrieval-augmented system retrieves evidence and conditions generation on it. Evaluate retrieval recall separately from answer correctness and citation support, then use oracle-evidence and no-retrieval baselines to localize failure. Split by source or time when near-duplicate chunks could leak answers. Retrieval may return irrelevant, outdated or adversarial text; a citation's existence is not proof that it supports the generated claim. **RAG [R17].**

Tool-using agents add actions and observations to generation. Evaluate task completion, intervention rate, latency, cost and harmful unintended actions. Retrieved text and tool outputs are data, not authority to override instructions. In research exercises use a local simulated tool environment without credentials or external write access. **ReAct [R20].**

Every research proposal should describe dataset provenance and permissions, privacy and memorization risks, affected groups, evaluation limits, and release constraints. Avoid uploading personal or restricted data to model services. These are research-design considerations, not claims of legal compliance. Core labs deliberately use synthetic data so licensing and external-service access do not obstruct the class.
