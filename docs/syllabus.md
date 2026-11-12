# Three-class teaching plan

Preparation is limited to setup and the prerequisite diagnostic (suggested 60–90 minutes). Reading full papers and the optional research project are not prerequisites for completing the in-class assessment.

Students follow [the numbered lesson pages](../lessons/README.md); this document supplies the matching clock times. Lesson 01 maps to Class 1, Lesson 02 to Class 2, and Lesson 03 to Class 3. The added navigation and worksheets organize the activities below without increasing their time allocations.

## Learning outcomes

By the end, participants should be able to:

- **O1:** derive and distinguish likelihood, an ELBO, a denoising surrogate and a flow-matching objective, stating assumptions and reduction conventions;
- **O2:** implement or audit reparameterization and causal attention, and interpret autoregressive token likelihood;
- **O3:** connect diffusion and flow matching to probability transport, and separate model error from numerical sampling error;
- **O4:** derive the DPO objective from a KL-regularized reward objective, and distinguish preference fitting from verified reasoning;
- **O5:** design a reproducible, compute-aware comparison with appropriate splits, baselines, uncertainty and failure analysis;
- **O6:** critically assess a current paper and formulate a falsifiable research question.

## Class 1 — Objectives, latent variables and transformers

| Minutes | Activity | Material / outcome |
|---|---|---|
| 0–15 | Diagnostic discussion; generative vs discriminative modelling | Prerequisites; O1 |
| 15–45 | Likelihood, cross-entropy, KL, latent-variable inference | Theory §§1–2; O1 |
| 45–85 | Guided ELBO derivation and reparameterization | Board exercise; O1–O2 |
| 85–120 | VAE lab: event sums, KL weight, held-out diagnostics | Lab 1; O1–O2 |
| 120–140 | GAN and exact-flow contrasts; mode coverage vs likelihood | Theory §3; O1 |
| 140–180 | Autoregression, teacher forcing, causal scaled attention | Theory §4; O2 |
| 180–210 | Attention lab: masking and numerical stability | Lab 2; O2 |
| 210–230 | Tokenization, scaling, leakage and compute budgets | Theory §4; O5 |
| 230–240 | Individual exit ticket and feedback | Assessment A; O1–O2 |

## Class 2 — Diffusion and flow matching

| Minutes | Activity | Material / outcome |
|---|---|---|
| 0–15 | Retrieval practice: ELBO and attention assumptions | O1–O2 |
| 15–50 | Forward diffusion and noise-prediction objective | Theory §5; O1–O3 |
| 50–85 | Scores, reverse SDE, probability-flow ODE | Guided derivation; O3 |
| 85–120 | Conditional flow matching and marginal velocity | Theory §6; O3 |
| 120–155 | Flow lab: train, sample, inspect held-out metrics | Lab 3; O3–O5 |
| 155–175 | Solver-budget ablation with paired seeds | Lab 3; O3–O5 |
| 175–210 | Evaluation: distribution metrics, uncertainty, sample bias | Theory §7; O5 |
| 210–230 | Latent diffusion, multimodality, diffusion language models | Research comparison; O6 |
| 230–240 | Individual exit ticket and feedback | Assessment B; O1–O3 |

## Class 3 — Post-training and credible research

| Minutes | Activity | Material / outcome |
|---|---|---|
| 0–20 | Pretraining vs SFT; LoRA and quantization tradeoffs | Theory §8; O4 |
| 20–55 | Reward models, KL regularization and DPO derivation | Theory §8; O4 |
| 55–80 | DPO lab: preference, KL and missing generalization | Lab 4; O4–O5 |
| 80–120 | Reasoning: verifiable rewards, inference budgets and failure modes | Theory §9; O4–O6 |
| 120–145 | RAG and agents: component evaluation and trust boundaries | Theory §10; O5 |
| 145–180 | Critique a 2026 reasoning evaluation paper | Reading packet; O6 |
| 180–215 | Group research-design exercise | Assessment C; O5–O6 |
| 215–235 | Short group defenses and peer review | Two-minute defense per group; O5–O6 |
| 235–240 | Individual closing reflection | Assessment C; O6 |

For more than eight groups, pair groups for simultaneous defenses rather than extending the class. The instructor supplies paper excerpts and prompts in class; students need not read an entire paper beforehand.

## Deliberate scope boundaries

This course teaches selected mechanisms deeply enough to support a research discussion. Full transformer training, distributed systems, GPU kernel engineering, image/video/audio foundation-model training, RL implementation, formal privacy guarantees and production agent deployment require additional courses. HMMs, GMMs and naive Bayes are prerequisite refreshers under `archive/`, not extra scheduled classes.
