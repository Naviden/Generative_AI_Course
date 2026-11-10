# Reading guide and source ledger

Reviewed 2026-11-10. Dates below identify the original paper/preprint, not the most recent web crawl. The in-class packet uses the specified concepts and short instructor summaries; full-paper reading is optional. This is a selected research bibliography, not an exhaustive claim to cover all publications through the review date.

## Core papers

| ID | Primary source | In-class focus / critical question |
|---|---|---|
| R1 | Kingma & Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), 2013 | Class 1: ELBO and pathwise estimator. When is the bound tight? |
| R2 | Goodfellow et al., [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), 2014 | Class 1 comparison: ideal discriminator vs finite optimization. |
| R3 | Dinh et al., [Density Estimation Using Real NVP](https://arxiv.org/abs/1605.08803), 2016 | Class 1 comparison: tractable Jacobian and architectural constraints. |
| R4 | Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017 | Class 1: scaled attention and masking; distinguish original encoder-decoder from a causal decoder. |
| R5 | Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020 | Class 2: forward marginal and simplified loss; which likelihood weights are omitted? |
| R6 | Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), 2020/2021 | Class 2: reverse SDE and probability-flow ODE; assumptions and numerical errors. |
| R7 | Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), 2022/2023 | Class 2: conditional vs marginal velocity; independent coupling vs OT coupling. |
| R12 | Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023 | Class 3: partition-function cancellation, reference policy and preference assumptions. |

## Research context and optional extensions

| ID | Primary source | Purpose |
|---|---|---|
| R8 | Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022 | Scaling at a stated compute budget; limits of extrapolation. |
| R9 | Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021 | Low-rank updates and parameter/memory accounting. |
| R10 | Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023 | Quantization, memory and quality tradeoffs. |
| R11 | Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155), 2022 | SFT, reward modelling and RLHF; preference distribution matters. |
| R13 | DeepSeek-AI et al., [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), 2025 | Verifiable rewards, post-training and distinction between R1 and R1-Zero. Record the version used if replicating. |
| R14 | Nie et al., [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992v3), 2025, v3 | Masked diffusion vs autoregressive language modelling; fair compute comparisons. |
| R15 | Yuan et al., [Native Sparse Attention](https://arxiv.org/abs/2502.11089v2), 2025, v2 | Hardware-aware sparse attention and quality/throughput tradeoffs. |
| R16 | Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), 2021/2022 | Autoencoder compression and conditional generation. |
| R17 | Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020 | Retrieval and generation as separately evaluable components. |
| R18 | Golikov et al., [Robust Reasoning Benchmark](https://arxiv.org/abs/2604.08571v3), 2026, v3 | Class 3 critique: robustness under textual perturbations; refusal and accuracy must be separated. Treat as a preprint, not independently reproduced course evidence. |
| R19 | Zheng et al., [Unmasking Reasoning Processes: A Process-aware Benchmark for Evaluating Structural Mathematical Reasoning in LLMs](https://arxiv.org/abs/2602.00564v2), 2026, v2 | Alternative class 3 critique: final answers vs process scoring; construct validity and annotation assumptions. Treat as a preprint. |
| R20 | Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 2022/2023 | Tool interaction and task evaluation; observation/action boundaries. |

## Class 3 critique packet (no full-paper homework required)

Choose R18 by default; use R19 as an alternative if the cohort works on process rewards. Give students the primary-source abstract and the relevant experimental setup/table to read in class; avoid redistributing the entire paper unless its license allows it. The instructor should open the pinned version before teaching.

For R18, the authors evaluate deterministic textual perturbations on mathematical benchmarks. Discuss whether the perturbations preserve the mathematical task and how to distinguish a refusal from an incorrect solution. For R19, the proposed evaluation separates answer correctness and structural reasoning; ask what evidence validates the process score itself. These summaries describe author claims; they do not endorse the benchmarks as universal measurements of reasoning.

In pairs: identify the claim, experimental unit, baseline, data-selection procedure and cost budget. Name one plausible confound. Design one experiment that could overturn your initial interpretation. The class need not agree on the paper's conclusion; arguments must connect claims to evidence.

## Updating this list

Before each delivery, inspect newer versions of the selected papers and relevant primary research. Record the search date, exact version and the reason for replacing a reading. Include a new paper because it changes an important concept or evaluation issue, not solely because its release is recent. Never silently present preprint claims as independently verified findings.
