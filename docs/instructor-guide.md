# Delivery notes

Use [the numbered lessons](../lessons/README.md) as the student-facing entrance. After setup, send students to Lesson 01, Step 1. Keep the compact lab workbook and theory notes as linked references; do not ask students to browse every topic folder. The lesson checkpoints and learning log support the existing timed activities rather than adding homework.


## Before teaching

Confirm students have graduate ML foundations. Run the locked environment checks and all four labs on the actual classroom machines before class. Open the pinned paper versions in the reading list and review the experiment/table selected for critique. Have the validation table and code available offline. The repository supplies Markdown notes and a workbook, not a slide deck; these can be projected directly or used as board prompts.

Keep the derivation, lab and interpretation together. If discussion overruns, trim optional frontier examples rather than the exit ticket. No full-paper homework is needed; bring the selected excerpt/figure or show its primary-source page during the scheduled critique.

## Suggested board sequence

Class 1: write likelihood and KL; derive the ELBO in three steps (insert $`q`$, Jensen, posterior gap); compute Gaussian KL for $`\mu=(1,2)`$; trace reparameterization; contrast GAN/flow density access; draw the causal attention matrix and a single next-token prediction.

Class 2: compose two Gaussian noising transitions; derive the conditional score; mark reverse-time direction; compare SDE and ODE drifts; derive the straight path and velocity; interpret conditional expectation; show Euler updates; ask what a finite-sample metric can establish.

Class 3: distinguish pretraining, SFT and preference data; optimize a KL-regularized reward with a normalization multiplier; cancel log Z in preference differences; run the categorical DPO example; separate preference, factuality and verified outcomes; critique a contemporary benchmark; run the protocol defense.

## Expected reasoning and common mistakes

- ELBO equality requires the exact posterior, not merely a large encoder.
- A beta-VAE with beta different from one changes the objective.
- VAE decoder means alone omit likelihood noise; sampling a Gaussian observation model includes it.
- The noise predictor estimates a conditional mean given noisy x, not the particular noise used for every possible pair.
- Independent straight conditional paths do not imply optimal-transport coupling.
- Lower numerical error does not automatically mean lower model error.
- Causal masking does not provide positional information or a complete language model.
- DPO log probabilities are completion sums; the reference is fixed.
- A toy DPO fit has no evidence of held-out language generalization.
- Bootstrap intervals need an appropriate independent sampling unit; repeated generations from one problem are not independent problems.

## Scoring anchors

For the first exit ticket, expect $`\log p_\theta(x)=\mathcal{L}(x)+D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))`$, the nonnegative gap, the sum over latent dimensions and mean over batch. For the second, expect $`\bar\alpha_t`$, the noise/score relation and noise-to-data flow direction. In research defenses, reward a clear falsification criterion and fair compute matching more than a fashionable model name.

For R18, acceptable critiques include semantic changes caused by perturbations, refusal handling and correlated benchmark items. For R19, acceptable critiques include the validity of the process scoring rule and the separation between a readable trace and internal reasoning. Students may defend the design if they support their argument with evidence.

## Classroom fallback

If installation fails, pairs inspect the objective and recorded outputs, predict the result of a controlled change and explain the necessary test. Do not spend class time installing optional frameworks. Legacy notebooks and full pretrained-model downloads are not required activities.
