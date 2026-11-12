# Assessment

No compulsory multi-week project or extensive homework is hidden inside this short course. The assessments below fit the syllabus. A suggested pass threshold is 60/100, subject to the institution's rules. PhD-level expectations concern reasoning, assumptions and evidence, not the size of the model trained.

## A. Class 1 exit ticket — 25 points, 10 minutes

1. Derive the ELBO and state when equality holds (10).
2. Explain the correct latent-dimension reduction in Gaussian KL and identify a consequence of averaging instead (8).
3. Explain the causal-mask perturbation test and one limitation of what it establishes (7).

Full credit requires a valid derivation with support/Jensen reasoning, correct reduction conventions, and a test that explicitly distinguishes future from past tokens. A formula copied without explanation earns at most half of its points.

## B. Class 2 exit ticket — 25 points, 10 minutes

1. Write $`q(x_t\mid x_0)`$ and relate its noise to a conditional score (10).
2. Specify the flow-matching path, target and sampling direction (8).
3. Explain how the lab isolates numerical sampling budget and name one remaining uncertainty (7).

Full credit distinguishes conditional targets from marginal optima, and surrogate objectives from exact likelihood. Do not require a complete SDE proof in this timed exercise.

## C. Class 3 research design and defense — 40 points

Groups of two to four use the allocated 35-minute design period and 20-minute defense period. Choose one: robustness under text perturbation; fixed-compute flow sampling; preference fitting with noisy labels; retrieval under distracting evidence.

Deliver a one-page protocol with a falsifiable hypothesis, primary outcome, independent experimental unit, train/validation/test separation, baseline, compute budget, an ablation, uncertainty method and one ethical/data issue. Defend it in two minutes; peers challenge one potential confound. The instructor scores:

| Criterion | Points | Full-credit evidence |
|---|---:|---|
| Question and theory | 8 | Specific mechanism and a result that would falsify it |
| Experimental design | 12 | Appropriate baseline, split, controlled budget and ablation |
| Evaluation | 10 | Valid metric, independence unit, uncertainty and failure analysis |
| Reproducibility and data | 6 | Versions, seeds, provenance, resource constraints and release limits |
| Defense | 4 | Responds to criticism without overstating the evidence |

## D. Individual final reflection — 10 points, 5 minutes

State one conclusion your experiment could support (4), one it could not support (4), and the next measurement needed (2). This provides individual evidence alongside group work. Core labs are formative evidence supporting the exit tickets and protocol, not additional out-of-class grading.

## Optional research extension (outside the course)

For students wanting a longer project, budget 10–20 additional hours with supervisor agreement. Replicate a narrowly scoped published result or a controlled ablation. Submit a 4–6 page report, environment snapshot, all configurations, raw results including unsuccessful runs, an executable reproduction command and a statement of deviations from the paper. Use at least three seeds as a minimum exploratory check and choose a larger sample/seed count justified by the claim and resources. Neither novelty nor a favorable result is required; a credible negative result is valuable. This extension is not needed to pass the course.
