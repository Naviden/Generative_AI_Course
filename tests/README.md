# Implementation checks

Run `python -m pytest` from the repository root after setup. The tests check objective values, gradients, causal masking, sampler behavior, reproducibility and run metadata. They do not establish that a short experiment has research-level sample quality.

Follow [the lessons](../lessons/README.md) first. Lesson 01 points to the attention perturbation test as an exercise in understanding what a check can establish. See [test_objectives.py](test_objectives.py) for mathematical/gradient checks and [test_labs.py](test_labs.py) for end-to-end experiments.
