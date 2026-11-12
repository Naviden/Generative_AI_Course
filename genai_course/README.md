# Core implementations

This folder contains the shared implementation used by all four labs. It is not the order in which to study the course. Follow [the numbered lessons](../lessons/README.md), which point to specific functions as they become relevant.

| File | Purpose | First used |
|---|---|---|
| [objectives.py](objectives.py) | Gaussian KL/NLL, scaled attention, flow path and DPO loss | Lesson 01: VAE objective |
| [models.py](models.py) | VAE, velocity network and Euler sampler | Lesson 01: reparameterization |
| [labs.py](labs.py) | Data, training loops, evaluation and output records | Lesson 01: first VAE run |
| [evaluation.py](evaluation.py) | Sample-distance and uncertainty utilities | Lesson 02: evaluation |

Run through `python -m genai_course.labs ...` from the repository root, not by executing these files individually. The [output guide](../labs/outputs.md) explains what the CLI writes. Mathematical and behavior checks are in [tests](../tests/README.md).
