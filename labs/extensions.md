# Optional worked lab extensions

These experiments extend the [lab workbook](README.md). They are optional; completing the numbered lessons does not depend on them. Use the installed environment from the repository root. Save a whole Python block as `runs/worked_example.py` (create `runs/` if needed), then run `python runs/worked_example.py`. Replace the file contents for the next example. Each block is self-contained. Keep these experiments separate from the CLI's recorded runs.

## Heun — Check a solver before comparing generators

**Question:** does a more accurate integration rule improve sampling when both rules receive the same number of velocity evaluations? A velocity evaluation is one call to the trained network. Euler uses one per step. Heun uses two: predict an endpoint with Euler, evaluate its velocity, then advance using the average of the starting and predicted-endpoint velocities.

For current state $`x`$, time $`t`$ and step size $`h`$, the update is:

```math
k_1=v(x,t),\qquad \widetilde{x}=x+h k_1,\qquad
k_2=v(\widetilde{x},t+h),\qquad
x_{\mathrm{next}}=x+\frac{h}{2}(k_1+k_2).
```

**Worked example with a known answer:** take scalar velocity $`v(x,t)=x`$, initial value 1 and integrate from time 0 to 1. The exact solution at time 1 is $`e\approx2.718282`$. Euler with two steps gives $`(1+1/2)^2=2.25`$. Heun with two steps gives $`(1+1/2+(1/2)^2/2)^2=2.640625`$, but spends four evaluations. At the same four-evaluation budget, Euler takes four steps and gives $`(1+1/4)^4=2.44140625`$. Heun is closer in this example; that is a solver check, not evidence about generated-data quality.

```python
import math
import torch
from genai_course.models import euler_sample

@torch.no_grad()
def heun_sample(velocity, noise, steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    x = noise.clone()
    h = 1.0 / steps
    for i in range(steps):
        t = x.new_full((len(x), 1), i * h)
        k1 = velocity(x, t)
        k2 = velocity(x + h * k1, t + h)
        x = x + 0.5 * h * (k1 + k2)
    return x

initial = torch.ones(1, 1, dtype=torch.float64)
velocity = lambda x, t: x
for name, solver, steps in [("Euler", euler_sample, 2),
                            ("Euler", euler_sample, 4),
                            ("Heun", heun_sample, 2)]:
    result = solver(velocity, initial, steps).item()
    evaluations = steps * (2 if name == "Heun" else 1)
    print(name, "steps", steps, "evaluations", evaluations,
          "result", result, "absolute error", abs(result - math.e))
print("Original input:", initial.item())
```

Expect results 2.25, 2.44140625 and 2.640625, and unchanged input 1. The decorator `@torch.no_grad()` disables gradient recording for sampling. `new_full` constructs a time column with the state's numeric type and device. The second velocity uses the *predicted* endpoint; using the starting state twice would not implement Heun.

### Compare one learned field at equal evaluation cost

The standard CLI supports Euler only and saves samples, not model weights. Its `arrays.npz` cannot reconstruct a trained velocity network. The following independent experiment trains once and keeps the model in memory, then supplies identical initial noise to Euler50 and Heun25. It uses the same training objective and data distribution as Lab 3. The training loop selects a batch, constructs a noisy intermediate point, predicts its target velocity, sums squared coordinate errors, averages examples, and updates parameters. Repeating this loop 300 times trains the field before either sampler runs.

```python
import torch
from genai_course.labs import mixture
from genai_course.models import Velocity, euler_sample
from genai_course.objectives import flow_path
from genai_course.evaluation import sliced_wasserstein

@torch.no_grad()
def heun_sample(velocity, noise, steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    x = noise.clone()
    h = 1.0 / steps
    for i in range(steps):
        t = x.new_full((len(x), 1), i * h)
        k1 = velocity(x, t)
        k2 = velocity(x + h * k1, t + h)
        x = x + 0.5 * h * (k1 + k2)
    return x

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
train = mixture(1024, 0)
heldout = mixture(256, 10000)
model = Velocity()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
for _ in range(300):
    batch = train[torch.randint(len(train), (64,))]
    time = torch.rand(64, 1)
    noise = torch.randn_like(batch)
    point, target = flow_path(noise, batch, time)
    loss = (model(point, time) - target).square().sum(-1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
torch.manual_seed(20000)
initial = torch.randn(256, 2)
euler = euler_sample(model, initial, 50)
heun = heun_sample(model, initial, 25)
# Default metric settings use the same 64 projection directions for both.
d_euler = sliced_wasserstein(euler, heldout)
d_heun = sliced_wasserstein(heun, heldout)
print("Velocity evaluations (Euler, Heun):", 50, 2 * 25)
print("Sample shapes:", tuple(euler.shape), tuple(heun.shape))
print("Sliced W1 (Euler, Heun):", d_euler, d_heun)
print("Heun minus Euler:", d_heun - d_euler)
```

Expect equal evaluation counts `(50, 50)` and sample shapes `(256, 2)` for both. Record the measured distances; there is no required winner. This experiment chooses its own evaluation-noise draw, so its Euler distance need not equal a CLI run's distance even with seed 0.

**Worked interpretation with illustrative values:** distances 0.24 for Euler and 0.23 for Heun give a paired difference of −0.01. That favours Heun under this diagnostic for this one trained field and sample draw. It neither measures wall-clock speed nor establishes a reliable advantage across training seeds. A positive difference instead favours Euler in this comparison. Equal network-call counts do not guarantee equal latency.

You are done when you have checked the known solution, recorded the equal-budget result, and separated numerical integration accuracy from distributional sample quality. To design a larger experiment, use the complete [worked research protocol](../lessons/checkpoints.md#9-a-research-protocol-that-can-fail). Its uncertainty analysis needs independent trained models, not repeated measurements of this single model.

## Conflicting preferences — Fit a known label probability

**Question:** what happens if the same pair receives conflicting labels? Lab 4 repeats one preference direction. Here, isolate disagreement using **two outcomes**, numbered 0 and 1, with a fixed uniform reference. Imagine 100 labels for the same context and pair: 75 prefer outcome 0, and 25 prefer outcome 1. Their exact average loss is a weighted combination of the two directions; random label sampling is unnecessary for this example.

Define $`m=\log\pi(0)-\log\pi(1)`$. The reference log ratio is zero, so the model's predicted probability of the label “0 is preferred” is $`s=\sigma(\beta m)`$. This *preference-label probability* is distinct from the policy's outcome probability. The average loss is:

```math
J(m)=-0.75\log\sigma(\beta m)-0.25\log\sigma(-\beta m).
```

The derivative is $`\beta[\sigma(\beta m)-0.75]`$. It vanishes when the predicted label probability equals the observed fraction, so $`m^*=\log(3)/\beta`$. At beta 1, the optimum policy is `(0.75, 0.25)` and loss is $`-0.75\log0.75-0.25\log0.25\approx0.562335`$. At beta 0.5 the same label probability requires margin $`\log9`$, so the policy is `(0.9, 0.1)`. In both cases the fitted preference-label probability is 0.75. These absolute policy probabilities follow because this example has exactly two outcomes; the three-outcome lab includes an unpaired outcome and does not fix its mass from this ratio alone.

```python
import math
import torch
from genai_course.objectives import dpo_loss

for beta in (1.0, 0.5):
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    reference = torch.full((2,), -math.log(2.0), dtype=torch.float64)
    optimizer = torch.optim.SGD([logits], lr=0.5)
    for _ in range(1000):
        logp = logits.log_softmax(dim=-1)
        forward = dpo_loss(logp[0:1], logp[1:2], reference[0:1], reference[1:2], beta)
        reverse = dpo_loss(logp[1:2], logp[0:1], reference[1:2], reference[0:1], beta)
        loss = 0.75 * forward + 0.25 * reverse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logp = logits.log_softmax(dim=-1)
        margin = logp[0] - logp[1]
        label_probability = torch.sigmoid(beta * margin)
        final_loss = -0.75 * label_probability.log() - 0.25 * (1 - label_probability).log()
        print("beta:", beta, "policy:", logp.exp().tolist())
        print("margin:", margin.item(), "label probability:", label_probability.item(),
              "final loss:", final_loss.item())
```

Expect policy probabilities close to `(0.75, 0.25)` and `(0.9, 0.1)` respectively, margins approximately 1.098612 and 2.197225, label probability 0.75 and loss 0.562335. `forward` and `reverse` swap the preferred and rejected outcome while keeping the reference fixed. This optimizes the empirical 75:25 label distribution directly.

**Interpretation:** a positive minimum loss is appropriate when labels disagree. Sending the margin to positive infinity would make the 25 reverse labels arbitrarily costly. Fitting this label distribution does not show that either outcome is correct or that these annotators represent another population. You are done when you can distinguish outcome probability from preference-label probability and explain why zero training loss would be the wrong target here. Return to [Lab 4](README.md#lab-4--understand-a-preference-update) or your current lesson.
