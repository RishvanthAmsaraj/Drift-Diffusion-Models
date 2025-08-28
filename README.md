
# PyDDM Overview

## What is PyDDM

PyDDM (Python Drift Diffusion Model) is a Python library for building, simulating, and fitting drift-diffusion models (DDMs).
A DDM is a type of sequential sampling model used in cognitive science, neuroscience, and decision-making research to model how agents make binary decisions over time. It assumes that noisy evidence accumulates until a decision threshold is reached.

---

## How It Works

1. **Model Parameters**

   * Drift rate (v): speed of evidence accumulation
   * Boundary separation (a): amount of evidence required to make a decision
   * Starting point (z): initial bias toward one choice
   * Non-decision time (t0): perceptual or motor delays not part of the decision process

2. **Simulation**

   * PyDDM generates trajectories of evidence accumulation leading to choices and response times.

3. **Fitting**

   * Models are fit to experimental data (choice and reaction time distributions) using likelihood-based methods.

4. **Extensions**

   * Supports custom models, such as time-varying drift rates, collapsing bounds, or multiple decision boundaries.

---

## Pros

* Flexible and extensible: custom components are easy to define
* Python-native: integrates with NumPy, SciPy, and pandas
* Visualization tools: trajectories, distributions, parameter recovery
* Research-grade: designed for psychology and neuroscience experiments
* Supports complex models beyond the standard DDM

---

## Cons

* Requires background in drift-diffusion theory to use effectively
* Slower than optimized C/C++ implementations on large datasets
* Smaller community and fewer tutorials than mainstream ML libraries
* Specialized for cognitive modeling rather than general-purpose ML

---

## Alternatives

* **HDDM**: Bayesian hierarchical DDM in Python, built on PyMC/Stan. Good for group-level modeling, but slower due to sampling.
* **fast-dm**: C++ implementation with Python interface. Very fast for standard DDMs but less flexible.
* **rtdists (R)**: R package for fitting response time distributions, including DDMs.
* **Stan / PyMC**: General-purpose Bayesian frameworks where you can build custom DDMs from scratch.

---

# Example: Basic Drift Diffusion Model with PyDDM

import pyddm as ddm
from pyddm import Model, Fittable, Sample
from pyddm.models import DriftConstant, BoundConstant, ICPointSourceCenter, NoiseConstant, OverlayNonDecision

# 1. Define the model
model = Model(
    drift=DriftConstant(drift=0.5),
    noise=NoiseConstant(noise=1.0),
    bound=BoundConstant(B=1.0),
    IC=ICPointSourceCenter(),
    overlay=OverlayNonDecision(nondectime=0.3)
)

# 2. Simulate data
sim_data = model.simulate_n_trials(n=500)

# 3. Inspect simulated output
print(sim_data[:10])  # first 10 simulated trials

# 4. Fit model to data (requires empirical dataset)
# Example: fitting to simulated data
sample = Sample.from_numpy_array(sim_data)
fitted_model = model.fit(sample)

print(fitted_model.parameters())
