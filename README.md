
## 🔹 What is PyDDM?

**PyDDM** (Python Drift Diffusion Model) is a Python library used to **build, simulate, and fit drift-diffusion models (DDMs)**.

* A **DDM** is a type of **sequential sampling model** used in cognitive science, neuroscience, and decision-making research to model how people (or agents) make two-choice decisions over time.
* It assumes decisions are made by **accumulating noisy evidence** until a boundary (decision threshold) is reached.

---

## 🔹 How PyDDM Works

1. **Model Setup**

   * Define the parameters of the diffusion process:

     * **Drift rate (v):** how quickly evidence accumulates toward one decision.
     * **Boundary separation (a):** the amount of evidence needed before making a choice.
     * **Starting point (z):** initial bias toward a choice.
     * **Non-decision time (t0):** perceptual/motor delays not part of decision process.

2. **Simulation**

   * PyDDM simulates decision-making trajectories (like noisy paths of evidence accumulation).

3. **Fitting**

   * It fits the model to **reaction time distributions and choice data** using maximum likelihood or Bayesian inference.
   * You can fit experimental data (e.g., psychology experiments) to infer cognitive parameters.

4. **Extensions**

   * PyDDM allows custom models beyond the “vanilla” DDM, such as time-varying drift rates, collapsing boundaries, or more than two decision bounds.

---

## 🔹 Pros of PyDDM

✅ **Flexible & Extensible** – You can customize drift rates, bounds, noise, etc.
✅ **Python-native** – Easy integration with scientific stack (NumPy, SciPy, pandas).
✅ **Visualization Tools** – Can plot trajectories, reaction time distributions, parameter recovery.
✅ **Research-Grade** – Designed for psychology/neuroscience research.
✅ **Supports complex models** – Not limited to classic DDM.

---

## 🔹 Cons of PyDDM

❌ **Learning Curve** – Requires some mathematical/statistical background in DDM theory.
❌ **Performance** – Slower than highly optimized C/C++ implementations for large datasets.
❌ **Limited Community** – Smaller user base compared to mainstream stats/ML libraries.
❌ **Specialized** – Useful mainly in cognitive modeling, not general-purpose ML.

---

## 🔹 Alternatives to PyDDM

1. **HDDM (Hierarchical Drift Diffusion Model in Python)**

   * Bayesian approach (uses PyMC/Stan under the hood).
   * Good for hierarchical modeling (group-level + individual-level).
   * Strong in cognitive modeling.
   * But: slower due to Bayesian sampling.

2. **fast-dm (C++ with Python interface)**

   * Very fast and efficient for fitting classic DDMs.
   * Less flexible for custom models.

3. **rtdists (R package)**

   * For fitting response time distributions, including DDMs.
   * Good if working in R.

4. **Stan / PyMC** (general-purpose Bayesian frameworks)

   * You can code a DDM from scratch if you want full flexibility.

---

## 🔹 Rule of Thumb:

* Use **PyDDM** if you want flexibility and Python integration.
* Use **HDDM** if you want Bayesian hierarchical modeling.
* Use **fast-dm** if you need raw speed on standard DDMs.
