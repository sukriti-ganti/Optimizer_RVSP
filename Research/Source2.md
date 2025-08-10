# From Gradient Descent Video → RVSP PBL Mapping

---

## **1. Data and Inputs as Random Variables**

- **Video Point:** Network takes 28×28 grayscale images (MNIST) → 784 input activations.
- **RVSP PBL Link:**
    - Each pixel intensity is a **random variable** due to handwriting variation.
    - Across samples, these form a **stochastic process** feeding into the network.
    - Variability in inputs → variability in gradients → variability in optimizer updates.

---

## **2. Cost Function as a Noisy Signal**

- **Video Point:** Cost function measures error; training reduces it via gradient descent.
- **RVSP PBL Link:**
    - Cost values at each iteration are noisy due to mini-batch sampling.
    - This is a **time series** we can test for **stationarity** (do its stats remain stable?) and **autocorrelation** (are errors related over time?).

---

## **3. Gradients as Stochastic Signals**

- **Video Point:** Gradients point in the direction of steepest cost reduction; computed for thousands of parameters.
- **RVSP PBL Link:**
    - Gradients are **random vectors** due to sampling and parameter initialization.
    - Their evolution can be analyzed via:
        - **Autocorrelation** → check if gradients are related over iterations.
        - **PSD** → detect smooth vs jittery gradient behavior.
        - **Variance** → track stabilization.

---

## **4. Parameter Updates as the Core Stochastic Process**

- **Video Point:** Gradient descent iteratively updates ~13,000 parameters.
- **RVSP PBL Link:**
    - Each parameter’s update history is a **stochastic process** — our main focus.
    - Analyzing these time series with RVSP tools reveals:
        - Stability (stationarity)
        - Noise levels (PSD)
        - Memory in the process (autocorrelation)
        - Whether one run represents all runs (ergodicity)

---

## **5. Generalization vs Memorization**

- **Video Point:** Networks can memorize random labels perfectly but fail to generalize.
- **RVSP PBL Link:**
    - Memorization = higher noise, less stable updates (fitting random patterns).
    - Generalization = smoother, more consistent updates.
    - Can be detected through variance tracking and PSD changes.

---

## **6. Local Minima in High-Dimensional Spaces**

- **Video Point:** Local minima found via gradient descent are often “good enough.”
- **RVSP PBL Link:**
    - Approaching a local minimum often reduces high-frequency noise in updates.
    - Noise analysis + autocorrelation trends can signal convergence.

---

## **RVSP Tools to Apply**

- **Autocorrelation:** Detects how related current updates are to past updates.
- **Stationarity Test:** Confirms if update behavior stabilizes over training.
- **PSD Analysis:** Identifies smooth vs noisy behavior in updates/gradients.
- **Variance Tracking:** Reveals stability or instability over time.
- **Ergodicity Check:** Verifies if one training run can represent the general process.
