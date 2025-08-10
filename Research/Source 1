## **But what is a neural network? | Deep learning chapter 1 by** [3Blue1Brown](https://www.youtube.com/@3blue1brown)

## **1. Data as a Random Process**

- **Video point:** Input is a 28×28 pixel image, different handwriting styles produce variation.
- **RVSP PBL link:**
    - Each pixel = **random variable**.
    - Across the dataset, pixel values form a **stochastic process**.
    - This randomness is the **source** of variability in training.

---

## **2. Activations as Transformed Random Variables**

- **Video point:** Hidden layers extract edges, curves, and shapes from pixels.
- **RVSP PBL link:**
    - Each activation = **new random variable** derived from inputs.
    - Their distribution changes over time during training.
    - Can test **stationarity** of these activations.

---

## **3. Weights & Biases as Time-Dependent Signals**

- **Video point:** Connections between neurons have weights & biases that adjust during training.
- **RVSP PBL link:**
    - Each parameter changes every iteration = **time series**.
    - The optimizer update signal is **random but with patterns** (stochastic process).
    - Analyze **autocorrelation** to check temporal dependencies.

---

## **4. Optimizer as a Stochastic Process**

- **Video point:** Parameters are updated based on errors in prediction.
- **RVSP PBL link:**
    - Weight updates have noise from mini-batch sampling.
    - PSD analysis tells us if updates are **smooth (low freq)** or **jittery (high freq)**.
    - Stationarity checks tell if update behavior is consistent across training.

---

## **5. Activation Functions Influence Noise**

- **Video point:** Sigmoid vs ReLU affects how neurons “fire” and process inputs.
- **RVSP PBL link:**
    - Activation choice changes gradient distribution.
    - ReLU reduces saturation → more stable gradient magnitudes.
    - Can compare PSD & variance of updates under different activations.

---

## **6. Training as a Noisy Signal Path**

- **Video point:** Network is one big function mapping inputs → outputs via many small computations.
- **RVSP PBL link:**
    - The **sequence of updates** is like a noisy signal in a communication channel.
    - Our analysis uses RVSP tools to “filter” and understand this signal.

---

### **Main RVSP Tools Applied**

- **Autocorrelation:** Measures how related current updates are to past updates.
- **Stationarity Test:** Checks if update behavior changes over time.
- **Ergodicity:** Tests if one training run is representative of all possible runs.
- **PSD Analysis:** Shows noise energy in different frequencies of the update sign
