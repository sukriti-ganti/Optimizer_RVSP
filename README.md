# Design and Analysis of a Custom Optimizer Using Random Process Theory
## PROJECT OVERVIEW:
This project explores the behavior of machine learning optimizers—through the lens of Random Variables and Stochastic Processes (RVSP). Instead of treating the optimizer as a black box, we model its update dynamics as a stochastic process.
Building on this foundation, the project goes one step further by designing a new optimizer from scratch—grounded entirely in RVSP principles such as autocorrelation, power spectral density, and ergodicity. The goal is not just to analyze, but to engineer a more theoretically robust learning algorithm.

**##OBJECTIVES:**
1. Initially, we start by modelling the learning dynamics of optimization algorithms as stochastic processes using RVSP principles
2. The following principles help us to understand how optimizers behave over time:
   - autocorrelation
   - stationarity
   - power spectral density
   - ergodicity
3. We identify patterns and use the results to fix the inaccuracies that are often overlooked in deep learning workflows
4. Following this, the objective is to design a custom optimization algorithm that works with RVSP concepts as its foundation
5. Finally, we use mathematical modeling and experimental simulation to validate the proposed optimizer


**## RVSP CONCEPTS USED:**
This project is built on the foundation of RVSP concepts, which include:
1. Random Variables: The sequence of gradients or updates to the model can be thought of as a signal which can be further classified as a random process (due to mini batches, model behaviour, etc.)
2. Autocorrelation and cross-correlation: Used to analyze temporal dependencies in update signals that determine how stable or jittery the resultant signal is.
3. Stationarity: Explores whether the statistical properties of updates remain consistent over time 
4. Ergodicity: Used to verify whether one model running at a time over multiple instances of time would give the same result as multiple models running at the same time (average values)
5. Power Spectral Density (PSD): Used to determine the energy distribution over the frequency content to help us understand the noise present in the model
6. Gaussian and Non-Gaussian Processes: Explored in the context of modeling gradient distributions


## MODELING ASSUMPTIONS
- The gradient \( g_t \) is a realization of a discrete-time random process.
- Gradients across mini-batches are assumed to be statistically independent or weakly correlated and are classified as random variables
- The update signal \( u(t) \) is wide-sense stationary (WSS) within local regions of training.
- The optimizer behaves as an LTI (Linear Time-Invariant) system under small perturbations.

## REPOSITORY STRUCTURE
```bash
├── src/                 # Python modules: custom optimizer, analysis tools
├── notebooks/           # Jupyter notebooks for modeling, simulation, plotting
├── simulations/         # Scripts to run experiments and save results
├── results/             # Generated plots: ACF, PSD, loss curves
├── docs/                # Review documents (R1 to R4)
├── report/              # Final report and presentation
├── requirements.txt     # Python dependencies
└── README.md            # Project overview and instructions
