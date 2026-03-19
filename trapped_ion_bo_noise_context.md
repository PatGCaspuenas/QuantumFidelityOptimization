# Context Document: Simulating Electric Field Noise Effects on Trapped Ion Gate Calibration via Bayesian Optimization

## Purpose

This document provides the physics and algorithmic context needed to build a prototype simulator. The simulator should model how **1/f electric field noise** causes motional mode frequency drift in a trapped ion system, and how that drift degrades the performance of **Bayesian optimization (BO)** when calibrating a Mølmer-Sørensen (MS) gate.

---

## 1. Physical Setup

### 1.1 The MS Gate and Its Parameters

The Mølmer-Sørensen gate is a two-qubit entangling gate for trapped ions. Its fidelity depends on several control parameters, but the most noise-sensitive is the **sideband detuning** δ — the offset of the laser frequency from a motional sideband.

Typical control parameters for simulation:
- **δ** (sideband detuning): ~1–50 kHz from the motional mode. This is the parameter most affected by noise.
- **Ω** (Rabi frequency / laser intensity): controls gate speed.
- **t_gate** (gate duration): typically 50–500 μs.

Gate fidelity is very sensitive to detuning errors (short lengthscale in GP language) and more tolerant to Rabi frequency variations (longer lengthscale).

### 1.2 Electric Field Noise and Mode Frequency Drift

Electric field noise from trap electrode surfaces causes fluctuations in the trapping potential. This shifts the motional mode frequency ω_m(t). The key consequence:

```
δ_effective(t) = δ_set + Δω_m(t)
```

where δ_set is the detuning the experimentalist programs, and Δω_m(t) is the noise-induced drift.

**Noise spectrum**: Electric field noise in ion traps follows a power-law PSD:

```
S_E(f) ∝ 1/f^α,  with α ≈ 1–2
```

- α = 1: classic 1/f (pink) noise — typical for many traps
- α ≈ 1.5: commonly observed in surface electrode traps
- α = 2: Brownian / random walk noise

This means **low-frequency drift dominates**. Over the timescale of a calibration run (seconds to minutes), the mode frequency wanders slowly. Measurements taken at different times during calibration are not directly comparable because the effective detuning has drifted.

**Typical drift magnitudes**: Mode frequency fluctuations of ~0.1–5 kHz RMS over a calibration run (tens of seconds to minutes), depending on trap quality and temperature.

### 1.3 How Noise Enters the Gate Fidelity

For simulation purposes, a simplified fidelity model for an MS gate as a function of detuning:

```python
def ms_gate_fidelity(delta, Omega, t_gate, delta_opt, noise_std=0.01):
    """
    Simplified MS gate fidelity model.
    
    Args:
        delta: sideband detuning (kHz)
        Omega: Rabi frequency (kHz)
        t_gate: gate time (μs)
        delta_opt: optimal detuning (kHz) — this drifts with mode frequency
        noise_std: measurement shot noise std
    
    Returns:
        Measured fidelity (0 to 1), including shot noise
    """
    # Detuning error sensitivity (sharp — short lengthscale)
    detuning_error = delta - delta_opt
    fid_detuning = np.exp(-(detuning_error / sigma_delta)**2)
    
    # Rabi frequency sensitivity (gentler — longer lengthscale)
    rabi_error = Omega - Omega_opt
    fid_rabi = np.exp(-(rabi_error / sigma_Omega)**2)
    
    # Combined fidelity + shot noise
    fidelity = fid_detuning * fid_rabi
    measured = fidelity + np.random.normal(0, noise_std)
    return np.clip(measured, 0, 1)
```

The critical point: `delta_opt` is NOT fixed — it drifts over time because the mode frequency drifts. This is the core effect to simulate.

---

## 2. Generating 1/f^α Noise via Spectral Filtering

### 2.1 Algorithm

The standard approach (Timmer & Koenig, 1995):

1. Generate white noise in frequency domain: random complex amplitudes with Gaussian-distributed magnitude and uniform phase at each frequency bin.
2. Multiply each frequency bin by the filter `√(1/f^α)` — this shapes the spectrum from flat (white) to the desired power law.
3. Set the DC component (f=0) to zero (removes mean drift).
4. Inverse FFT to get the time-domain signal.
5. Scale to desired physical amplitude (e.g., kHz of mode frequency shift).

### 2.2 Reference Implementation

The `colorednoise` Python package implements this cleanly:

```python
import colorednoise as cn
import numpy as np

def generate_mode_frequency_drift(alpha, duration_s, dt_s, amplitude_kHz, seed=None):
    """
    Generate a time series of motional mode frequency fluctuation.
    
    Args:
        alpha: noise exponent (1.0 = pink, 1.5 = typical trap, 2.0 = Brownian)
        duration_s: total calibration duration in seconds
        dt_s: time step (e.g., time between BO evaluations)
        amplitude_kHz: RMS amplitude of the drift in kHz
        seed: random seed for reproducibility
    
    Returns:
        t: time array (seconds)
        delta_omega: mode frequency fluctuation array (kHz)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration_s / dt_s)
    
    # Generate 1/f^alpha noise using spectral filtering
    noise = cn.powerlaw_psd_gaussian(alpha, n_samples)
    
    # Normalize to desired RMS amplitude
    noise = noise / np.std(noise) * amplitude_kHz
    
    t = np.arange(n_samples) * dt_s
    return t, noise
```

### 2.3 Manual Implementation (if not using colorednoise)

```python
def generate_colored_noise(alpha, n_samples, seed=None):
    """Generate 1/f^alpha noise via spectral filtering (Timmer & Koenig 1995)."""
    rng = np.random.default_rng(seed)
    
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # avoid division by zero; will set DC to 0
    
    # Filter shape: 1/f^(alpha/2) in amplitude
    filter_shape = freqs ** (-alpha / 2)
    
    # White noise in frequency domain
    white = rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))
    
    # Apply filter
    colored_freq = white * filter_shape
    colored_freq[0] = 0  # zero DC
    
    # Back to time domain
    signal = np.fft.irfft(colored_freq, n=n_samples)
    
    # Normalize to unit variance
    signal = signal / np.std(signal)
    return signal
```

---

## 3. Bayesian Optimization Framework

### 3.1 GP Surrogate Model

The BO loop uses a Gaussian Process to model fidelity as a function of control parameters. Key components:

**Kernel**: Squared Exponential with Automatic Relevance Determination (SE-ARD):

```
k(x, x') = σ²_f * exp(-0.5 * Σ_d (x_d - x'_d)² / ℓ_d²)
```

where:
- `σ²_f`: signal variance (overall amplitude of fidelity variation)
- `ℓ_d`: lengthscale for dimension d (how quickly fidelity changes along that axis)
- For MS gate: ℓ_δ (detuning) is short, ℓ_Ω (Rabi frequency) is longer

**Noise model**: `σ²_n` — assumed constant (homoscedastic). In reality, noise is often non-stationary (higher near the fidelity peak from shot noise alone, but potentially lower variance there if coherent errors dominate elsewhere). This mismatch can cause the GP to occasionally recommend suboptimal points.

### 3.2 Hyperparameter Optimization

GP hyperparameters (ℓ, σ²_f, σ²_n) are optimized by maximizing the log marginal likelihood (LML):

```
log p(y|X,θ) = -0.5 * y^T K^{-1} y - 0.5 * log|K| - n/2 * log(2π)
```

where K = K(X,X) + σ²_n I.

**Seed sensitivity**: The LML is non-convex. Different optimizer seeds → different initial hyperparameters → different local optima → different GP posteriors → different acquisition function landscapes → different BO suggestions. This is most pronounced with few data points.

**Transferred lengthscales**: A key strategy is to fix ℓ from prior calibration runs and only optimize σ²_f and σ²_n. This reduces seed sensitivity and requires fewer initial points. After accumulating enough data (e.g., >5d points where d is dimension), ℓ can be unfrozen and allowed to drift.

### 3.3 Acquisition Function

Expected Improvement (EI) is standard:

```
EI(x) = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)
Z = (μ(x) - f_best - ξ) / σ(x)
```

where Φ and φ are the CDF and PDF of a standard normal, and ξ is an exploration parameter.

### 3.4 BO Loop with Time-Varying Noise

The prototype should implement this loop:

```
1. Generate a noise realization: Δω_m(t) for the full calibration duration
2. Choose initial points (Latin hypercube or random)
3. For each BO iteration i:
   a. Record timestamp t_i
   b. Compute effective detuning: δ_eff = δ_suggested + Δω_m(t_i)
   c. Evaluate fidelity at (δ_eff, Ω, t_gate) + shot noise
   d. Feed (δ_suggested, fidelity) to GP — note: GP sees δ_suggested, not δ_eff
   e. Update GP hyperparameters (per schedule)
   f. Optimize acquisition function → next δ_suggested
4. Compare: best fidelity found vs. true optimum
```

The key subtlety in step 3d: the GP is trained on the *programmed* parameters, but the *measured* fidelity reflects the drifted parameters. The GP doesn't know about the drift — it just sees noisy fidelity values that are inconsistent with a stationary landscape. This is the core challenge.

---

## 4. Simulation Architecture

### 4.1 Suggested Module Structure

```
trapped_ion_bo_sim/
├── noise.py           # 1/f noise generation
├── gate_model.py      # MS gate fidelity model (simplified)
├── gp_model.py        # GP surrogate with hyperparameter management
├── bo_loop.py         # Main BO loop with time-stamped evaluations
├── analysis.py        # Plotting, comparison across seeds/noise levels
└── config.py          # Physical parameters, BO settings
```

### 4.2 Key Configuration Parameters

```python
# Physical parameters
MOTIONAL_FREQ_MHZ = 2.0          # Motional mode frequency
DETUNING_RANGE_KHZ = (1, 50)     # Search range for sideband detuning
RABI_RANGE_KHZ = (10, 100)       # Search range for Rabi frequency
GATE_TIME_US = 200                # Fixed gate time for simplicity

# Noise parameters
NOISE_ALPHA = 1.0                 # Noise exponent (1/f)
NOISE_AMPLITUDE_KHZ = 1.0        # RMS mode frequency drift
SHOT_NOISE_STD = 0.02            # Measurement noise std on fidelity

# BO parameters
N_INITIAL = 5                     # Initial random evaluations
N_BO_ITERATIONS = 30              # BO iterations after initial
EVAL_INTERVAL_S = 2.0            # Time between evaluations
CALIBRATION_DURATION_S = 70.0    # Total calibration time

# GP parameters
KERNEL = "SE_ARD"                 # Squared exponential with ARD
FIX_LENGTHSCALES = True           # Whether to use transferred lengthscales
LENGTHSCALE_PRIOR = [5.0, 20.0]  # Prior ℓ for [detuning, Rabi]
HYPEROPT_EVERY = 5                # Re-optimize hyperparams every N steps
```

### 4.3 Recommended Libraries

**Python**:
- `colorednoise`: 1/f noise generation (pip install colorednoise)
- `botorch` + `gpytorch`: State-of-the-art BO and GP (PyTorch-based)
- OR `scikit-optimize`: Simpler API, sklearn-based GPs
- OR `bayesian-optimization`: Lightweight, good for prototyping

**Julia** (if preferred):
- `BayesianOptimization.jl` + `GaussianProcesses.jl`
- `DSP.jl` for spectral filtering

### 4.4 Key Experiments to Run

1. **Baseline (no noise)**: Run BO with Δω_m = 0. Establishes best-case convergence.

2. **With noise, varying α**: Compare α = 0 (white), 1 (pink), 1.5, 2 (Brownian) at fixed amplitude. Shows how noise color affects BO performance.

3. **With noise, varying amplitude**: Fix α = 1, sweep amplitude from 0.1 to 5 kHz. Identifies the threshold where BO starts to degrade.

4. **Seed sensitivity study**: Run the same noise + amplitude with different GP hyperparameter seeds. Quantifies how much the seed matters at different data regime sizes.

5. **Fixed vs. drifting lengthscales**: Compare BO with transferred (fixed) ℓ vs. learning ℓ from scratch. Should show that fixed ℓ is more robust with few points but may become suboptimal as landscape changes.

6. **GP oversaturation test**: Run very long calibrations (many BO iterations) and observe whether performance degrades. Check kernel matrix conditioning, predictive variance collapse, and whether a sliding window helps.

---

## 5. Known Failure Modes and Diagnostics

### 5.1 Non-Zero Failure Rate with "Well-Trained" GP

Even a perfectly trained GP can yield non-zero failure rates because:

1. **Shot noise on fidelity measurements**: Finite sampling gives noisy fidelity estimates. The GP's recommended parameters may be slightly off.

2. **Stochastic function evaluation**: Even at the optimal parameters, each gate execution samples from a noisy process. Some realizations will give lower fidelity.

3. **Non-stationary noise (σ²_n varies across parameter space)**: The GP assumes constant noise variance. Near the optimum, noise may be dominated by shot noise (low variance). Away from optimum, coherent errors and heating effects add variance. A single σ²_n averages these, miscalibrating predictions.

4. **Mode frequency drift between training and execution**: If the GP was trained when ω_m was at one value, and you execute the recommended parameters when ω_m has drifted, you're effectively operating at a wrong detuning.

### 5.2 GP Oversaturation

When the GP is trained on too many points:

- **Computational**: O(N³) matrix inversion becomes slow.
- **Numerical**: Kernel matrix K + σ²_n I becomes ill-conditioned; Cholesky decomposition fails or gives garbage.
- **Model**: Predictive variance → σ²_n everywhere. The acquisition function loses exploration signal. BO stalls because the GP "thinks" it knows everything.

**Remedies**: Sliding window (keep last N_max points), sparse GP approximations (inducing points), or periodic pruning of points far from current region of interest.

### 5.3 Seed Impact on Hyperparameter Optimization

The seed controls initial conditions for the LML optimizer. With few data points, the LML surface is poorly constrained:
- Multiple local optima with similar likelihood
- Flat regions where the optimizer wanders
- Pathological optima (very large ℓ → GP thinks landscape is flat; very small ℓ → pure interpolation with max uncertainty between points)

As data accumulates, the LML sharpens and seed sensitivity decreases. If results vary a lot across seeds, the GP is underdetermined.

---

## 6. References

- Timmer, J. and Koenig, M., "On generating power law noise," Astron. Astrophys. 300, 707-710 (1995) — the standard spectral filtering algorithm
- Sedlacek et al., "Distance scaling of electric-field noise in a surface-electrode ion trap," Phys. Rev. A 97, 020302(R) (2018) — d^{-4} scaling and ~1/f frequency scaling
- Brownnutt et al., "Ion-trap measurements of electric-field noise near surfaces," Rev. Mod. Phys. 87, 1419 (2015) — comprehensive review of anomalous heating
- `colorednoise` Python package: https://github.com/felixpatzelt/colorednoise
- `botorch` documentation: https://botorch.org/
- `BayesianOptimization.jl`: https://github.com/jbrea/BayesianOptimization.jl
