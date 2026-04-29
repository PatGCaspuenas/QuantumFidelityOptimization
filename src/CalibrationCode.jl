# src/CalibrationCode.jl
module CalibrationCode

# Keep core stdlibs available across files
using Random
using Statistics
using LinearAlgebra
using Logging

using IonSim, QuantumOptics, StatsBase
const pc = IonSim.PhysicalConstants

# --- Physics / estimators (IonSim + QuantumOptics live in calibration.jl) ---
include("calibration.jl")
include("ms_sequences.jl")
include("ms_sequence_search.jl")
include("coordinate_search.jl")

# --- Bayesian optimization variants ---
include("bayes_opt.jl")             # homoscedastic GP + EI
include("bayes_hetero_opt.jl")  # heteroscedastic GP + UCB + σ-threshold
include("bayes_cokrig_opt.jl")      # 2-fidelity AR(1) co-kriging
include("bayes_MF_opt.jl")          # N-fidelity augmented-input GP (z-levels)

# If you also added the N-fidelity AR(1) co-kriging implementation:
# include("bayes_mfcokrig_opt.jl")

# -------------------------------
# Public exports (keep this tight)
# -------------------------------

# Estimators / calibration
export bell_fidelity_phi_plus, ideal, Q_det, Q_noisy,
       sigma_binomial, sigma_delta,
       Q_varMS, Q_varMS_σ, Q_varMS_balance_σ, Q_mc_varMS,
       Q_ms_sequence, Q_ms_sequence_det, Q_ms_sequence_σ, sequence_C_subgates

# Homoscedastic BO
export BOResult, bayesopt

# Heteroscedastic BO
export HeteroGP, HeteroBOResult
export fit_heterogp, predict_latent, predict_latent_grad
export bayesopt_ucb_threshold

# 2-fidelity co-kriging BO
export CoKrigResult, cokrig_bayesopt
export predict_high

# Augmented-input multi-fidelity BO (N levels via z_levels)
export MFResult, bayesopt_mf

# N-fidelity AR(1) co-kriging BO (only if you included bayes_mfcokrig_opt.jl)
# export MFCoKrigModel, MFCoKrigResult, fit_mfcokrig, predict_level, mfcokrig_bayesopt

end # module
