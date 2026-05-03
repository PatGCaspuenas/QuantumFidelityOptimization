# src/CalibrationCode.jl
module CalibrationCode

using Random
using Statistics
using LinearAlgebra
using Logging

using IonSim, QuantumOptics, StatsBase
const pc = IonSim.PhysicalConstants

include("calibration.jl")
include("bayes_hetero_opt.jl")

# -------------------------------
# Public exports
# -------------------------------

# Physics / estimators
export bell_fidelity_phi_plus, ideal, Q_det,
       sigma_binomial, sigma_delta,
       Q_varMS, Q_varMS_σ, Q_varMS_balance_σ

# Heteroscedastic BO
export HeteroGP, HeteroBOResult
export fit_heterogp, predict_latent
export bayesopt_ucb_threshold

end # module
