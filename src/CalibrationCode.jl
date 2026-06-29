# src/CalibrationCode.jl
module CalibrationCode

using Random
using Statistics
using LinearAlgebra
using Logging

using IonSim, QuantumOptics, StatsBase
const pc = IonSim.PhysicalConstants

include("calibration.jl")
include("ms_sequences.jl")

include("bayes_opt.jl")
include("bayes_hetero_opt.jl")

# Estimators / calibration
export bell_fidelity_phi_plus, ideal, Q_det, Q_noisy, Q_varMS, Q_mc_varMS,
       Q_ms_sequence, Q_ms_sequence_det, sequence_C_subgates

# Homoscedastic BO
export BOResult, bayesopt

# Heteroscedastic BO
export HeteroGP, HeteroBOResult, PathGuardConfig
export fit_heterogp, predict_latent, recommend_mean
export bayesopt_ucb_threshold

end # module
