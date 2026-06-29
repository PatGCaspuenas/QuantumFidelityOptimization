# src/CalibrationCode.jl
module CalibrationCode

using Random
using Statistics
using LinearAlgebra
using Logging

using IonSim, QuantumOptics, StatsBase, Distributions
const pc = IonSim.PhysicalConstants

include("calibration.jl")
include("ms_sequences.jl")

include("bayes_opt.jl")

# Estimators / calibration
export bell_fidelity_phi_plus, ideal, Q_det, Q_noisy, Q_varMS

# Heteroscedastic BO
export HeteroGP, HeteroBOResult
export fit_heterogp, predict_latent, recommend_mean
export bayesopt_ucb

end # module
