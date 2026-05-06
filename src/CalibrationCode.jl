# src/CalibrationCode.jl
module CalibrationCode

using Random
using Statistics
using LinearAlgebra
using Logging

include("bayes_hetero_opt.jl")

# -------------------------------
# Public exports
# -------------------------------

# Heteroscedastic BO
export HeteroGP, HeteroBOResult
export fit_heterogp, predict_latent
export bayesopt_ucb_threshold

end # module
