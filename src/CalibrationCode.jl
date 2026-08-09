# src/CalibrationCode.jl
module CalibrationCode

using Random
using Statistics
using LinearAlgebra
using Logging

# bayes_hetero_opt.jl is self-contained: it carries the shared GP + BO core
# (verbatim from the main branch's bayes_opt.jl) plus the hardware-facing
# driver, so this is the only include needed.
include("bayes_hetero_opt.jl")

# Heteroscedastic GP + BO
export HeteroGP, HeteroBOResult
export fit_heterogp, predict_latent, recommend_mean
export bayesopt_ucb

# Hardware-facing driver (population callback, optional threshold stop)
export ThresholdBOResult
export score_from_probs, bayesopt_ucb_threshold

end # module
