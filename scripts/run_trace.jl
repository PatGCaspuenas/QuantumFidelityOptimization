# scripts/run_trace.jl
#
# Finds the median simulation (by Q_det) from a benchmark results file,
# re-runs the BO for that seed, and saves per-iteration data:
#   {prefix}_trace.txt    — acquisition point and GP recommendation per iteration
#   {prefix}_gp_slices.txt — GP posterior (μ, σ) on two 1-D slices at u[3]=0 per iteration
#
# Usage (standalone):
#   julia scripts/run_trace.jl
#
# Usage (via run_benchmark_sweep.jl):
#   Set BO_INPUT_FILE and BO_OUTPUT_PREFIX in the combination entry.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))

# ============================================================
# CONFIG — edit these or override via ENV variables
# ============================================================

# Data file to read median seed from (inside data/)
input_file       = get(ENV, "BO_INPUT_FILE",     "QvarMS_N400_v2_0999.txt")

# Output file prefix (inside data/); two files will be created
_default_prefix  = replace(input_file, r"\.txt$" => "_trace")
output_prefix    = get(ENV, "BO_OUTPUT_PREFIX",  _default_prefix)

# BO config — must match the settings used to generate input_file
optimize_det      = get(ENV, "BO_OPT_DET",            "false") == "true"
N_shots           = parse(Int,     get(ENV, "BO_N_SHOTS",         "400"))
sigma_mode        = Symbol(         get(ENV, "BO_SIGMA_MODE",      "binomial"))
use_log_fidelity  = get(ENV, "BO_LOG_FID",             "false") == "true"
n_initial_samples = parse(Int,     get(ENV, "BO_N_INIT",          "12"))
hyper_every       = parse(Int,     get(ENV, "BO_HYPER_EVERY",     "10"))
n_restarts        = parse(Int,     get(ENV, "BO_N_RESTARTS",      "6"))
n_iter            = parse(Int,     get(ENV, "BO_N_ITER",          "120"))
min_iter          = parse(Int,     get(ENV, "BO_MIN_ITER",        "1"))
n_checks          = parse(Int,     get(ENV, "BO_N_CHECKS",        "2"))
add_check_points  = get(ENV, "BO_ADD_CHECK_POINTS",    "true") == "true"
lcb_stop          = get(ENV, "BO_LCB_STOP",            "false") == "true"
random_acq        = get(ENV, "BO_RANDOM_ACQ",          "false") == "true"
M_acq             = parse(Int,     get(ENV, "BO_M_ACQ",           "5000"))

_thresh_env       = get(ENV, "BO_THRESH_Q", "0.999")
fidelity_threshold_Q = if isempty(_thresh_env)
    nothing
elseif _thresh_env == "auto"
    1.0 - 1.0 / N_shots
else
    parse(Float64, _thresh_env)
end

variable_n_mode   = Symbol(get(ENV, "BO_VAR_N_MODE",  "none"))
n_floor           = parse(Int,     get(ENV, "BO_N_FLOOR",         "50"))
n_max_shots       = parse(Int,     get(ENV, "BO_N_MAX",           "2000"))
acq_n_mode        = Symbol(get(ENV, "BO_ACQ_N_MODE",  "floor"))
n_precision       = parse(Float64, get(ENV, "BO_N_PRECISION",     "0.05"))

_sigma_levels_str = get(ENV, "BO_SIGMA_LEVELS", "0.1412,0.1,0.06,0.04472")
sigma_levels      = parse.(Float64, split(_sigma_levels_str, ","))

# Number of points per GP slice (u[3]=0 plane)
n_gp_pts = 100

# ============================================================
# 1. PARSE INPUT FILE — FIND MEDIAN SEED
# ============================================================

data_path = joinpath(@__DIR__, "..", "data", input_file)
isfile(data_path) || error("Input file not found: $data_path")

lines = readlines(data_path)
results_start = findfirst(l -> startswith(l, "=== All Results ==="), lines)
results_start === nothing && error("'=== All Results ===' section not found in $data_path")

header_line = lines[results_start + 1]
data_lines  = filter(!isempty, lines[results_start+2:end])

headers   = split(header_line, "\t")
col_seed  = findfirst(==("Seed"),       headers)
col_qdet  = findfirst(==("Q_det"),      headers)
col_iters = findfirst(==("Iterations"), headers)
(col_seed === nothing || col_qdet === nothing) &&
    error("Columns 'Seed' and 'Q_det' not found in header: $header_line")

sim_seeds  = Int[]
sim_qdets  = Float64[]
sim_niters = Int[]
for line in data_lines
    parts = split(line, "\t")
    length(parts) < max(col_seed, col_qdet) && continue
    push!(sim_seeds,  parse(Int,     parts[col_seed]))
    push!(sim_qdets,  parse(Float64, parts[col_qdet]))
    push!(sim_niters, col_iters !== nothing && length(parts) >= col_iters ?
                      parse(Int, parts[col_iters]) : n_iter)
end
isempty(sim_qdets) && error("No data rows found in $data_path")

sorted_idx   = sortperm(sim_qdets)
median_rank  = div(length(sorted_idx) + 1, 2)
median_pos   = sorted_idx[median_rank]
median_seed  = sim_seeds[median_pos]
median_qdet  = sim_qdets[median_pos]
# Use the iteration count from the benchmark so the trace replicates the
# exact run length (stopping criteria may have ended it before n_iter).
n_iter       = sim_niters[median_pos]

println("=== Trace Run ===")
println("Input file   : $input_file")
println("Simulations  : $(length(sim_qdets))")
println("Median rank  : $median_rank / $(length(sim_qdets))")
println("Median Q_det : $median_qdet")
println("Median seed  : $median_seed")
println("Median iters : $n_iter")
flush(stdout)

# ============================================================
# 2. SETUP PHYSICS / OBJECTIVE
# ============================================================

t = 100.0
base = CalibrationCode.ideal(t)
f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

span_kHz = 2.0
span_fcl = span_kHz * 1e3 * 2π
span_fsb = span_kHz * 1e3 * 2π
span_A   = 1.2 * A0 - A0

u_to_params(u) = (f_cl0 + span_fcl * u[1], f_sb0 + span_fsb * u[2], A0 + span_A * u[3])

function apply_log_fidelity(Q::Float64)
    use_log_fidelity ? log10(max(Q, 1e-15)) : Q
end

function sigma_y_fun(Q_raw::Float64, N::Int)
    optimize_det && return 0.0
    if sigma_mode === :binomial
        one_minus_Q = max(1.0 - Q_raw, 1e-15)
        σ_Q = sqrt(max(Q_raw, 0.0) * one_minus_Q / N)
        return use_log_fidelity ? σ_Q / (max(Q_raw, 1e-15) * log(10)) : σ_Q
    else
        return 1.0 / sqrt(N)
    end
end

function Q_fun(u, N::Int)
    fcl, fsb, A = u_to_params(u)
    Q_raw = if optimize_det
        clamp(CalibrationCode.Q_det(t, fcl, fsb, A), 0.0, 1.0)
    else
        CalibrationCode.Q_varMS(t, fcl, fsb, A; N=N, numMS=2)
    end
    return (apply_log_fidelity(Q_raw), sigma_y_fun(Q_raw, N))
end

bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
α = 1.5
κ = 1.9

fidelity_threshold = fidelity_threshold_Q === nothing ? nothing :
                     (use_log_fidelity ? log10(max(fidelity_threshold_Q, 1e-15)) : fidelity_threshold_Q)
fidelity_threshold_vn = variable_n_mode ∈ (:ci, :verify) ? fidelity_threshold : nothing

# ============================================================
# 3. CALLBACK — collect per-iteration data
# ============================================================

# Pre-compute slice coordinates once
u_slice = collect(range(-1.0, 1.0, length=n_gp_pts))

# Per-iteration accumulation (preallocate roughly)
acq_rec_rows = Vector{NTuple{12,Float64}}()   # (iter, u1_acq, u2_acq, u3_acq, y_acq, u1_rec, u2_rec, u3_rec, mu_rec, sigma_rec, q_det_rec)
gp_u1_rows   = Vector{NTuple{4,Float64}}()    # (iter, u1, mu, sigma)
gp_u2_rows   = Vector{NTuple{4,Float64}}()    # (iter, u2, mu, sigma)

function iter_callback(it::Int, gp::CalibrationCode.HeteroGP,
                       x_acq::Vector{Float64}, y_acq_raw::Float64,
                       x_rec::Vector{Float64}, mu_rec::Float64, sigma_rec::Float64,
                       y_rec_cur::Float64)
    fcl_r, fsb_r, A_r = u_to_params(x_rec)
    q_det_r = clamp(CalibrationCode.Q_det(t, fcl_r, fsb_r, A_r), 0.0, 1.0)
    push!(acq_rec_rows, (Float64(it),
        x_acq[1], x_acq[2], x_acq[3], y_acq_raw,
        x_rec[1], x_rec[2], x_rec[3], mu_rec, sigma_rec, q_det_r, y_rec_cur))

    # GP slice: u[1] varies, u[2]=0, u[3]=0
    for u1 in u_slice
        μ, s2 = CalibrationCode.predict_latent(gp, [u1, 0.0, 0.0])
        push!(gp_u1_rows, (Float64(it), u1, μ, sqrt(max(s2, 0.0))))
    end

    # GP slice: u[2] varies, u[1]=0, u[3]=0
    for u2 in u_slice
        μ, s2 = CalibrationCode.predict_latent(gp, [0.0, u2, 0.0])
        push!(gp_u2_rows, (Float64(it), u2, μ, sqrt(max(s2, 0.0))))
    end
end

# ============================================================
# 4. RUN BO
# ============================================================

println("\nRunning BO (seed=$median_seed, n_iter=$n_iter, N_shots=$N_shots) ...")
flush(stdout)

result = CalibrationCode.bayesopt_ucb_threshold(Q_fun;
    bounds            = bounds,
    n_shots           = N_shots,
    sigma_mode        = sigma_mode,
    n_init            = n_initial_samples,
    n_iter            = n_iter,
    M_acq             = M_acq,
    κ                 = κ,
    α                 = α,
    seed              = median_seed,
    maximize          = true,
    fidelity_threshold = fidelity_threshold,
    min_iter          = min_iter,
    n_checks          = n_checks,
    add_check_points  = add_check_points,
    hyper_every       = hyper_every,
    learn_noise_scale = !optimize_det,
    n_restarts        = n_restarts,
    variable_n_mode   = variable_n_mode,
    n_floor           = n_floor,
    n_max_shots       = n_max_shots,
    n_precision       = n_precision,
    acq_n_mode        = acq_n_mode,
    sigma_levels      = sigma_levels,
    fidelity_threshold_vn = fidelity_threshold_vn,
    use_log_fidelity  = use_log_fidelity,
    lcb_stop          = lcb_stop,
    random_acq        = random_acq,
    iter_callback     = iter_callback,
)

fcl_rec, fsb_rec, A_rec = u_to_params(result.x_rec)
Q_det_final = clamp(CalibrationCode.Q_det(t, fcl_rec, fsb_rec, A_rec), 0.0, 1.0)
n_iters_run = result.n_iter_actual > 0 ? result.n_iter_actual : n_iter

println("Done.")
println("  Q_det (final)   = $Q_det_final")
println("  Iterations run  = $n_iters_run")
println("  Total shots     = $(result.total_shots)")
println("  Training points = $(length(result.y))")

# ============================================================
# 5. SAVE RESULTS
# ============================================================

out_trace_file  = joinpath(@__DIR__, "..", "data", output_prefix * "_trace.txt")
out_slices_file = joinpath(@__DIR__, "..", "data", output_prefix * "_gp_slices.txt")

# --- File 1: acquisition point + GP recommendation per iteration ---
open(out_trace_file, "w") do io
    println(io, "# run_trace.jl output — acquisition and recommendation per BO iteration")
    println(io, "# input_file=$(input_file)  median_seed=$(median_seed)  Q_det_median_input=$(median_qdet)")
    println(io, "# Q_det_final=$(Q_det_final)  n_iter_run=$(n_iters_run)  total_shots=$(result.total_shots)")
    println(io, "# N_shots=$(N_shots)  sigma_mode=$(sigma_mode)  threshold=$(fidelity_threshold_Q)")
    println(io, "# variable_n_mode=$(variable_n_mode)  lcb_stop=$(lcb_stop)  random_acq=$(random_acq)")
    println(io, "#")
    println(io, "# SECTION: INITIAL_TRAINING")
    println(io, "# First $(n_initial_samples) training points (before BO loop)")
    println(io, "# u1\tu2\tu3\ty\tsigma_y")
    for i in 1:min(n_initial_samples, length(result.y))
        u = result.X[:, i]
        println(io, "$(u[1])\t$(u[2])\t$(u[3])\t$(result.y[i])\t$(result.σy[i])")
    end
    println(io, "#")
    println(io, "# SECTION: ACQ_REC")
    println(io, "# iter\tu1_acq\tu2_acq\tu3_acq\ty_acq\tu1_rec\tu2_rec\tu3_rec\tmu_rec\tsigma_rec\tq_det_rec\ty_rec_cur")
    for r in acq_rec_rows
        println(io, "$(Int(r[1]))\t$(r[2])\t$(r[3])\t$(r[4])\t$(r[5])\t$(r[6])\t$(r[7])\t$(r[8])\t$(r[9])\t$(r[10])\t$(r[11])\t$(r[12])")
    end
    println(io, "#")
    println(io, "# SECTION: FINAL_TRAINING")
    println(io, "# All training points accumulated by end of run")
    println(io, "# u1\tu2\tu3\ty\tsigma_y")
    for i in 1:length(result.y)
        u = result.X[:, i]
        println(io, "$(u[1])\t$(u[2])\t$(u[3])\t$(result.y[i])\t$(result.σy[i])")
    end
end

# --- File 2: GP slice predictions per iteration ---
open(out_slices_file, "w") do io
    println(io, "# run_trace.jl output — GP posterior on 1-D slices at u[3]=0 per iteration")
    println(io, "# input_file=$(input_file)  median_seed=$(median_seed)")
    println(io, "# slice u1: vary u[1] with u[2]=0, u[3]=0")
    println(io, "# slice u2: vary u[2] with u[1]=0, u[3]=0")
    println(io, "#")
    println(io, "# SECTION: GP_SLICE_U1")
    println(io, "# iter\tu1\tmu\tsigma")
    for r in gp_u1_rows
        println(io, "$(Int(r[1]))\t$(r[2])\t$(r[3])\t$(r[4])")
    end
    println(io, "#")
    println(io, "# SECTION: GP_SLICE_U2")
    println(io, "# iter\tu2\tmu\tsigma")
    for r in gp_u2_rows
        println(io, "$(Int(r[1]))\t$(r[2])\t$(r[3])\t$(r[4])")
    end
end

println("\nResults saved to:")
println("  $out_trace_file")
println("  $out_slices_file")
