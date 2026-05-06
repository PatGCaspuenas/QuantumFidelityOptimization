import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed, Statistics, Printf, Random, LinearAlgebra
using QuasiMonteCarlo

if nprocs() == 1
    n_add = parse(Int, get(ENV, "GP_STUDY_N_WORKERS",
                           string(max(1, Sys.CPU_THREADS - 1))))
    addprocs(n_add)
end

@everywhere begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
    include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
    using LinearAlgebra, Random, Statistics
    using SpecialFunctions: erf
    BLAS.set_num_threads(1)
    const _INV_SQRT2π = 1.0 / 2.5066282746310002
    const _SQRT_2     = sqrt(2.0)
    const _INV_SQRTΠ  = 1.0 / sqrt(π)
    
    @inline _Φ(z::Float64) = 0.5 * (1.0 + erf(z / _SQRT_2))
    @inline _φ(z::Float64) = exp(-0.5 * z^2) * _INV_SQRT2π
    
    @inline function _crps(μ::Float64, σ::Float64, y::Float64)
        σ < 1e-12 && return abs(y - μ)
        z = (y - μ) / σ
        return σ * (z * (2.0 * _Φ(z) - 1.0) + 2.0 * _φ(z) - _INV_SQRTΠ)
    end
end

# ── Physical setup ────────────────────────────────────────────────────────────
t    = 100.0
base = CalibrationCode.ideal(t)
f_cl0 = base.f_cl; f_sb0 = base.f_sb; A0 = base.A
span_fcl = 2.0e3 * 2π
span_fsb = 2.0e3 * 2π
span_A   = 1.2 * A0 - A0

@everywhere const _t       = $t
@everywhere const _f_cl0   = $f_cl0
@everywhere const _f_sb0   = $f_sb0
@everywhere const _A0      = $A0
@everywhere const _span_fcl = $span_fcl
@everywhere const _span_fsb = $span_fsb
@everywhere const _span_A   = $span_A

@everywhere function Q_fun(u::Vector{Float64}, N::Int)
    fcl = _f_cl0 + _span_fcl * u[1]
    fsb = _f_sb0 + _span_fsb * u[2]
    A   = _A0    + _span_A   * u[3]
    Q, σ = CalibrationCode.Q_varMS_σ(_t, fcl, fsb, A; N=N, numMS=2)
    return clamp(Float64(Q), 0.0, 1.0), Float64(σ)
end

# ── Study parameters ──────────────────────────────────────────────────────────
const N_PTS_LIST   = [100, 250, 500, 1000]
const N_SHOTS_LIST = [50, 100, 250, 400, 1000, 2500]
const N_SEEDS      = parse(Int, get(ENV, "GP_STUDY_N_SEEDS", "50"))
const N_HIGH       = parse(Int, get(ENV, "GP_STUDY_N_HIGH",  "100000"))
const N_TEST       = parse(Int, get(ENV, "GP_STUDY_N_TEST",  "5000"))
@everywhere const N_INIT_ADAPT = 50

println("GP fit quality study — Q_varMS(m=2), 3D")
println("  N_pts    : $(N_PTS_LIST)")
println("  N_shots  : $(N_SHOTS_LIST)")
println("  Seeds    : $(N_SEEDS)   N_high = $(N_HIGH)   N_test = $(N_TEST)")
println("  Workers  : $(nworkers())")

# ── Ground truth: Sobol test grid ─────────────────────────────────────────────
println("\nBuilding ground truth test grid ($(N_TEST) pts × N=$(N_HIGH)) …")
flush(stdout)
lb_3d = [-1.0, -1.0, -1.0]; ub_3d = [1.0, 1.0, 1.0]
_test_mat = QuasiMonteCarlo.sample(N_TEST, lb_3d, ub_3d, QuasiMonteCarlo.SobolSample())
_test_pts = [_test_mat[:, i] for i in 1:N_TEST]

Q_true_vec = pmap(_test_pts; batch_size=8) do u
    Q_fun(u, N_HIGH)[1]
end
Q_true_arr = Float64.(Q_true_vec)
println(@sprintf("  Q_true  mean=%.4f  min=%.4f  max=%.4f",
                 mean(Q_true_arr), minimum(Q_true_arr), maximum(Q_true_arr)))
flush(stdout)

@everywhere const _test_mat_w  = $_test_mat
@everywhere const _Q_true_w    = $Q_true_arr
@everywhere const _N_TEST_w    = $N_TEST
@everywhere const _N_HIGH_w    = $N_HIGH

# ── 1D Slice Setup ────────────────────────────────────────────────────────────
const N_SLICE = 100
const u_grid = range(-1.0, 1.0, length=N_SLICE)

slice_fcl_pts = [[u, 0.0, 0.0] for u in u_grid]
slice_fsb_pts = [[0.0, u, 0.0] for u in u_grid]

println("Computing high-N ground truth for 1D slices...")
Q_true_slice_fcl = [Q_fun(pt, N_HIGH)[1] for pt in slice_fcl_pts]
Q_true_slice_fsb = [Q_fun(pt, N_HIGH)[1] for pt in slice_fsb_pts]

@everywhere const _slice_fcl_mat = $(hcat(slice_fcl_pts...))
@everywhere const _slice_fsb_mat = $(hcat(slice_fsb_pts...))
@everywhere const _N_SLICE       = $N_SLICE

# ── Core Functions ────────────────────────────────────────────────────────────
@everywhere function predict_test(gp)
    μs = Vector{Float64}(undef, _N_TEST_w)
    σs = Vector{Float64}(undef, _N_TEST_w)
    for i in 1:_N_TEST_w
        μ_i, s2_i = CalibrationCode.predict_latent(gp, _test_mat_w[:, i])
        μs[i] = μ_i
        σs[i] = sqrt(max(s2_i, 0.0))
    end
    return μs, σs
end

@everywhere function compute_metrics(μs::Vector{Float64}, σs::Vector{Float64}, Q_ref::Vector{Float64})
    n  = length(Q_ref)
    e² = (μs .- Q_ref) .^ 2          

    rmse   = sqrt(mean(e²))
    rmse_se = std(e²) / (2.0 * max(rmse, 1e-12) * sqrt(n))

    σ_ref² = Q_ref .* (1.0 .- Q_ref) ./ _N_HIGH_w
    rmse_corr = sqrt(max(mean(e²) - mean(σ_ref²), 0.0))

    mae    = mean(abs.(μs .- Q_ref))
    msig   = mean(σs)

    z_gp  = (μs .- Q_ref) ./ max.(σs, 1e-10)
    cov80 = mean(abs.(z_gp) .<= 1.282)
    cov90 = mean(abs.(z_gp) .<= 1.645)
    cov95 = mean(abs.(z_gp) .<= 1.960)

    σ_total = sqrt.(σs .^ 2 .+ σ_ref²)
    z_adj   = (μs .- Q_ref) ./ max.(σ_total, 1e-10)
    cov95_adj = mean(abs.(z_adj) .<= 1.960)

    nlpd = mean(0.5 .* z_gp .^ 2 .+ log.(max.(σs, 1e-10)) .- log(_INV_SQRT2π))
    nlpd_adj = mean(0.5 .* z_adj .^ 2 .+ log.(max.(σ_total, 1e-10)) .- log(_INV_SQRT2π))
    crps = mean(_crps.(μs, σ_total, Q_ref))

    return (rmse=rmse, rmse_se=rmse_se, rmse_corr=rmse_corr,
            mae=mae, mean_sigma=msig,
            cov80=cov80, cov90=cov90, cov95=cov95, cov95_adj=cov95_adj,
            nlpd=nlpd, nlpd_adj=nlpd_adj, crps=crps)
end

@everywhere function fit_and_metrics(X_mat::Matrix{Float64}, y::Vector{Float64}, σy::Vector{Float64}, rng::AbstractRNG; min_sep::Float64=0.02)
    gp = try
        CalibrationCode.fit_heterogp(X_mat, y, σy;
            n_restarts=4, rng=rng,
            ℓ_bounds=(0.05, 1.5), σf_bounds=(0.3, 2.0), c_bounds=(0.05, 3.0))
    catch
        return nothing
    end

    d    = size(X_mat, 1)
    n_tr = size(X_mat, 2)
    sep2 = min_sep * min_sep
    valid = trues(_N_TEST_w)
    @inbounds for j in 1:n_tr
        for i in 1:_N_TEST_w
            valid[i] || continue  
            d2 = 0.0
            for k in 1:d
                Δ = _test_mat_w[k, i] - X_mat[k, j]
                d2 += Δ * Δ
            end
            d2 < sep2 && (valid[i] = false)
        end
    end
    n_valid = sum(valid)
    n_valid < 10 && return nothing   

    μs, σs = predict_test(gp)
    m = compute_metrics(μs[valid], σs[valid], _Q_true_w[valid])
    return (; m..., n_valid=n_valid, l1=gp.ℓ[1], l2=gp.ℓ[2], l3=gp.ℓ[3], sf=gp.σf, c=gp.c, gp=gp)
end

@everywhere function save_slices(gp, mode::String, n_pts::Int, N_shots::Int)
    μ_fcl = Vector{Float64}(undef, _N_SLICE)
    σ_fcl = Vector{Float64}(undef, _N_SLICE)
    for i in 1:_N_SLICE
        μ_fcl[i], s2 = CalibrationCode.predict_latent(gp, _slice_fcl_mat[:, i])
        σ_fcl[i] = sqrt(max(s2, 0.0))
    end
    
    μ_fsb = Vector{Float64}(undef, _N_SLICE)
    σ_fsb = Vector{Float64}(undef, _N_SLICE)
    for i in 1:_N_SLICE
        μ_fsb[i], s2 = CalibrationCode.predict_latent(gp, _slice_fsb_mat[:, i])
        σ_fsb[i] = sqrt(max(s2, 0.0))
    end
    
    return (μ_fcl=μ_fcl, σ_fcl=σ_fcl, μ_fsb=μ_fsb, σ_fsb=σ_fsb)
end

@everywhere function run_config(n_pts::Int, N_shots::Int, seed::Int)
    # Console output removed from here to prevent terminal flooding
    rng = MersenneTwister(seed)

    pts = [rand(rng, 3) .* 2.0 .- 1.0 for _ in 1:n_pts]
    X_mat = hcat(pts...)   

    # ─── Mode 1: Fixed N 
    y_fix  = Vector{Float64}(undef, n_pts)
    σ_fix  = Vector{Float64}(undef, n_pts)
    for i in 1:n_pts
        y_fix[i], σ_fix[i] = Q_fun(pts[i], N_shots)
    end
    met_fix = fit_and_metrics(X_mat, y_fix, σ_fix, MersenneTwister(seed + 100_000))

    # ─── Mode 2: Adaptive N 
    met_adp = if N_shots <= N_INIT_ADAPT
        nothing  
    else
        y_p1 = Vector{Float64}(undef, n_pts)
        for i in 1:n_pts
            y_p1[i], _ = Q_fun(pts[i], N_INIT_ADAPT)
        end

        B_extra = n_pts * (N_shots - N_INIT_ADAPT)
        var_i   = max.(y_p1 .* (1.0 .- y_p1), 1e-4)
        weights = var_i ./ sum(var_i)
        N_extra = max.(1, round.(Int, B_extra .* weights))
        δ = B_extra - sum(N_extra)
        δ != 0 && (N_extra[argmax(weights)] += δ)

        y_adp = Vector{Float64}(undef, n_pts)
        σ_adp = Vector{Float64}(undef, n_pts)
        for i in 1:n_pts
            N_extra_i = max(N_extra[i], 0)
            if N_extra_i > 0
                y2, _ = Q_fun(pts[i], N_extra_i)
                N_tot      = N_INIT_ADAPT + N_extra_i
                y_comb     = (y_p1[i] * N_INIT_ADAPT + y2 * N_extra_i) / N_tot
                y_adp[i]   = clamp(y_comb, 0.0, 1.0)
                σ_adp[i]   = sqrt(y_adp[i] * (1.0 - y_adp[i]) / N_tot)
            else
                y_adp[i]   = y_p1[i]
                σ_adp[i]   = sqrt(y_p1[i] * (1.0 - y_p1[i]) / N_INIT_ADAPT)
            end
        end

        fit_and_metrics(X_mat, y_adp, σ_adp, MersenneTwister(seed + 200_000))
    end

    slices_fix = nothing
    slices_adp = nothing
    
    if seed == 1
        if met_fix !== nothing
            slices_fix = save_slices(met_fix.gp, "fixed", n_pts, N_shots)
        end
        if met_adp !== nothing
            slices_adp = save_slices(met_adp.gp, "adaptive", n_pts, N_shots)
        end
    end

    return (fixed=met_fix, adaptive=met_adp, 
            slices_fix=slices_fix, slices_adp=slices_adp,
            n_pts=n_pts, N_shots=N_shots, seed=seed)
end

# ── CSV Initializers and Helpers ──────────────────────────────────────────────
outdir  = joinpath(@__DIR__, "data")
mkpath(outdir)
outfile = joinpath(outdir, "gp_fit_quality_3d_2ms.csv")
slice_outfile = joinpath(outdir, "slices_output.csv")

# Create Metrics File & Write Header
_header = "mode,N_shots,n_pts,N_total,n_calls,seed," *
          "rmse,rmse_se,rmse_corr,mae,mean_sigma," *
          "cov80,cov90,cov95,cov95_adj," *
          "nlpd,nlpd_adj,crps,n_valid,l1,l2,l3,sf,c"
open(outfile, "w") do io
    println(io, _header)
end

function _write_row(io, mode, r, m, n_calls)
    N_total = r.n_pts * r.N_shots
    @printf(io, "%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        mode, r.N_shots, r.n_pts, N_total, n_calls, r.seed,
        m.rmse, m.rmse_se, m.rmse_corr, m.mae, m.mean_sigma,
        m.cov80, m.cov90, m.cov95, m.cov95_adj,
        m.nlpd, m.nlpd_adj, m.crps, m.n_valid,
        m.l1, m.l2, m.l3, m.sf, m.c)
end

# Create Slices File & Write Header
slice_header = "mode,N_shots,n_pts,slice_axis,u,mu,sigma"
open(slice_outfile, "w") do io
    println(io, slice_header)
end

function write_slice(io, mode_str, r, slices)
    slices === nothing && return
    for (i, u) in enumerate(u_grid)
        @printf(io, "%s,%d,%d,fcl,%.6f,%.6f,%.6f\n", mode_str, r.N_shots, r.n_pts, u, slices.μ_fcl[i], slices.σ_fcl[i])
        @printf(io, "%s,%d,%d,fsb,%.6f,%.6f,%.6f\n", mode_str, r.N_shots, r.n_pts, u, slices.μ_fsb[i], slices.σ_fsb[i])
    end
end

# ── Main Incremental Loop ─────────────────────────────────────────────────────
println("\nStarting incremental GP sweep...")
flush(stdout)
t_start = time()

# We will collect all results here so the summary table at the end still works
all_results = [] 

# Outer loop: Iterating over N_pts first ensures the heaviest workloads (1000) run last
for n_pts in N_PTS_LIST
    for N_shots in N_SHOTS_LIST
        println("Processing: n_pts = $(lpad(n_pts, 4)), N_shots = $(lpad(N_shots, 4)) ...")
        flush(stdout)
        
        # Build the config queue for JUST this shot/point combination
        configs = [(n_pts, N_shots, seed) for seed in 1:N_SEEDS]
        
        # Distribute over workers
        results_chunk = pmap(configs; batch_size=1) do (np, ns, s)
            run_config(np, ns, s)
        end
        
        # Append to our master list for the final summary table
        append!(all_results, results_chunk)
        
        # 1. Append Metrics immediately to CSV
        open(outfile, "a") do io
            for r in results_chunk
                r.fixed    !== nothing && _write_row(io, "fixed",    r, r.fixed,    r.n_pts)
                r.adaptive !== nothing && _write_row(io, "adaptive", r, r.adaptive, 2*r.n_pts)
            end
        end
        
        # 2. Append Slices immediately to CSV (only seed == 1)
        open(slice_outfile, "a") do io
            for r in results_chunk
                r.seed == 1 || continue 
                write_slice(io, "fixed", r, r.slices_fix)
                write_slice(io, "adaptive", r, r.slices_adp)
            end
        end
    end
end

elapsed = round(Int, time() - t_start)
println("\nDone in $(elapsed÷60)m$(elapsed%60)s.")
println("Data has been incrementally saved to:")
println("  - $outfile")
println("  - $slice_outfile")

# ── Quick summary table ───────────────────────────────────────────────────────
for (label, field) in [("RMSE (raw)",  :rmse),
                        ("RMSE (corr)", :rmse_corr),
                        ("Coverage-95 (GP σ)", :cov95),
                        ("Coverage-95 (adj σ)", :cov95_adj),
                        ("CRPS",        :crps)]
    println("\n=== $label — mean across seeds (fixed-N) ===")
    println(@sprintf("%-10s", "n_pts\\N") * join([@sprintf("%8d", N) for N in N_SHOTS_LIST]))
    for n in N_PTS_LIST
        row = @sprintf("%-10d", n)
        for N in N_SHOTS_LIST
            vals = [getfield(r.fixed, field) for r in all_results
                    if r.n_pts == n && r.N_shots == N && r.fixed !== nothing]
            row *= isempty(vals) ? "       —" : @sprintf("%8.4f", mean(vals))
        end
        println(row)
    end
end

rmprocs(workers())