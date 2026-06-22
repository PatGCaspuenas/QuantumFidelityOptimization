import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed, Statistics, Printf, Random, LinearAlgebra
using QuasiMonteCarlo

# ── Workers ───────────────────────────────────────────────────────────────────
const N_WORKERS = parse(Int, get(ENV, "GP_STUDY_N_WORKERS", "19"))
if nprocs() == 1
    addprocs(N_WORKERS)
end

@everywhere begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
    include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
    using LinearAlgebra, Random, Statistics, Distributions
    using SpecialFunctions: erf
    BLAS.set_num_threads(1)

    const _INV_SQRT2π = 1.0 / sqrt(2π)
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

# ── Physical setup ─────────────────────────────────────────────────────────────
# Matches optimization_data_generation.jl: TRACE_FREQ_SPAN_KHZ=10, TRACE_BOUND_SCALE=0.5
const FREQ_SPAN_KHZ = 10.0
const BOUND_SCALE   = 0.5
const NEAR_THRESH   = 0.1   # off-axis proximity for training-point overlay

t = 100.0
base = CalibrationCode.ideal(t)
f_cl0 = Float64(base.f_cl)
f_sb0 = Float64(base.f_sb)
A0    = Float64(base.A)

# span_fcl = FREQ_SPAN_KHZ_OVERRIDE * 1e3  (no 2pi factor, matching opt_data_gen.jl line 228)
span_fcl = FREQ_SPAN_KHZ * 1e3
span_fsb = FREQ_SPAN_KHZ * 1e3
span_A   = 1.2 * A0 - A0

@everywhere const _t        = $t
@everywhere const _f_cl0    = $f_cl0
@everywhere const _f_sb0    = $f_sb0
@everywhere const _A0       = $A0
@everywhere const _span_fcl = $span_fcl
@everywhere const _span_fsb = $span_fsb
@everywhere const _span_A   = $span_A
@everywhere const _BOUND       = $BOUND_SCALE   # domain is [-_BOUND, _BOUND]^3
@everywhere const _NEAR_THRESH = $NEAR_THRESH

# ── Score functions (full_l1, matching TRACE_SCORE_MODE=full_l1) ───────────────
@everywhere begin
    function _u_to_params(u::Vector{Float64})
        return (_f_cl0 + _span_fcl * u[1],
                _f_sb0 + _span_fsb * u[2],
                _A0    + _span_A   * u[3])
    end

    function _varms_weights(u::Vector{Float64})
        fcl, fsb, A = _u_to_params(u)
        subgates = [CalibrationCode.MSSubgate(π / 2, 0.0),
                    CalibrationCode.MSSubgate(π / 2, 0.0)]
        pulses = CalibrationCode.build_closed_loop_ms_sequence(_t, fcl, fsb, A, subgates)
        pops   = CalibrationCode.populations_ms_sequence(pulses)
        w = Float64[max(pops.gg, 0.0), max(pops.eg, 0.0),
                    max(pops.ge, 0.0), max(pops.ee, 0.0)]
        s = sum(w)
        s > 0.0 && (w ./= s)
        return w
    end

    @inline function _full_l1_score(p_ss::Float64, p_sd::Float64,
                                    p_ds::Float64, p_dd::Float64)
        l1 = abs(p_ss) + abs(p_sd) + abs(p_ds) + abs(p_dd - 1.0)
        return clamp(1.0 - 0.5 * l1, 0.0, 1.0)
    end

    # Deterministic (noise-free) score.
    function Q_det(u::Vector{Float64})
        w = _varms_weights(u)
        return _full_l1_score(w[1], w[2], w[3], w[4])
    end

    # Stochastic score via multinomial shot noise.
    # σ uses the exact full_l1 noise model: sqrt(p_dd*(1-p_dd)/N).
    function Q_fun(u::Vector{Float64}, N::Real, rng::AbstractRNG)
        w = _varms_weights(u)
        if isinf(N)
            return _full_l1_score(w[1], w[2], w[3], w[4]), 0.0
        end
        n_int  = Int(N)
        counts = rand(rng, Multinomial(n_int, w))
        n_f    = Float64(n_int)
        y = _full_l1_score(counts[1]/n_f, counts[2]/n_f,
                            counts[3]/n_f, counts[4]/n_f)
        σ = sqrt(max(w[4] * (1.0 - w[4]), 0.0) / n_f)
        return y, σ
    end
end

# ── Study parameters ───────────────────────────────────────────────────────────
const N_PTS_LIST   = [10, 50, 100, 250, 500, 1000]
const N_SHOTS_LIST = [100, 1000, 10000, 100000, Inf]
const N_SEEDS = parse(Int, get(ENV, "GP_STUDY_N_SEEDS", "50"))
const N_TEST  = parse(Int, get(ENV, "GP_STUDY_N_TEST",  "5000"))
const N_SLICE = 100

println("GP fit quality study — full_l1 score, FREQ_SPAN=$(FREQ_SPAN_KHZ) kHz, BOUND=$(BOUND_SCALE)")
println("  N_pts    : $N_PTS_LIST")
println("  N_shots  : $N_SHOTS_LIST")
println("  Seeds    : $N_SEEDS   N_test = $N_TEST")
println("  Workers  : $(nworkers())")
flush(stdout)

# ── Ground truth test grid ─────────────────────────────────────────────────────
println("\nBuilding test grid ($N_TEST pts in [-$(BOUND_SCALE), $(BOUND_SCALE)]^3) ...")
flush(stdout)
lb3d = fill(-BOUND_SCALE, 3)
ub3d = fill( BOUND_SCALE, 3)
_test_mat = QuasiMonteCarlo.sample(N_TEST, lb3d, ub3d, QuasiMonteCarlo.SobolSample())

# Precompute Q_det and p_dd (ee population) for every test point.
# p_dd feeds the exact full_l1 shot noise variance: σ² = p_dd*(1-p_dd)/N.
_q_pdd_vec = pmap(1:N_TEST; batch_size=32) do i
    u = _test_mat[:, i]
    w = _varms_weights(u)
    q = _full_l1_score(w[1], w[2], w[3], w[4])
    (q, w[4])
end
Q_true_arr = Float64[r[1] for r in _q_pdd_vec]
p_dd_arr   = Float64[r[2] for r in _q_pdd_vec]
println(@sprintf("  Q_true: mean=%.4f  min=%.4f  max=%.4f",
                 mean(Q_true_arr), minimum(Q_true_arr), maximum(Q_true_arr)))
flush(stdout)

@everywhere const _test_mat_w = $_test_mat
@everywhere const _Q_true_w   = $Q_true_arr
@everywhere const _p_dd_w     = $p_dd_arr
@everywhere const _N_TEST_w   = $N_TEST

# ── Slice ground truth (Q_det, independent of seed/N_shots/n_pts) ─────────────
println("Precomputing 1D slice ground truth ...")
flush(stdout)
const _slice_u_vec = collect(range(-BOUND_SCALE, BOUND_SCALE; length=N_SLICE))
Q_det_fcl = Float64[Q_det([u, 0.0, 0.0]) for u in _slice_u_vec]
Q_det_fsb = Float64[Q_det([0.0, u, 0.0]) for u in _slice_u_vec]
Q_det_amp = Float64[Q_det([0.0, 0.0, u]) for u in _slice_u_vec]

@everywhere const _slice_u         = $_slice_u_vec
@everywhere const _N_SLICE         = $N_SLICE
@everywhere const _Q_det_slice_fcl = $Q_det_fcl
@everywhere const _Q_det_slice_fsb = $Q_det_fsb
@everywhere const _Q_det_slice_amp = $Q_det_amp

# ── Core worker function ───────────────────────────────────────────────────────
@everywhere begin
    # GP quality metrics against precomputed test grid.
    # σ_noise² = p_dd*(1-p_dd)/N (exact full_l1 model); 0 when N=Inf.
    # The combined metric (cov95_adj, nlpd_adj, crps) accounts for both
    # GP uncertainty and measurement noise simultaneously.
    function compute_metrics(μs::Vector{Float64}, σs::Vector{Float64},
                             Q_ref::Vector{Float64}, p_dd_ref::Vector{Float64},
                             N_shots::Real)
        n = length(Q_ref)

        e² = (μs .- Q_ref) .^ 2
        rmse    = sqrt(mean(e²))
        rmse_se = std(e²) / (2.0 * max(rmse, 1e-12) * sqrt(n))

        σ_noise² = isinf(N_shots) ? zeros(Float64, n) :
                   p_dd_ref .* (1.0 .- p_dd_ref) ./ Float64(N_shots)
        rmse_corr        = sqrt(max(mean(e²) - mean(σ_noise²), 0.0))
        mae              = mean(abs.(μs .- Q_ref))
        mean_sigma_gp    = mean(σs)
        mean_sigma_noise = sqrt(mean(σ_noise²))

        # GP-only (ignores measurement noise)
        z_gp  = (μs .- Q_ref) ./ max.(σs, 1e-10)
        cov80 = mean(abs.(z_gp) .<= 1.282)
        cov90 = mean(abs.(z_gp) .<= 1.645)
        cov95 = mean(abs.(z_gp) .<= 1.960)
        nlpd  = mean(0.5 .* z_gp .^ 2 .+ log.(max.(σs, 1e-10)) .- log(_INV_SQRT2π))

        # GP + measurement noise combined
        σ_total   = sqrt.(σs .^ 2 .+ σ_noise²)
        z_adj     = (μs .- Q_ref) ./ max.(σ_total, 1e-10)
        cov95_adj = mean(abs.(z_adj) .<= 1.960)
        nlpd_adj  = mean(0.5 .* z_adj .^ 2 .+ log.(max.(σ_total, 1e-10)) .- log(_INV_SQRT2π))
        crps      = mean(_crps.(μs, σ_total, Q_ref))
        # Mean squared standardized error: (μ_GP - Q_true)² / σ_total²
        # Calibrated predictor → msse ≈ 1; >1 overconfident, <1 underconfident.
        msse      = mean(z_adj .^ 2)
        msse_gp   = mean(z_gp  .^ 2)

        return (rmse=rmse, rmse_se=rmse_se, rmse_corr=rmse_corr,
                mae=mae, mean_sigma_gp=mean_sigma_gp,
                mean_sigma_noise=mean_sigma_noise,
                cov80=cov80, cov90=cov90, cov95=cov95,
                cov95_adj=cov95_adj, nlpd=nlpd, nlpd_adj=nlpd_adj,
                crps=crps, msse=msse, msse_gp=msse_gp)
    end

    function run_config(n_pts::Int, N_shots::Real, seed::Int)
        rng_train = MersenneTwister(seed)
        rng_gp    = MersenneTwister(seed + 100_000)
        rng_slice = MersenneTwister(seed + 200_000)

        # Training points: uniform random in [-BOUND, BOUND]^3
        pts   = [rand(rng_train, 3) .* (2.0 * _BOUND) .- _BOUND for _ in 1:n_pts]
        X_mat = hcat(pts...)

        y  = Vector{Float64}(undef, n_pts)
        σy = Vector{Float64}(undef, n_pts)
        for i in 1:n_pts
            y[i], σy[i] = Q_fun(pts[i], N_shots, rng_train)
        end

        # Fit GP once — reused for both metrics and slice predictions
        gp = try
            CalibrationCode.fit_heterogp(X_mat, y, σy;
                n_restarts=4, rng=rng_gp,
                ℓ_bounds=(0.05, 1.5), σf_bounds=(0.3, 2.0), c_bounds=(0.05, 3.0))
        catch
            return (; metrics=nothing, slices=nothing,
                      n_pts=n_pts, N_shots=N_shots, seed=seed)
        end

        # ── Error metrics ────────────────────────────────────────────────────
        d, n_tr = size(X_mat)
        sep2  = 0.02 * 0.02
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

        metrics = nothing
        if n_valid >= 10
            μs = Vector{Float64}(undef, _N_TEST_w)
            σs = Vector{Float64}(undef, _N_TEST_w)
            @inbounds for i in 1:_N_TEST_w
                μ, s2 = CalibrationCode.predict_latent(gp, _test_mat_w[:, i])
                μs[i] = μ
                σs[i] = sqrt(max(s2, 0.0))
            end
            m = compute_metrics(μs[valid], σs[valid],
                                _Q_true_w[valid], _p_dd_w[valid], N_shots)
            metrics = (; m..., n_valid=n_valid,
                         l1=gp.ℓ[1], l2=gp.ℓ[2], l3=gp.ℓ[3],
                         sf=gp.σf, c=gp.c)
        end

        # ── 1D slices: GP prediction + noisy Q_fun sample ───────────────────
        μ_fcl = Vector{Float64}(undef, _N_SLICE); σ_fcl = similar(μ_fcl)
        μ_fsb = Vector{Float64}(undef, _N_SLICE); σ_fsb = similar(μ_fcl)
        μ_amp = Vector{Float64}(undef, _N_SLICE); σ_amp = similar(μ_fcl)
        q_fun_fcl = similar(μ_fcl)
        q_fun_fsb = similar(μ_fcl)
        q_fun_amp = similar(μ_fcl)

        @inbounds for i in 1:_N_SLICE
            u_f = [_slice_u[i], 0.0, 0.0]
            u_s = [0.0, _slice_u[i], 0.0]
            u_a = [0.0, 0.0, _slice_u[i]]

            μ, s2 = CalibrationCode.predict_latent(gp, u_f)
            μ_fcl[i] = μ; σ_fcl[i] = sqrt(max(s2, 0.0))
            μ, s2 = CalibrationCode.predict_latent(gp, u_s)
            μ_fsb[i] = μ; σ_fsb[i] = sqrt(max(s2, 0.0))
            μ, s2 = CalibrationCode.predict_latent(gp, u_a)
            μ_amp[i] = μ; σ_amp[i] = sqrt(max(s2, 0.0))

            q_fun_fcl[i] = Q_fun(u_f, N_shots, rng_slice)[1]
            q_fun_fsb[i] = Q_fun(u_s, N_shots, rng_slice)[1]
            q_fun_amp[i] = Q_fun(u_a, N_shots, rng_slice)[1]
        end

        # q_det arrays are @everywhere const — copied into NamedTuple for transfer
        slices = (μ_fcl=μ_fcl, σ_fcl=σ_fcl,
                  μ_fsb=μ_fsb, σ_fsb=σ_fsb,
                  μ_amp=μ_amp, σ_amp=σ_amp,
                  q_det_fcl=copy(_Q_det_slice_fcl),
                  q_det_fsb=copy(_Q_det_slice_fsb),
                  q_det_amp=copy(_Q_det_slice_amp),
                  q_fun_fcl=q_fun_fcl,
                  q_fun_fsb=q_fun_fsb,
                  q_fun_amp=q_fun_amp)

        # ── Nearby training points (off-axis distance ≤ _NEAR_THRESH) ──────────
        near_rows = Tuple{String,Float64,Float64,Float64}[]
        for j in 1:n_pts
            u = pts[j]
            abs(u[2]) ≤ _NEAR_THRESH && abs(u[3]) ≤ _NEAR_THRESH &&
                push!(near_rows, ("fcl", u[1], y[j], σy[j]))
            abs(u[1]) ≤ _NEAR_THRESH && abs(u[3]) ≤ _NEAR_THRESH &&
                push!(near_rows, ("fsb", u[2], y[j], σy[j]))
            abs(u[1]) ≤ _NEAR_THRESH && abs(u[2]) ≤ _NEAR_THRESH &&
                push!(near_rows, ("amp", u[3], y[j], σy[j]))
        end

        return (; metrics=metrics, slices=slices, near=near_rows,
                  n_pts=n_pts, N_shots=N_shots, seed=seed)
    end
end

# ── CSV helpers ────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "data")
mkpath(outdir)
metrics_file = joinpath(outdir, "gp_fit_quality_3d_2ms.csv")
slices_file  = joinpath(outdir, "slices_output.csv")
near_file    = joinpath(outdir, "train_near_output.csv")

open(metrics_file, "w") do io
    println(io,
        "n_pts,N_shots,N_total,seed," *
        "rmse,rmse_se,rmse_corr,mae,mean_sigma_gp,mean_sigma_noise," *
        "cov80,cov90,cov95,cov95_adj,nlpd,nlpd_adj,crps,msse,msse_gp," *
        "n_valid,l1,l2,l3,sf,c")
end
open(slices_file, "w") do io
    println(io, "n_pts,N_shots,seed,axis,u,mu_gp,sigma_gp,q_det,q_fun_N")
end
open(near_file, "w") do io
    println(io, "n_pts,N_shots,seed,axis,u_proj,Q_obs,sig_obs")
end

_ns_str(N::Real) = isinf(N) ? "Inf" : string(Int(N))

function _write_metrics_row(io, r, m)
    ns = _ns_str(r.N_shots)
    nt = isinf(r.N_shots) ? "Inf" : string(Int(r.n_pts * r.N_shots))
    @printf(io, "%d,%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        r.n_pts, ns, nt, r.seed,
        m.rmse, m.rmse_se, m.rmse_corr, m.mae,
        m.mean_sigma_gp, m.mean_sigma_noise,
        m.cov80, m.cov90, m.cov95, m.cov95_adj,
        m.nlpd, m.nlpd_adj, m.crps, m.msse, m.msse_gp,
        m.n_valid, m.l1, m.l2, m.l3, m.sf, m.c)
end

function _write_near_rows(io, r)
    isempty(r.near) && return
    ns = _ns_str(r.N_shots)
    for (axis, u_proj, Q_obs, sig_obs) in r.near
        @printf(io, "%d,%s,%d,%s,%.6f,%.6f,%.6f\n",
            r.n_pts, ns, r.seed, axis, u_proj, Q_obs, sig_obs)
    end
end

function _write_slice_rows(io, r, sl)
    sl === nothing && return
    np_s   = string(r.n_pts)
    ns_s   = _ns_str(r.N_shots)
    seed_s = string(r.seed)
    for (axis, μv, σv, q_det_v, q_fun_v) in (
            ("fcl", sl.μ_fcl, sl.σ_fcl, sl.q_det_fcl, sl.q_fun_fcl),
            ("fsb", sl.μ_fsb, sl.σ_fsb, sl.q_det_fsb, sl.q_fun_fsb),
            ("amp", sl.μ_amp, sl.σ_amp, sl.q_det_amp, sl.q_fun_amp))
        @inbounds for i in eachindex(_slice_u_vec)
            @printf(io, "%s,%s,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                np_s, ns_s, seed_s, axis,
                _slice_u_vec[i], μv[i], σv[i], q_det_v[i], q_fun_v[i])
        end
    end
end

# ── Main sweep ─────────────────────────────────────────────────────────────────
# pmap over seeds for each (n_pts, N_shots) pair so results are flushed to CSV
# after every batch of N_SEEDS configs rather than waiting for all 1500 to finish.
n_total = length(N_PTS_LIST) * length(N_SHOTS_LIST) * N_SEEDS
println("\nRunning $n_total configs ($(nworkers()) workers), writing after each batch ...")
flush(stdout)
t_start = time()

all_results = []

for n_pts in N_PTS_LIST
    for N_shots in N_SHOTS_LIST
        t_batch = time()
        print(@sprintf("  n_pts=%4d  N_shots=%-7s  ...", n_pts, _ns_str(N_shots)))
        flush(stdout)

        configs = [(n_pts, N_shots, seed) for seed in 1:N_SEEDS]
        chunk   = pmap(configs; batch_size=1) do (np, ns, s)
            run_config(np, ns, s)
        end
        append!(all_results, chunk)

        open(metrics_file, "a") do mio
            open(slices_file, "a") do sio
                open(near_file, "a") do nio
                    for r in chunk
                        r.metrics !== nothing && _write_metrics_row(mio, r, r.metrics)
                        r.slices  !== nothing && _write_slice_rows(sio, r, r.slices)
                        _write_near_rows(nio, r)
                    end
                end
            end
        end

        n_ok = count(r -> r.metrics !== nothing, chunk)
        println(@sprintf(" done in %.1fs  (%d/%d ok)", time()-t_batch, n_ok, N_SEEDS))
        flush(stdout)
    end
end

elapsed = round(Int, time() - t_start)
println("\nDone in $(elapsed÷60)m$(elapsed%60)s.")
println("  Metrics : $metrics_file")
println("  Slices  : $slices_file")
println("  Near pts: $near_file")

# ── Summary table ──────────────────────────────────────────────────────────────
for (label, field) in [("RMSE (raw)",              :rmse),
                       ("RMSE (corr)",              :rmse_corr),
                       ("Coverage-95 (GP σ)",       :cov95),
                       ("Coverage-95 (GP+noise)",   :cov95_adj),
                       ("CRPS (GP+noise)",           :crps),
                       ("MSSE (GP+noise, ≈1=good)", :msse),
                       ("MSSE (GP σ only, ≈1=good)",:msse_gp)]
    println("\n=== $label — mean across seeds ===")
    println(@sprintf("%-10s", "n_pts\\N") *
            join([@sprintf("%10s", _ns_str(N)) for N in N_SHOTS_LIST]))
    for n in N_PTS_LIST
        row = @sprintf("%-10d", n)
        for N in N_SHOTS_LIST
            vals = [getfield(r.metrics, field) for r in all_results
                    if r.n_pts == n && r.N_shots == N && r.metrics !== nothing]
            row *= isempty(vals) ? "         —" : @sprintf("%10.4f", mean(vals))
        end
        println(row)
    end
end

rmprocs(workers())
