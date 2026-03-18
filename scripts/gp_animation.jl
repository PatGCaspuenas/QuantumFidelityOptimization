
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Random
using Statistics
using LinearAlgebra
using Printf
using GaussianProcesses
using Distributions
using Plots
using Measures

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode


const CFG = (
    t=100.0,           # gate time (µs)
    seed=12345,           # RNG seed for reproducibility
    n_dense=120,             # points in true curve
    sb_span_kHz=2.5,            # ±2.5 × 2π kHz sweep window
    n_samples=20,              # total samples to collect during animation
    n_init=4,               # number of random initial samples
    gif_fps=4,               # frames per second in output GIF
    gp_noise=1e-4,
    outfile=joinpath(@__DIR__, "plots", "gp_animation.gif"),
)


"""
    fit_and_predict_1d(fsb_samples, fid_samples, fsb_grid; noise=1e-4)

Fit a Matern-3/2 GP to the 1-D data `(fsb_samples, fid_samples)` and return
`(μ_grid, σ_grid)` on `fsb_grid` in the *original* (unscaled) fidelity units.
"""
function fit_and_predict_1d(fsb_samples::Vector{Float64},
    fid_samples::Vector{Float64},
    fsb_grid::AbstractVector{Float64};
    noise::Float64=1e-4)
    n = length(fsb_samples)
    @assert n == length(fid_samples) "length mismatch"
    @assert n >= 2 "need ≥ 2 observations"

    x_μ = mean(fsb_samples)
    x_σ = max(std(fsb_samples), 1e-12)
    y_μ = mean(fid_samples)
    y_σ = max(std(fid_samples), 1e-12)

    X_std = reshape((fsb_samples .- x_μ) ./ x_σ, 1, :)
    y_std = (fid_samples .- y_μ) ./ y_σ

    kern = Matern(3 / 2, [0.0], 0.0)
    gp = GP(X_std, y_std, MeanZero(), kern, log(noise))

    try
        optimize!(gp; noise=false)
    catch

    end


    xg_std = reshape((collect(fsb_grid) .- x_μ) ./ x_σ, 1, :)
    μ_std, σ2_std = predict_f(gp, xg_std)

    μ = y_μ .+ y_σ .* μ_std
    σ = y_σ .* sqrt.(max.(σ2_std, 0.0))

    return μ, σ
end


function main(cfg=CFG)
    rng = MersenneTwister(cfg.seed)

    println("Computing baseline (ideal) parameters...")
    base = CalibrationCode.ideal(cfg.t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A
    @printf "  f_cl0 = %.6e Hz\n" f_cl0
    @printf "  f_sb0 = %.6e Hz\n" f_sb0
    @printf "  A0    = %.6e\n" A0
    @printf "  fid0  = %.6f\n" base.fid

    sb_span_Hz = cfg.sb_span_kHz * 1e3 * 2π
    fsb_grid = range(f_sb0 - sb_span_Hz, f_sb0 + sb_span_Hz; length=cfg.n_dense)
    fsb_offset_kHz = collect((fsb_grid .- f_sb0) ./ (2π * 1e3))

    println("Evaluating true fidelity on dense grid ($(cfg.n_dense) pts)...")
    fid_true = map(fsb -> CalibrationCode.Q_det(cfg.t, f_cl0, fsb, A0), fsb_grid)
    println("  done. max fidelity = $(round(maximum(fid_true), digits=5))")

    init_offsets = (rand(rng, cfg.n_init) .- 0.5) .* (2 * 0.85 * sb_span_Hz)
    init_fsb = collect(init_offsets .+ f_sb0)

    extra_fsb = [f_sb0 + (rand(rng) * 2 - 1) * sb_span_Hz
                 for _ in 1:(cfg.n_samples-cfg.n_init)]

    sample_fsb_schedule = vcat(init_fsb, extra_fsb)

    println("Evaluating fidelity at sample points...")
    sample_fidelities = map(fsb -> CalibrationCode.Q_det(cfg.t, f_cl0, fsb, A0),
        sample_fsb_schedule)
    println("  done.")

    outdir = dirname(cfg.outfile)
    isdir(outdir) || mkdir(outdir)

    col_truth = RGB(0.15, 0.15, 0.15)
    col_mean = RGB(0.20, 0.44, 0.69)
    col_ci = RGBA(0.20, 0.44, 0.69, 0.20)
    col_sample = RGB(0.86, 0.37, 0.12)

    ylim_pad = 0.05
    y_lo = max(0.0, minimum(fid_true) - ylim_pad)
    y_hi = min(1.0, maximum(fid_true) + ylim_pad)

    println("Building animation ($(cfg.n_samples) frames)...")
    anim = @animate for frame in cfg.n_init:cfg.n_samples
        fsb_obs = sample_fsb_schedule[1:frame]
        fid_obs = sample_fidelities[1:frame]
        x_obs_kHz = (fsb_obs .- f_sb0) ./ (2π * 1e3)

        # GP fit & predict
        μ_gp, σ_gp = fit_and_predict_1d(
            fsb_obs, fid_obs, fsb_grid; noise=cfg.gp_noise)

        ci_lo = μ_gp .- 1.96 .* σ_gp
        ci_hi = μ_gp .+ 1.96 .* σ_gp

        p = plot(;
            xlabel="f_sb offset  (2π kHz)",
            ylabel="Fidelity",
            title="GP Regression: f_sb vs Fidelity   [n = $frame]",
            xlims=(fsb_offset_kHz[1], fsb_offset_kHz[end]),
            ylims=(y_lo, y_hi),
            legend=:bottomright,
            size=(800, 500),
            dpi=120,
            left_margin=8mm,
            bottom_margin=8mm,
            grid=true,
            gridalpha=0.3,
            framestyle=:box,
        )

        plot!(p, fsb_offset_kHz, ci_hi;
            fillrange=ci_lo,
            fillcolor=col_ci,
            linewidth=0,
            label="95% CI")

        plot!(p, fsb_offset_kHz, fid_true;
            color=col_truth,
            linewidth=2,
            linestyle=:dash,
            label="True fidelity (Q_det)")

        plot!(p, fsb_offset_kHz, μ_gp;
            color=col_mean,
            linewidth=2.5,
            label="GP posterior mean")
        scatter!(p, x_obs_kHz, fid_obs;
            color=col_sample,
            markersize=6,
            markerstroke=stroke(1, :white),
            label="Observations (n=$frame)")

        p
    end

    gif(anim, cfg.outfile; fps=cfg.gif_fps)
    println("Saved → $(cfg.outfile)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
