# scripts/exploratory_mf_clean.jl
#
# Exploratory figures for calibration fidelity:
# 1) Fidelity vs N (Q_noisy) + nominal Q_det reference
# 2) 2D landscapes:
#       (a) f_cl vs f_sb @ A0
#       (b) f_cl vs A    @ f_sb0
#       (c) f_sb vs A    @ f_cl0
# 3) 1D slices (det + noisy mean±SEM) over f_cl, f_sb, A
#
# Run:
#   julia --project=. scripts/exploratory_mf_clean.jl
#
# Outputs saved to `outdir`.

using Random
using Statistics
using Plots
using Measures
using CalibrationCode

# Optional: silence IonSim / other package stderr chatter
redirect_stderr(devnull)

# -------------------------
# Configuration
# -------------------------

const CFG = (
    seed = 42,
    t = 100.0,

    outdir = "figures/exploratory",

    # (1) fidelity vs N
    Ns = [1, 5, 10, 20, 50, 100, 200, 500, 1000],
    n_rep_vsN = 30,

    # (2) 2D landscapes windows
    Δf_cl_land = 2e3,
    Δf_sb_land = 2e3,
    ΔA_land    = 8e4,
    nx_land = 41,
    ny_land = 41,
    Ns_land = [10, 100, 1000],
    n_rep_land = 1,

    # (3) 1D slices windows
    Δf_slice = 2e3,
    ΔA_slice = 8e4,
    n1_slice = 201,
    Ns_slice = [10, 100, 1000],
    n_rep_slice = 30,

    # plotting / guards
    clamp01 = true,
)

# -------------------------
# Helpers
# -------------------------

@inline _clamp01(x) = clamp(x, 0.0, 1.0)

function _safe_eval(f; verbose=false)
    try
        return f()
    catch err
        verbose && @warn "Evaluation failed" err
        return NaN
    end
end

function _mean_sem(vals)
    μ = mean(vals)
    s = std(vals)
    sem = s / sqrt(length(vals))
    return μ, s, sem
end

# Build 2D map Z with rows=y, cols=x (fits contourf(x, y, Z))
function _map2d(xgrid, ygrid, f; verbose=false)
    Z = Matrix{Float64}(undef, length(ygrid), length(xgrid))
    for (iy, yv) in enumerate(ygrid)
        verbose && println("row $iy / $(length(ygrid))")
        for (ix, xv) in enumerate(xgrid)
            Z[iy, ix] = f(xv, yv)
        end
    end
    return Z
end

# common clims from one or more Zs
function _clims(Zs...)
    finite_vals = filter(isfinite, vcat((vec(Z) for Z in Zs)...))
    isempty(finite_vals) && return (0.0, 1.0)
    return (minimum(finite_vals), maximum(finite_vals))
end

# -------------------------
# Main
# -------------------------

function main(; cfg=CFG, verbose=false)
    rng = MersenneTwister(cfg.seed)
    mkpath(cfg.outdir)

    t = cfg.t

    # Baseline from ideal evolution
    base = CalibrationCode.ideal(t)
    f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

    println("Baseline (t=$(t) μs):")
    println(" f_cl0 = $(round(f_cl0, digits=3))")
    println(" f_sb0 = $(round(f_sb0, digits=3))")
    println(" A0    = $(round(A0,    digits=3))")
    println(" fid0  = $(round(base.fid, digits=6))")

    # convenience closures
    Qdet(fcl, fsb, A) = CalibrationCode.Q_det(t, fcl, fsb, A)
    Qnoisy(fcl, fsb, A, N) = CalibrationCode.Q_noisy(t, fcl, fsb, A; N=N)

    clampfun = cfg.clamp01 ? _clamp01 : identity

    # =========================================================
    # 1) Fidelity vs N at nominal point
    # =========================================================
    det0 = clampfun(_safe_eval(() -> Qdet(f_cl0, f_sb0, A0); verbose=verbose))

    means = Float64[]; sems = Float64[]
    for N in cfg.Ns
        vals = Vector{Float64}(undef, cfg.n_rep_vsN)
        for i in 1:cfg.n_rep_vsN
            vals[i] = clampfun(_safe_eval(() -> Qnoisy(f_cl0, f_sb0, A0, N); verbose=verbose))
        end
        μ, _, sem = _mean_sem(vals)
        push!(means, μ)
        push!(sems, sem)
    end

    pN = plot(cfg.Ns, means; ribbon=sems, lw=2, marker=:circle,
              xlabel="N", ylabel="Fidelity",
              title="Q_noisy vs N at (f_cl0,f_sb0,A0)",
              label="Q_noisy mean ± SEM")
    hline!(pN, [det0]; lw=2, ls=:dash, label="Q_det (nominal)")

    savefig(pN, joinpath(cfg.outdir, "fidelity_vs_N.png"))

    # =========================================================
    # 2) 2D landscapes
    #   (a) f_cl vs f_sb @ A0
    #   (b) f_cl vs A    @ f_sb0
    #   (c) f_sb vs A    @ f_cl0
    # =========================================================
    fcl_grid = range(f_cl0 - cfg.Δf_cl_land, f_cl0 + cfg.Δf_cl_land; length=cfg.nx_land)
    fsb_grid = range(f_sb0 - cfg.Δf_sb_land, f_sb0 + cfg.Δf_sb_land; length=cfg.ny_land)
    A_grid   = range(A0   - cfg.ΔA_land,    A0   + cfg.ΔA_land;    length=cfg.ny_land)

    # (a) f_cl vs f_sb @ A0
    Zdet_fcl_fsb = _map2d(fcl_grid, fsb_grid, (fcl, fsb) ->
        clampfun(_safe_eval(() -> Qdet(fcl, fsb, A0); verbose=verbose))
    )
    cmin, cmax = _clims(Zdet_fcl_fsb)
    pdet = contourf(fcl_grid, fsb_grid, Zdet_fcl_fsb;
                    xlabel="f_cl (Hz)", ylabel="f_sb (Hz)",
                    title="Q_det landscape (A=A0)",
                    colorbar_title="F", clims=(cmin, cmax))
    savefig(pdet, joinpath(cfg.outdir, "landscape_det_fcl_fsb.png"))

    plots_land = Any[pdet]
    for N in cfg.Ns_land
        Z = _map2d(fcl_grid, fsb_grid, (fcl, fsb) -> begin
            vals = [clampfun(_safe_eval(() -> Qnoisy(fcl, fsb, A0, N); verbose=verbose))
                    for _ in 1:cfg.n_rep_land]
            mean(vals)
        end)
        cminN, cmaxN = _clims(Zdet_fcl_fsb, Z)
        p = contourf(fcl_grid, fsb_grid, Z;
                     xlabel="f_cl (Hz)", ylabel="f_sb (Hz)",
                     title="Q_noisy mean (N=$N, A=A0)",
                     colorbar_title="F", clims=(cminN, cmaxN))
        push!(plots_land, p)
        savefig(p, joinpath(cfg.outdir, "landscape_noisy_fcl_fsb_N$(N).png"))
    end
    fig_land = plot(plots_land...; layout=(1, length(plots_land)), size=(350 * length(plots_land), 320))
    savefig(fig_land, joinpath(cfg.outdir, "landscape_comparison_fcl_fsb.png"))

    # (b) f_cl vs A @ f_sb0
    Zdet_fcl_A = _map2d(fcl_grid, A_grid, (fcl, A) ->
        clampfun(_safe_eval(() -> Qdet(fcl, f_sb0, A); verbose=verbose))
    )
    cmin2, cmax2 = _clims(Zdet_fcl_A)
    pdet2 = contourf(fcl_grid, A_grid, Zdet_fcl_A;
                     xlabel="f_cl (Hz)", ylabel="A",
                     title="Q_det landscape (f_sb=f_sb0)",
                     colorbar_title="F", clims=(cmin2, cmax2))
    savefig(pdet2, joinpath(cfg.outdir, "landscape_det_fcl_A.png"))

    plots_land2 = Any[pdet2]
    for N in cfg.Ns_land
        Z = _map2d(fcl_grid, A_grid, (fcl, A) -> begin
            vals = [clampfun(_safe_eval(() -> Qnoisy(fcl, f_sb0, A, N); verbose=verbose))
                    for _ in 1:cfg.n_rep_land]
            mean(vals)
        end)
        cminN, cmaxN = _clims(Zdet_fcl_A, Z)
        p = contourf(fcl_grid, A_grid, Z;
                     xlabel="f_cl (Hz)", ylabel="A",
                     title="Q_noisy mean (N=$N, f_sb=f_sb0)",
                     colorbar_title="F", clims=(cminN, cmaxN))
        push!(plots_land2, p)
        savefig(p, joinpath(cfg.outdir, "landscape_noisy_fcl_A_N$(N).png"))
    end
    fig_land2 = plot(plots_land2...; layout=(1, length(plots_land2)), size=(350 * length(plots_land2), 320))
    savefig(fig_land2, joinpath(cfg.outdir, "landscape_comparison_fcl_A.png"))

    # (c) f_sb vs A @ f_cl0
    Zdet_fsb_A = _map2d(fsb_grid, A_grid, (fsb, A) ->
        clampfun(_safe_eval(() -> Qdet(f_cl0, fsb, A); verbose=verbose))
    )
    cmin3, cmax3 = _clims(Zdet_fsb_A)
    pdet3 = contourf(fsb_grid, A_grid, Zdet_fsb_A;
                     xlabel="f_sb (Hz)", ylabel="A",
                     title="Q_det landscape (f_cl=f_cl0)",
                     colorbar_title="F", clims=(cmin3, cmax3))
    savefig(pdet3, joinpath(cfg.outdir, "landscape_det_fsb_A.png"))

    plots_land3 = Any[pdet3]
    for N in cfg.Ns_land
        Z = _map2d(fsb_grid, A_grid, (fsb, A) -> begin
            vals = [clampfun(_safe_eval(() -> Qnoisy(f_cl0, fsb, A, N); verbose=verbose))
                    for _ in 1:cfg.n_rep_land]
            mean(vals)
        end)
        cminN, cmaxN = _clims(Zdet_fsb_A, Z)
        p = contourf(fsb_grid, A_grid, Z;
                     xlabel="f_sb (Hz)", ylabel="A",
                     title="Q_noisy mean (N=$N, f_cl=f_cl0)",
                     colorbar_title="F", clims=(cminN, cmaxN))
        push!(plots_land3, p)
        savefig(p, joinpath(cfg.outdir, "landscape_noisy_fsb_A_N$(N).png"))
    end
    fig_land3 = plot(plots_land3...; layout=(1, length(plots_land3)), size=(350 * length(plots_land3), 320))
    savefig(fig_land3, joinpath(cfg.outdir, "landscape_comparison_fsb_A.png"))

    # =========================================================
    # 3) 1D slices: det + noisy mean±SEM over f_cl, f_sb, A
    # =========================================================
    function slice_over_fcl(N)
        xs = range(f_cl0 - cfg.Δf_slice, f_cl0 + cfg.Δf_slice; length=cfg.n1_slice)
        det = [clampfun(_safe_eval(() -> Qdet(fcl, f_sb0, A0); verbose=verbose)) for fcl in xs]
        μs  = similar(det); sem = similar(det)
        for (i, fcl) in enumerate(xs)
            vals = [clampfun(_safe_eval(() -> Qnoisy(fcl, f_sb0, A0, N); verbose=verbose))
                    for _ in 1:cfg.n_rep_slice]
            μ, _, se = _mean_sem(vals)
            μs[i] = μ
            sem[i] = se
        end
        return xs, det, μs, sem
    end

    function slice_over_fsb(N)
        xs = range(f_sb0 - cfg.Δf_slice, f_sb0 + cfg.Δf_slice; length=cfg.n1_slice)
        det = [clampfun(_safe_eval(() -> Qdet(f_cl0, fsb, A0); verbose=verbose)) for fsb in xs]
        μs  = similar(det); sem = similar(det)
        for (i, fsb) in enumerate(xs)
            vals = [clampfun(_safe_eval(() -> Qnoisy(f_cl0, fsb, A0, N); verbose=verbose))
                    for _ in 1:cfg.n_rep_slice]
            μ, _, se = _mean_sem(vals)
            μs[i] = μ
            sem[i] = se
        end
        return xs, det, μs, sem
    end

    function slice_over_A(N)
        xs = range(A0 - cfg.ΔA_slice, A0 + cfg.ΔA_slice; length=cfg.n1_slice)
        det = [clampfun(_safe_eval(() -> Qdet(f_cl0, f_sb0, A); verbose=verbose)) for A in xs]
        μs  = similar(det); sem = similar(det)
        for (i, A) in enumerate(xs)
            vals = [clampfun(_safe_eval(() -> Qnoisy(f_cl0, f_sb0, A, N); verbose=verbose))
                    for _ in 1:cfg.n_rep_slice]
            μ, _, se = _mean_sem(vals)
            μs[i] = μ
            sem[i] = se
        end
        return xs, det, μs, sem
    end

    # f_cl slice
    p1 = plot(; xlabel="f_cl (Hz)", ylabel="Fidelity", title="Slice over f_cl (f_sb0,A0)")
    x, det, _, _ = slice_over_fcl(cfg.Ns_slice[end])
    plot!(p1, x, det; lw=2, ls=:dash, label="Q_det")
    for N in cfg.Ns_slice
        x, _, μ, se = slice_over_fcl(N)
        plot!(p1, x, μ; ribbon=se, lw=2, label="Q_noisy N=$N")
    end
    savefig(p1, joinpath(cfg.outdir, "slice_fcl.png"))

    # f_sb slice
    p2 = plot(; xlabel="f_sb (Hz)", ylabel="Fidelity", title="Slice over f_sb (f_cl0,A0)")
    x, det, _, _ = slice_over_fsb(cfg.Ns_slice[end])
    plot!(p2, x, det; lw=2, ls=:dash, label="Q_det")
    for N in cfg.Ns_slice
        x, _, μ, se = slice_over_fsb(N)
        plot!(p2, x, μ; ribbon=se, lw=2, label="Q_noisy N=$N")
    end
    savefig(p2, joinpath(cfg.outdir, "slice_fsb.png"))

    # A slice
    p3 = plot(; xlabel="A", ylabel="Fidelity", title="Slice over A (f_cl0,f_sb0)")
    x, det, _, _ = slice_over_A(cfg.Ns_slice[end])
    plot!(p3, x, det; lw=2, ls=:dash, label="Q_det")
    for N in cfg.Ns_slice
        x, _, μ, se = slice_over_A(N)
        plot!(p3, x, μ; ribbon=se, lw=2, label="Q_noisy N=$N")
    end
    savefig(p3, joinpath(cfg.outdir, "slice_A.png"))

    println("Saved figures to: $(cfg.outdir)")
    println("  fidelity_vs_N.png")
    println("  landscape_det_fcl_fsb.png")
    println("  landscape_noisy_fcl_fsb_N*.png")
    println("  landscape_comparison_fcl_fsb.png")
    println("  landscape_det_fcl_A.png")
    println("  landscape_noisy_fcl_A_N*.png")
    println("  landscape_comparison_fcl_A.png")
    println("  landscape_det_fsb_A.png")
    println("  landscape_noisy_fsb_A_N*.png")
    println("  landscape_comparison_fsb_A.png")
    println("  slice_fcl.png")
    println("  slice_fsb.png")
    println("  slice_A.png")

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
