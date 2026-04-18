# scripts/ms_sensitivity.jl
#
# Recreate and extend the MS sensitivity scans from Gerster et al. Fig. 6.
#
# Phase-convention note:
# The shared `build_closed_loop_ms_sequence()` helper realizes a closed-loop
# `MS_φ(θ)` pulse by setting `(phi_1, phi_2) = (2φ, 0)` and then injecting an
# accumulated shared `Δϕ/2` offset on every subsequent pulse. This reduces to
# the validated paper-matching conventions for:
#   * `3 × MS₀(π/2)` and
#   * `MS₀(π/2)` followed by `MS_{π/4}(π/2)`.
# Using the source helper keeps the plot and Jacobian search on the same
# pulse-construction and IonSim evolution path.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode
using CairoMakie

const CC = CalibrationCode
const ERROR_RANGE_HZ = 2.0e3
const OMEGA_REFINE_MIN = 0.995
const OMEGA_REFINE_MAX = 1.03
const OMEGA_REFINE_STEPS = 71
const POP_COLORS = (gg=:forestgreen, ee=:royalblue3, mid=:darkorange2)
const SEARCH_SEQUENCE_NAME = :seq_C
const SEARCH_SEQUENCE_LABEL = "Best Jacobian probe"
const SEARCH_SEQUENCE_STYLE = :dashdot
const SEARCH_RESULT_PATH = joinpath(@__DIR__, "..", "data", "ms_sequence_search_result.jl")

function build_sequence(spec, t, f_cl, Δ, I_pi2;
                        omega_ratio::Float64=1.0,
                        relative_phase::Float64=0.0,
                        phase_drift::Float64=0.0)
    return CC.build_closed_loop_ms_sequence(
        t, f_cl, Δ, I_pi2, spec.subgates;
        omega_ratio=omega_ratio,
        relative_phase=relative_phase,
        phase_drift=phase_drift)
end

function scan_sequence(make_pulses, grid)
    gg = Vector{Float64}(undef, length(grid))
    ee = similar(gg)
    mid = similar(gg)
    for (i, x) in enumerate(grid)
        pops = CC.populations_ms_sequence(make_pulses(x))
        gg[i] = pops.gg
        ee[i] = pops.ee
        mid[i] = pops.eg + pops.ge
    end
    return (gg=gg, ee=ee, mid=mid)
end

scan_rabi(Ω_ratios, t, f_cl, Δ, I0, spec) = scan_sequence(Ω_ratios) do r
    build_sequence(spec, t, f_cl, Δ, I0; omega_ratio=r)
end

scan_relative_phase(δ_grid, t, f_cl, Δ, I, spec) = scan_sequence(δ_grid) do δ
    build_sequence(spec, t, f_cl, Δ, I; relative_phase=δ)
end

scan_sideband(Δ_error_grid, t, f_cl, Δ0, I, spec) = scan_sequence(Δ_error_grid) do Δerr
    build_sequence(spec, t, f_cl, Δ0 + Δerr, I)
end

scan_centerline(fcl_error_grid, t, f_cl0, Δ, I, spec) = scan_sequence(fcl_error_grid) do ferr
    build_sequence(spec, t, f_cl0 + ferr, Δ, I)
end

function format_pi_multiple(x::Float64)
    scaled = round(x / π; digits=3)
    return iszero(scaled) ? "0" : string(scaled, "π")
end

function describe_subgates(subgates)
    parts = String[]
    for subgate in subgates
        push!(parts, "MS_$(format_pi_multiple(subgate.phi))($(format_pi_multiple(subgate.theta)))")
    end
    return join(parts, " · ")
end

restore_subgates(subgates_data) = [CC.MSSubgate(item.theta, item.phi) for item in subgates_data]

restore_candidate(candidate) = merge(candidate, (subgates=restore_subgates(candidate.subgates),))

function load_saved_search_result(path)
    data = include(path)
    return (
        t=data.t,
        candidate_count=data.candidate_count,
        filtered_candidate_count=data.filtered_candidate_count,
        matching_baseline=data.matching_baseline,
        omega_center=data.omega_center,
        I_center=data.I_center,
        best_overall=restore_candidate(data.best_overall),
        best_new=isnothing(data.best_new) ? nothing : restore_candidate(data.best_new),
        top_results=[restore_candidate(candidate) for candidate in data.top_results],
        distinct=isnothing(data.matching_baseline),
    )
end

function resolve_search_result(t, omega_ratio_grid)
    if isfile(SEARCH_RESULT_PATH)
        loaded = load_saved_search_result(SEARCH_RESULT_PATH)
        if isapprox(loaded.t, t; atol=0.0, rtol=0.0)
            @info "Loaded cached MS Jacobian search" path=SEARCH_RESULT_PATH filtered=loaded.filtered_candidate_count
            return loaded
        end
    end
    @info "Running MS Jacobian search from plot script"
    return CC.search_ms_calibration_sequence(t; omega_ratio_grid=omega_ratio_grid)
end

function build_sequence_registry(search_result)
    specs = copy(CC.default_ms_sequence_specs())
    if search_result.distinct
        push!(specs, CC.MSSequenceSpec(
            name=SEARCH_SEQUENCE_NAME,
            label=SEARCH_SEQUENCE_LABEL,
            subgates=search_result.best_overall.subgates,
            linestyle=SEARCH_SEQUENCE_STYLE,
        ))
    end
    return specs
end

function draw_scan!(ax, xvals, results, specs)
    for (result, spec) in zip(results, specs)
        lines!(ax, xvals, result.gg; color=POP_COLORS.gg, linestyle=spec.linestyle, linewidth=3)
        lines!(ax, xvals, result.ee; color=POP_COLORS.ee, linestyle=spec.linestyle, linewidth=3)
        lines!(ax, xvals, result.mid; color=POP_COLORS.mid, linestyle=spec.linestyle, linewidth=3)
    end
    ylims!(ax, 0.0, 1.0)
    return ax
end

function add_legends!(fig, specs)
    pop_legend = Legend(
        fig[0, 1],
        [
            LineElement(color=POP_COLORS.gg, linewidth=4),
            LineElement(color=POP_COLORS.ee, linewidth=4),
            LineElement(color=POP_COLORS.mid, linewidth=4),
        ],
        ["gg", "ee", "eg + ge"];
        orientation=:horizontal,
        framevisible=false,
        tellheight=true,
        tellwidth=false,
    )
    seq_legend = Legend(
        fig[0, 2],
        [LineElement(color=:black, linestyle=spec.linestyle, linewidth=4) for spec in specs],
        [spec.label for spec in specs];
        orientation=:horizontal,
        framevisible=false,
        tellheight=true,
        tellwidth=false,
    )
    pop_legend.halign = :left
    seq_legend.halign = :right
    return fig
end

function main()
    t = 100.0
    omega_ratio_grid = collect(range(OMEGA_REFINE_MIN, OMEGA_REFINE_MAX; length=OMEGA_REFINE_STEPS))
    search_result = resolve_search_result(t, omega_ratio_grid)
    init = CC.ideal(t)
    f_cl_ideal = init.f_cl
    Δ_ideal = init.f_sb
    I_center = search_result.I_center
    specs = build_sequence_registry(search_result)

    Ω_ratios = collect(range(0.8, 1.2; length=41))
    Δφ_grid = collect(range(-π / 2, π / 2; length=41))
    Δ_error_grid = collect(range(-ERROR_RANGE_HZ, ERROR_RANGE_HZ; length=41))
    fcl_error_grid = collect(range(-ERROR_RANGE_HZ, ERROR_RANGE_HZ; length=41))

    @info "Using refined 3-gate Omega center" ratio=search_result.omega_center.ratio fidelity=search_result.omega_center.fid
    @info "Best MS Jacobian candidate" matched_baseline=search_result.matching_baseline score=search_result.best_overall.score sequence=describe_subgates(search_result.best_overall.subgates)
    if !isnothing(search_result.best_new)
        @info "Best distinct non-baseline candidate" score=search_result.best_new.score sequence=describe_subgates(search_result.best_new.subgates)
    end

    phase_results = [scan_relative_phase(Δφ_grid, t, f_cl_ideal, Δ_ideal, I_center, spec) for spec in specs]

    CairoMakie.activate!(type="png")
    fig = Figure(size=(700, 500), figure_padding=(24, 36, 18, 20))
    add_legends!(fig, specs)

    ax_phase = Axis(fig[1, 1];
        xlabel="Relative phase δ (π)",
        ylabel="Expectation value",
        xlabelpadding=10,
        ylabelpadding=10,
        title="Relative Phase Error (φ₁ − φ₂)",
    )

    draw_scan!(ax_phase, Δφ_grid ./ π, phase_results, specs)

    outdir = joinpath(@__DIR__, "..", "figures")
    mkpath(outdir)
    outpath = joinpath(outdir, "ms_sensitivity.png")
    save(outpath, fig)
    @info "Saved figure" outpath
    return fig
end

main()
