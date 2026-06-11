#!/usr/bin/env julia
#
# Data generator for paper figure figures/paper/figure_fcl_amp_n100_gpzoom.png.
# Writes the GP-slice and true-score caches the figure reads:
#   data/gp_slices/scale05_selectedN_full_l1_avg5_gp_slices_iters10_30_50_100.csv
#   data/score_cache/scale05_true_2ms_full_l1_slices_freqspan10_grid201.csv

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using CairoMakie
using Distributions
using LaTeXStrings
using Printf
using Random
using Statistics
using StatsBase

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "src", "CalibrationCode.jl"))

const DATA_DIR = joinpath(REPO_ROOT, "data")
const FIGURE_DIR = joinpath(REPO_ROOT, "figures", "paper")
const CACHE_DIR = joinpath(DATA_DIR, "score_cache")
const GP_SLICE_DIR = joinpath(DATA_DIR, "gp_slices")
const PX_PER_UNIT = parse(Float64, get(ENV, "PLOT_PX_PER_UNIT", "2"))

const SCORE_MODE = Symbol(lowercase(get(ENV, "QFO_SCORE_MODE", "odd_penalty")))
SCORE_MODE in (:odd_penalty, :full_l1) ||
    throw(ArgumentError("QFO_SCORE_MODE must be odd_penalty or full_l1"))
const GP_YMIN = parse(Float64, get(ENV, "QFO_GP_YMIN", "-0.02"))
const GP_YMAX = parse(Float64, get(ENV, "QFO_GP_YMAX", "1.04"))
const GP_OUTPUT_SUFFIX = get(ENV, "QFO_GP_OUTPUT_SUFFIX", "")

const PLOT_SNAPSHOT_ITERS = [10, 30, 100]
const DATA_SNAPSHOT_ITERS = [10, 30, 50, 100]
const AXES = (:fcl, :fsb, :amp)
const N_LABELS = ("100", "1000", "10000", "100000", "Inf")

const N_INIT = 12
const BOUND_SCALE = 0.5
const N_SEEDS_FOR_AVERAGE = parse(Int, get(ENV, "QFO_GP_AVG_SEEDS", "5"))
const GRID_N = parse(Int, get(ENV, "MEDIAN_GP_GRID_N", "201"))
const GP_N_RESTARTS = parse(Int, get(ENV, "MEDIAN_GP_N_RESTARTS", "6"))

const TRACE_FREQ_SPAN_KHZ = 10.0
const TRACE_FREQ_SPAN_HZ = TRACE_FREQ_SPAN_KHZ * 1e3
const T_GATE = 100.0
const BASE = CalibrationCode.ideal(T_GATE)
const F_CL0 = Float64(BASE.f_cl)
const F_SB0 = Float64(BASE.f_sb)
const A0 = Float64(BASE.A)
const SPAN_FCL = TRACE_FREQ_SPAN_HZ
const SPAN_FSB = TRACE_FREQ_SPAN_HZ
const SPAN_A = 1.2 * A0 - A0

const COLORS = Dict{String,Any}(
    "100" => colorant"#EE6677",
    "1000" => colorant"#CCBB44",
    "10000" => colorant"#228833",
    "100000" => colorant"#4477AA",
    "Inf" => colorant"#777777",
)

const LABELS = Dict(
    "100" => "N = 100",
    "1000" => "N = 1000",
    "10000" => "N = 10k",
    "100000" => "N = 100k",
    "Inf" => "Inf shots",
)

function trace_dir(score_mode::Symbol, n_label::String)
    if score_mode == :odd_penalty
        if n_label == "Inf"
            return joinpath(DATA_DIR, "traces_freqspan10_bound050_NInf_lhs12_restart_fullbudget100_100seeds")
        end
        return joinpath(DATA_DIR, "traces_freqspan10_bound050_N$(n_label)_nostop100_stream_100seeds")
    end
    return joinpath(DATA_DIR, "traces_freqspan10_bound050_full_l1_N$(n_label)_nostop100_stream_100seeds")
end

n_value(n_label::String) = n_label == "Inf" ? Inf : parse(Int, n_label)

function csv_field(value)
    if value isa AbstractFloat
        isnan(value) && return "NaN"
        isinf(value) && return value > 0 ? "Inf" : "-Inf"
        return @sprintf("%.17g", value)
    elseif value isa Bool
        return value ? "true" : "false"
    end
    return string(value)
end

function write_namedtuple_csv(path::String, rows)
    isempty(rows) && error("Refusing to write empty CSV: $path")
    mkpath(dirname(path))
    header = collect(keys(rows[1]))
    open(path, "w") do io
        println(io, join(String.(header), ","))
        for row in rows
            println(io, join((csv_field(getfield(row, key)) for key in header), ","))
        end
    end
    return path
end

function read_csv(path::String)
    open(path, "r") do io
        header = split(readline(io), ',')
        rows = Vector{Dict{String,String}}()
        for line in eachline(io)
            isempty(strip(line)) && continue
            cols = split(line, ',')
            push!(rows, Dict(header[i] => get(cols, i, "") for i in eachindex(header)))
        end
        return rows
    end
end

parse_float(value::AbstractString) = parse(Float64, value)
parse_bool(value::AbstractString) = lowercase(strip(value)) in ("1", "true", "yes", "on")

axis_name(axis::Symbol) = axis === :fcl ? "fcl" :
                          axis === :fsb ? "fsb" :
                          axis === :amp ? "amp" :
                          error("Unsupported axis $axis")

axis_label(axis::Symbol) = axis === :fcl ? L"\Delta f_{\mathrm{cl}}\;(\mathrm{kHz})" :
                           axis === :fsb ? L"\Delta f_{\mathrm{sb}}\;(\mathrm{kHz})" :
                           L"\Delta A/A_{\mathrm{opt}}"

axis_title(axis::Symbol) = axis === :fcl ? L"f_{\mathrm{cl}}" :
                           axis === :fsb ? L"f_{\mathrm{sb}}" :
                           L"A"

axis_unit(axis::Symbol) = axis === :amp ? "DeltaA_over_Aopt" : "kHz"

function axis_point(axis::Symbol, x_u::Float64)
    axis === :fcl && return [x_u, 0.0, 0.0]
    axis === :fsb && return [0.0, x_u, 0.0]
    axis === :amp && return [0.0, 0.0, x_u]
    error("Unsupported axis $axis")
end

function axis_physical(axis::Symbol, x_u::Float64)
    axis === :fcl && return SPAN_FCL * x_u / 1e3
    axis === :fsb && return SPAN_FSB * x_u / 1e3
    axis === :amp && return SPAN_A * x_u / A0
    error("Unsupported axis $axis")
end

function u_to_params(u)
    return (
        fcl=F_CL0 + SPAN_FCL * Float64(u[1]),
        fsb=F_SB0 + SPAN_FSB * Float64(u[2]),
        A=A0 + SPAN_A * Float64(u[3]),
    )
end

function ms2_weights(u)
    p = u_to_params(u)
    subgates = [
        CalibrationCode.MSSubgate(pi / 2, 0.0),
        CalibrationCode.MSSubgate(pi / 2, 0.0),
    ]
    pulses = CalibrationCode.build_closed_loop_ms_sequence(T_GATE, p.fcl, p.fsb, p.A, subgates)
    pops = CalibrationCode.populations_ms_sequence(pulses)
    weights = Float64[
        max(pops.gg, 0.0),
        max(pops.eg, 0.0),
        max(pops.ge, 0.0),
        max(pops.ee, 0.0),
    ]
    weights ./= sum(weights)
    return weights
end

function score_from_probabilities(p_ss::Float64, p_sd::Float64,
                                  p_ds::Float64, p_dd::Float64)
    if SCORE_MODE == :odd_penalty
        return clamp(p_dd - p_ss - p_sd - p_ds, 0.0, 1.0)
    end
    l1 = abs(p_ss) + abs(p_sd) + abs(p_ds) + abs(p_dd - 1.0)
    return clamp(1.0 - 0.5 * l1, 0.0, 1.0)
end

uses_legacy_initial_sampling(n_label::String) =
    SCORE_MODE == :odd_penalty && n_label in ("100", "1000")

function score_from_weights(weights::Vector{Float64}, N::Real; legacy_sample::Bool=false)
    n = Float64(N)
    if isinf(n)
        return score_from_probabilities(weights[1], weights[2], weights[3], weights[4])
    end
    n_int = Int(n)
    if legacy_sample
        samples = StatsBase.sample(1:4, StatsBase.Weights(weights), n_int)
        return score_from_probabilities(
            count(==(1), samples) / n_int,
            count(==(2), samples) / n_int,
            count(==(3), samples) / n_int,
            count(==(4), samples) / n_int,
        )
    end
    counts = rand(Distributions.Multinomial(n_int, weights))
    return score_from_probabilities(
        counts[1] / n_int,
        counts[2] / n_int,
        counts[3] / n_int,
        counts[4] / n_int,
    )
end

function score_noise_std(weights::Vector{Float64}, N::Real)
    n = Float64(N)
    isinf(n) && return 0.0
    p_ss, p_sd, p_ds, p_dd = weights
    if SCORE_MODE == :odd_penalty
        raw = p_dd - p_ss - p_sd - p_ds
        var_one = 1.0 - raw * raw
    else
        var_one = p_dd * (1.0 - p_dd)
    end
    return sqrt(max(var_one, 0.0) / n)
end

true_score(u) = score_from_weights(ms2_weights(u), Inf)

function true_score_cache_path()
    score_label = SCORE_MODE == :odd_penalty ? "odd_penalty" : "full_l1"
    return joinpath(CACHE_DIR,
        "scale05_true_2ms_$(score_label)_slices_freqspan10_grid$(GRID_N).csv")
end

function load_or_write_true_score_cache()
    path = true_score_cache_path()
    expected_rows = length(AXES) * GRID_N
    if isfile(path)
        rows = read_csv(path)
        if length(rows) == expected_rows
            println("Loaded true-score cache: $path")
            return rows
        end
    end

    println("Computing true 2MS score slices for cache...")
    grid = collect(range(-BOUND_SCALE, BOUND_SCALE; length=GRID_N))
    rows = NamedTuple[]
    for axis in AXES
        for x_u in grid
            u = axis_point(axis, Float64(x_u))
            push!(rows, (
                score_mode=String(SCORE_MODE),
                axis=axis_name(axis),
                x_u=Float64(x_u),
                x_physical=axis_physical(axis, Float64(x_u)),
                physical_unit=axis_unit(axis),
                score=true_score(u),
            ))
        end
    end
    write_namedtuple_csv(path, rows)
    println("Saved true-score cache: $path")
    return read_csv(path)
end

function true_score_by_axis(cache_rows)
    out = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()
    for axis in AXES
        rs = filter(row -> row["axis"] == axis_name(axis), cache_rows)
        sort!(rs; by=row -> parse_float(row["x_physical"]))
        out[axis] = (
            [parse_float(row["x_physical"]) for row in rs],
            [parse_float(row["score"]) for row in rs],
        )
    end
    return out
end

function seed_from_trace_path(path::String)
    m = match(r"trace_seed(\d+)_ucb\.csv$", basename(path))
    m === nothing && error("Could not parse seed from $path")
    return parse(Int, m.captures[1])
end

function trace_paths(dir::String)
    isdir(dir) || error("Missing trace directory: $dir")
    paths = sort(filter(path -> occursin(r"trace_seed\d+_ucb\.csv$", basename(path)),
                        readdir(dir; join=true)))
    isempty(paths) && error("No trace files in $dir")
    return paths
end

function read_trace(path::String)
    rows = read_csv(path)
    sort!(rows; by=row -> parse(Int, row["iter"]))
    return rows
end

function final_qdet(path::String)
    rows = read_trace(path)
    isempty(rows) && error("Empty trace file: $path")
    return parse_float(rows[end]["q_det_rec"])
end

function selected_seed_rows(n_label::String)
    dir = trace_dir(SCORE_MODE, n_label)
    rows = [(seed=seed_from_trace_path(path), path=path, q_det=final_qdet(path))
            for path in trace_paths(dir)]
    med = median([row.q_det for row in rows])
    sort!(rows; by=row -> abs(row.q_det - med))
    selected = rows[1:min(N_SEEDS_FOR_AVERAGE, length(rows))]
    length(selected) >= N_SEEDS_FOR_AVERAGE ||
        error("Need $N_SEEDS_FOR_AVERAGE traces for $n_label, found $(length(selected))")
    return selected, med, dir
end

function far_enough(x::Vector{Float64}, X::Matrix{Float64}, n::Int; min_dist::Float64=1e-4)
    @inbounds for i in 1:n
        d2 = 0.0
        for j in eachindex(x)
            delta = x[j] - X[j, i]
            d2 += delta * delta
        end
        d2 < min_dist * min_dist && return false
    end
    return true
end

function initial_points(rng::Random.AbstractRNG)
    lb = fill(-BOUND_SCALE, 3)
    ub = fill(BOUND_SCALE, 3)
    cfg = CalibrationCode.PathGuardConfig(init_design=:latin_hypercube)
    return CalibrationCode._path_guard_initial_points(rng, lb, ub, N_INIT, cfg)
end

function append_point!(X::Matrix{Float64}, y::Vector{Float64}, sigma_y::Vector{Float64},
                       write_idx::Int, x::Vector{Float64}, obs::Float64, N::Real)
    far_enough(x, X, write_idx) || return write_idx
    write_idx += 1
    X[:, write_idx] = x
    y[write_idx] = obs
    sigma_y[write_idx] = score_noise_std(ms2_weights(x), N)
    return write_idx
end

function fill_initial_design!(X::Matrix{Float64}, y::Vector{Float64},
                              sigma_y::Vector{Float64}, rng::Random.AbstractRNG,
                              N::Real; legacy_sample::Bool=false)
    write_idx = 0
    for x in initial_points(rng)
        obs = score_from_weights(ms2_weights(x), N; legacy_sample=legacy_sample)
        write_idx = append_point!(X, y, sigma_y, write_idx, x, obs, N)
    end
    return write_idx
end

function reconstruct_training(n_label::String, seed::Int, snapshot_iter::Int)
    N = n_value(n_label)
    legacy_sample = uses_legacy_initial_sampling(n_label)
    Random.seed!(seed)
    rng = MersenneTwister(seed)
    n_cap = N_INIT + snapshot_iter + 2 * N_INIT + 16
    X = Matrix{Float64}(undef, 3, n_cap)
    y = Vector{Float64}(undef, n_cap)
    sigma_y = Vector{Float64}(undef, n_cap)
    write_idx = fill_initial_design!(X, y, sigma_y, rng, N; legacy_sample=legacy_sample)

    for row in read_trace(joinpath(trace_dir(SCORE_MODE, n_label), "trace_seed$(seed)_ucb.csv"))
        iter = parse(Int, row["iter"])
        iter > snapshot_iter && break
        x = [
            parse_float(row["x_acq_u1"]),
            parse_float(row["x_acq_u2"]),
            parse_float(row["x_acq_u3"]),
        ]
        obs = parse_float(row["y_acq"])
        write_idx = append_point!(X, y, sigma_y, write_idx, x, obs, N)

        if parse_bool(row["restart_triggered"]) && iter < snapshot_iter
            restart_seed = parse(Int, row["restart_seed"])
            Random.seed!(restart_seed)
            rng = MersenneTwister(restart_seed)
            write_idx = fill_initial_design!(X, y, sigma_y, rng, N; legacy_sample=legacy_sample)
        end
    end

    return X[:, 1:write_idx], y[1:write_idx], sigma_y[1:write_idx]
end

function fit_snapshot_gp(n_label::String, seed::Int, snapshot_iter::Int)
    X, y, sigma_y = reconstruct_training(n_label, seed, snapshot_iter)
    rng = MersenneTwister(seed + 100_000 * snapshot_iter + abs(hash(n_label)) % 10_000)
    gp = CalibrationCode.fit_heterogp(
        X, y, sigma_y;
        learn_hypers=true,
        learn_noise_scale=true,
        n_restarts=GP_N_RESTARTS,
        jitter=1e-8,
        rng=rng,
    )
    return gp, size(X, 2), X, y, sigma_y
end

function predict_axis(gp, axis::Symbol)
    grid = collect(range(-BOUND_SCALE, BOUND_SCALE; length=GRID_N))
    rows = NamedTuple[]
    for x_u in grid
        u = axis_point(axis, Float64(x_u))
        mu, s2 = CalibrationCode.predict_latent(gp, u)
        sigma = sqrt(max(s2, 0.0))
        push!(rows, (
            axis=axis_name(axis),
            x_u=Float64(x_u),
            x_physical=axis_physical(axis, Float64(x_u)),
            physical_unit=axis_unit(axis),
            mu=mu,
            sigma=sigma,
        ))
    end
    return rows
end

function score_label()
    return SCORE_MODE == :odd_penalty ? "odd_penalty" : "full_l1"
end

function gp_csv_paths()
    stem = "scale05_selectedN_$(score_label())_avg$(N_SEEDS_FOR_AVERAGE)_gp_slices_iters10_30_50_100"
    return (
        gp=joinpath(GP_SLICE_DIR, "$stem.csv"),
        selected=joinpath(GP_SLICE_DIR, "$(stem)_selected_seeds.csv"),
        training=joinpath(GP_SLICE_DIR, "$(stem)_training_points.csv"),
    )
end

function collect_gp_slice_rows()
    output_rows = NamedTuple[]
    selected_rows = NamedTuple[]
    training_rows = NamedTuple[]
    for n_label in N_LABELS
        selected, med, dir = selected_seed_rows(n_label)
        @printf("%s: median final score %.8f; selected seeds %s\n",
                LABELS[n_label], med, join((s.seed for s in selected), ","))
        for (rank, info) in enumerate(selected)
            push!(selected_rows, (
                score_mode=String(SCORE_MODE),
                N=n_label,
                selected_rank=rank,
                seed=info.seed,
                final_q_det=info.q_det,
                median_q_det=med,
                trace_dir=dir,
            ))
            for iter in DATA_SNAPSHOT_ITERS
                @printf("  %s seed %d rank %d: fitting GP at n=%d\n",
                        LABELS[n_label], info.seed, rank, iter)
                gp, n_train, X_train, y_train, sigma_train = fit_snapshot_gp(n_label, info.seed, iter)
                for j in 1:n_train
                    push!(training_rows, (
                        score_mode=String(SCORE_MODE),
                        N=n_label,
                        seed=info.seed,
                        selected_rank=rank,
                        iter=iter,
                        point_idx=j,
                        x_u1=X_train[1, j],
                        x_u2=X_train[2, j],
                        x_u3=X_train[3, j],
                        y=y_train[j],
                        sigma_y=sigma_train[j],
                    ))
                end
                for axis in AXES
                    for row in predict_axis(gp, axis)
                        push!(output_rows, (
                            score_mode=String(SCORE_MODE),
                            N=n_label,
                            seed=info.seed,
                            selected_rank=rank,
                            iter=iter,
                            n_train=n_train,
                            axis=row.axis,
                            x_u=row.x_u,
                            x_physical=row.x_physical,
                            physical_unit=row.physical_unit,
                            mu=row.mu,
                            sigma=row.sigma,
                        ))
                    end
                end
            end
        end
    end
    paths = gp_csv_paths()
    mkpath(GP_SLICE_DIR)
    write_namedtuple_csv(paths.gp, output_rows)
    write_namedtuple_csv(paths.selected, selected_rows)
    write_namedtuple_csv(paths.training, training_rows)
    println("Saved GP slice data: $(paths.gp)")
    println("Saved selected seed table: $(paths.selected)")
    println("Saved training points: $(paths.training)")
    return output_rows, selected_rows
end

function rows_by_key(rows)
    out = Dict{Tuple{String,Int,Symbol},Vector{Dict{String,String}}}()
    for row in rows
        key = (row["N"], parse(Int, row["iter"]), Symbol(row["axis"]))
        push!(get!(out, key, Vector{Dict{String,String}}()), row)
    end
    for key in keys(out)
        sort!(out[key]; by=row -> (parse_float(row["x_physical"]), parse(Int, row["selected_rank"])))
    end
    return out
end

function averaged_curve(rows)
    grouped = Dict{Float64,Vector{Dict{String,String}}}()
    for row in rows
        x = parse_float(row["x_physical"])
        push!(get!(grouped, x, Vector{Dict{String,String}}()), row)
    end
    xs = sort(collect(keys(grouped)))
    mu = Float64[]
    sigma = Float64[]
    for x in xs
        rs = grouped[x]
        mus = [parse_float(row["mu"]) for row in rs]
        sigmas = [parse_float(row["sigma"]) for row in rs]
        m = mean(mus)
        total_var = mean(sigmas .^ 2) + var(mus; corrected=false)
        push!(mu, m)
        push!(sigma, sqrt(max(total_var, 0.0)))
    end
    return xs, clamp.(mu, 0.0, 1.0), sigma
end

function style_axis!(ax)
    ax.xlabelsize = 25
    ax.ylabelsize = 27
    ax.xticklabelsize = 18
    ax.yticklabelsize = 18
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.xtickalign = 1
    ax.ytickalign = 1
    ax.xticksize = 7
    ax.yticksize = 7
    return ax
end

function draw_combined_plot(gp_lookup, true_lookup)
    fig = Figure(size=(1760, 1360), figure_padding=(28, 44, 22, 30), fontsize=24)
    for (row_i, iter) in enumerate(PLOT_SNAPSHOT_ITERS)
        for (col_i, axis) in enumerate(AXES)
            ax = Axis(fig[row_i, col_i];
                xlabel=row_i == length(PLOT_SNAPSHOT_ITERS) ? axis_label(axis) : "",
                ylabel=col_i == 1 ? L"\mathrm{Score}" : "",
                title=row_i == 1 ? axis_title(axis) : "",
            )
            style_axis!(ax)
            col_i != 1 && (ax.yticklabelsvisible = false)
            row_i != length(PLOT_SNAPSHOT_ITERS) && (ax.xticklabelsvisible = false)

            true_x, true_y = true_lookup[axis]
            lines!(ax, true_x, true_y; color=:black, linewidth=3.4,
                   label=row_i == 1 && col_i == 1 ? "true 2MS score" : nothing)

            for n_label in N_LABELS
                key = (n_label, iter, axis)
                haskey(gp_lookup, key) || continue
                xs, mu, sigma = averaged_curve(gp_lookup[key])
                color = COLORS[n_label]
                lo = clamp.(mu .- sigma, 0.0, 1.0)
                hi = clamp.(mu .+ sigma, 0.0, 1.0)
                band!(ax, xs, lo, hi; color=(color, n_label == "Inf" ? 0.08 : 0.055))
                lines!(ax, xs, mu; color=color, linewidth=n_label == "Inf" ? 3.2 : 2.7,
                       label=row_i == 1 && col_i == 1 ? LABELS[n_label] : nothing)
            end

            text!(ax, 0.04, 0.90; text="n = $iter", space=:relative,
                  fontsize=22, font=:bold, color=:black)
            ylims!(ax, GP_YMIN, GP_YMAX)
            axis === :amp ? xlims!(ax, -0.1, 0.1) : xlims!(ax, -5.0, 5.0)
            row_i == 1 && col_i == 1 && axislegend(ax; position=:lb, labelsize=14,
                                                    framevisible=true,
                                                    backgroundcolor=(:white, 0.92))
        end
    end
    title = SCORE_MODE == :odd_penalty ?
        "Average GP score slices, odd-penalty score" :
        "Average GP score slices, full-L1 score"
    Label(fig[0, :], title; fontsize=27, font=:bold, padding=(0, 0, 0, 8))
    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 24)
    return fig
end

function save_figure(fig, stem::String)
    mkpath(FIGURE_DIR)
    png = joinpath(FIGURE_DIR, "$stem.png")
    pdf = joinpath(FIGURE_DIR, "$stem.pdf")
    save(png, fig; px_per_unit=PX_PER_UNIT)
    save(pdf, fig)
    println("Saved $png")
    println("Saved $pdf")
end

function main()
    true_rows = load_or_write_true_score_cache()
    true_lookup = true_score_by_axis(true_rows)
    paths = gp_csv_paths()
    if isfile(paths.gp) && isfile(paths.selected) && isfile(paths.training) &&
       lowercase(get(ENV, "RECOMPUTE_SELECTEDN_GP_SLICES", "false")) ∉ ("1", "true", "yes", "on")
        println("Loaded GP slice data: $(paths.gp)")
    else
        collect_gp_slice_rows()
    end
    gp_rows = read_csv(paths.gp)
    gp_lookup = rows_by_key(gp_rows)
    fig = draw_combined_plot(gp_lookup, true_lookup)
    save_figure(fig,
        "scale05_selectedN_$(score_label())_avg$(N_SEEDS_FOR_AVERAGE)_gp_slices_iters10_30_100$(GP_OUTPUT_SUFFIX)")
end

main()
