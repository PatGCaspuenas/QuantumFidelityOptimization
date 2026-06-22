#!/usr/bin/env julia
#
# Generates paper figures:
#   figures/paper/figure_ab_vertical.{png,pdf}
#   figures/paper/figure_fcl_amp_n100_gpzoom.{png,pdf}

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using CairoMakie
using LaTeXStrings
using Printf
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO_ROOT, "data")
const FIGURE_DIR = joinpath(REPO_ROOT, "figures", "paper")

const PX_PER_UNIT = parse(Float64, get(ENV, "COMBINED_FIG_PX_PER_UNIT", "2.0"))

# Shared paper figure typography (reference: figure_fcl_amp_n100_gpzoom top panels)
const PAPER_FIG_FONTSIZE = 36.0
const PAPER_AXIS_LABELSIZE = 34.0
const PAPER_TICK_LABELSIZE = 28.0
const PAPER_TITLE_SIZE = 32.0
const PAPER_COLORBAR_LABELSIZE = 26.0
const PAPER_COLORBAR_TICKSIZE = 22.0
const PAPER_LEGEND_SIZE = 22.0

const MAX_ITER = 100
const TRACE_INFLOOR = 1e-10
const SCALE_INFLOOR = 1e-5
const SELECTED_Y_FLOOR = 1e-7
const SCALE_DROP_FAILURES = true
const SCALE_FAILURE_Q_THRESHOLD = 0.99

const SCALE_GROUPS = (
    (label="scale = 0.1", color=colorant"#8fc7e8",
     dir=joinpath(DATA_DIR, "traces_freqspan10_bound010_NInf_lhs12_restart_fullbudget_40seeds")),
    (label="scale = 0.5", color=colorant"#b89adb",
     dir=joinpath(DATA_DIR, "traces_freqspan10_bound050_NInf_lhs12_restart_fullbudget100_40seeds")),
    (label="scale = 1.0", color=colorant"#f3ad7a",
     dir=joinpath(DATA_DIR, "traces_freqspan10_bound100_NInf_lhs12_restart_fullbudget100_40seeds")),
)

const N_LABELS = ("100", "1000", "10000", "100000", "Inf")
const N_COLORS = Dict{String,Any}(
    "100" => colorant"#EE6677",
    "1000" => colorant"#CCBB44",
    "10000" => colorant"#228833",
    "100000" => colorant"#4477AA",
    "Inf" => colorant"#777777",
)
const N_LEGENDS = Dict(
    "100" => "N = 100",
    "1000" => "N = 1000",
    "10000" => "N = 10k",
    "100000" => "N = 100k",
    "Inf" => L"\mathrm{N} = \infty",
)

# Empirical finite-shot saturation model fit to the final median full-L1
# deterministic infidelity values in the selected-N data:
#     epsilon_N = epsilon_inf + A * N^(-alpha)
const MODEL_DASH_EPS_INF = parse(Float64, get(ENV, "MODEL_DASH_EPS_INF", "4.6404956e-7"))
const MODEL_DASH_A = parse(Float64, get(ENV, "MODEL_DASH_A", "0.00891668"))
const MODEL_DASH_ALPHA = parse(Float64, get(ENV, "MODEL_DASH_ALPHA", "0.4947"))

const GP_ITERS = (10, 30, 100)
const GP_ITERS_ALT = (10, 30, 50)
const GP_AXES = (:fcl, :fsb, :amp)
const GP_CSV = joinpath(DATA_DIR, "gp_slices",
    "scale05_selectedN_full_l1_avg5_gp_slices_iters10_30_50_100.csv")
const TRUE_SCORE_CSV = joinpath(DATA_DIR, "score_cache",
    "scale05_true_2ms_full_l1_slices_freqspan10_grid201.csv")
const VARMS2_RABI_SIDEBAND_CSV = joinpath(DATA_DIR,
    "varms_2_refined_merged_heatmap_rabi_sideband.csv")
const SAMPLING_BINS = 44
const SAMPLING_SMOOTH_SIGMA = 1.15
const SAMPLING_SKIP_INITIAL = 1
const SAMPLING_OMEGA_LIMS = (0.9, 1.1)
const SAMPLING_FSB_LIMS = (-5.0, 5.0)
const SAMPLING_FREQ_SPAN_KHZ = 10.0
const SAMPLING_AMP_RATIO_SPAN = 0.2

function read_csv_columns(path::String)
    lines = readlines(path)
    isempty(lines) && error("Empty CSV file: $path")
    header = split(lines[1], ',')
    cols = Dict(name => String[] for name in header)
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, ',')
        length(values) == length(header) ||
            error("Malformed row in $path: expected $(length(header)) fields, got $(length(values))")
        for (name, value) in zip(header, values)
            push!(cols[name], value)
        end
    end
    return cols
end

function read_csv_rows(path::String)
    open(path, "r") do io
        header = split(readline(io), ',')
        rows = Vector{Dict{String,String}}()
        for line in eachline(io)
            isempty(strip(line)) && continue
            values = split(line, ',')
            push!(rows, Dict(header[i] => get(values, i, "") for i in eachindex(header)))
        end
        return rows
    end
end

float_col(cols, name::String) = parse.(Float64, cols[name])
parse_float(value::AbstractString) = parse(Float64, value)

function read_trace(path::String)
    cols = read_csv_columns(path)
    return (
        iter=round.(Int, float_col(cols, "iter")),
        qdet=clamp.(float_col(cols, "q_det_rec"), 0.0, 1.0),
    )
end

function trace_paths(dir::String)
    isdir(dir) || error("Missing trace directory: $dir")
    paths = sort(filter(path -> occursin(r"trace_seed\d+_ucb\.csv$", basename(path)),
                        readdir(dir; join=true)))
    isempty(paths) && error("No trace CSVs found in $dir")
    return paths
end

function read_trace_group(dir::String; drop_failures::Bool=false)
    traces = [read_trace(path) for path in trace_paths(dir)]
    if drop_failures
        traces = [trace for trace in traces if trace.qdet[end] >= SCALE_FAILURE_Q_THRESHOLD]
        isempty(traces) && error("No traces remain in $dir after failure filtering")
    end
    return traces
end

function forward_fill_infid(trace, max_iter::Int; floor::Float64=TRACE_INFLOOR)
    values = Vector{Float64}(undef, max_iter)
    cursor = 1
    last_q = trace.qdet[1]
    for iter in 1:max_iter
        while cursor <= length(trace.iter) && trace.iter[cursor] <= iter
            last_q = trace.qdet[cursor]
            cursor += 1
        end
        values[iter] = max(1.0 - last_q, floor)
    end
    return values
end

quantile_row(matrix::Matrix{Float64}, q::Float64) =
    [quantile(@view(matrix[:, i]), q) for i in axes(matrix, 2)]

model_dash_infid(n_label::String) =
    MODEL_DASH_EPS_INF + MODEL_DASH_A * parse(Float64, n_label)^(-MODEL_DASH_ALPHA)

iteration_ticks() = ([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100"])

function decade_ticks(y_floor::Real)
    min_exp = floor(Int, log10(Float64(y_floor)))
    exponents = collect(0:-1:min_exp)
    return (10.0 .^ exponents, [latexstring("10^{$e}") for e in exponents])
end

function trace_matrix(traces; floor::Float64=TRACE_INFLOOR)
    return reduce(vcat,
        (reshape(forward_fill_infid(t, MAX_ITER; floor=floor), 1, :) for t in traces))
end

function full_l1_trace_dir(n_label::String)
    return joinpath(DATA_DIR,
        "traces_freqspan10_bound050_full_l1_N$(n_label)_nostop100_stream_40seeds")
end

function load_true_score_lookup()
    rows = read_csv_rows(TRUE_SCORE_CSV)
    lookup = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()
    for axis in GP_AXES
        name = axis === :fcl ? "fcl" : axis === :fsb ? "fsb" : "amp"
        axis_rows = filter(row -> row["axis"] == name, rows)
        sort!(axis_rows; by=row -> parse_float(row["x_physical"]))
        lookup[axis] = (
            [parse_float(row["x_physical"]) for row in axis_rows],
            [parse_float(row["score"]) for row in axis_rows],
        )
    end
    return lookup
end

function load_gp_lookup()
    rows = read_csv_rows(GP_CSV)
    lookup = Dict{Tuple{String,Int,Symbol},Vector{Dict{String,String}}}()
    for row in rows
        key = (row["N"], parse(Int, row["iter"]), Symbol(row["axis"]))
        push!(get!(lookup, key, Vector{Dict{String,String}}()), row)
    end
    return lookup
end

function averaged_gp_curve(rows)
    grouped = Dict{Float64,Vector{Dict{String,String}}}()
    for row in rows
        x = parse_float(row["x_physical"])
        push!(get!(grouped, x, Vector{Dict{String,String}}()), row)
    end
    xs = sort(collect(keys(grouped)))
    mu = Float64[]
    sigma = Float64[]
    for x in xs
        group = grouped[x]
        mus = [parse_float(row["mu"]) for row in group]
        sigmas = [parse_float(row["sigma"]) for row in group]
        m = mean(mus)
        total_var = mean(sigmas .^ 2) + var(mus; corrected=false)
        push!(mu, clamp(m, 0.0, 1.0))
        push!(sigma, sqrt(max(total_var, 0.0)))
    end
    return xs, mu, sigma
end

function style_log_axis!(ax; ticksize=12.0,
                         labelsize=PAPER_AXIS_LABELSIZE,
                         ticklabelsize=PAPER_TICK_LABELSIZE)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.xtickalign = 1
    ax.ytickalign = 1
    ax.xticksize = ticksize
    ax.yticksize = ticksize
    ax.xlabelsize = labelsize
    ax.ylabelsize = labelsize
    ax.xticklabelsize = ticklabelsize
    ax.yticklabelsize = ticklabelsize
    return ax
end

function style_gp_axis!(ax)
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.xtickalign = 1
    ax.ytickalign = 1
    ax.xticksize = 11.0
    ax.yticksize = 11.0
    ax.xlabelsize = PAPER_AXIS_LABELSIZE
    ax.ylabelsize = PAPER_AXIS_LABELSIZE
    ax.xticklabelsize = PAPER_TICK_LABELSIZE
    ax.yticklabelsize = PAPER_TICK_LABELSIZE
    ax.titlesize = PAPER_TITLE_SIZE
    ax.titlegap = 8
    return ax
end

function draw_scale_trend!(ax)
    xs = collect(1:MAX_ITER)
    for group in SCALE_GROUPS
        traces = read_trace_group(group.dir; drop_failures=SCALE_DROP_FAILURES)
        matrix = trace_matrix(traces; floor=SCALE_INFLOOR)
        q05 = quantile_row(matrix, 0.05)
        q16 = quantile_row(matrix, 0.16)
        q50 = quantile_row(matrix, 0.50)
        q84 = quantile_row(matrix, 0.84)
        q95 = quantile_row(matrix, 0.95)
        band!(ax, xs, q05, q95; color=(group.color, 0.08))
        band!(ax, xs, q16, q84; color=(group.color, 0.15))
        lines!(ax, xs, q16; color=(group.color, 0.72), linewidth=1.0, linestyle=:dot)
        lines!(ax, xs, q84; color=(group.color, 0.72), linewidth=1.0, linestyle=:dot)
        lines!(ax, xs, q50; color=group.color, linewidth=1.7, label=group.label)
        @printf("%s traces=%d final median %.6g\n", group.label, size(matrix, 1), q50[end])
    end
    xlims!(ax, 0, MAX_ITER)
    ax.xticks = iteration_ticks()
    ylims!(ax, SCALE_INFLOOR, 1.0)
    ax.yticks = decade_ticks(SCALE_INFLOOR)
end

function draw_selected_n_trend!(ax)
    xs = collect(1:MAX_ITER)
    dash_refs = Tuple{Float64,Any}[]
    for n_label in N_LABELS
        traces = read_trace_group(full_l1_trace_dir(n_label))
        matrix = trace_matrix(traces; floor=TRACE_INFLOOR)
        q16 = quantile_row(matrix, 0.16)
        q50 = quantile_row(matrix, 0.50)
        q84 = quantile_row(matrix, 0.84)
        color = N_COLORS[n_label]
        band!(ax, xs, q16, q84; color=(color, n_label == "Inf" ? 0.10 : 0.07))
        lines!(ax, xs, q16; color=(color, 0.72), linewidth=0.9, linestyle=:dot)
        lines!(ax, xs, q84; color=(color, 0.72), linewidth=0.9, linestyle=:dot)
        lines!(ax, xs, q50; color=color, linewidth=n_label == "Inf" ? 1.7 : 1.5,
               label=N_LEGENDS[n_label])
        if n_label != "Inf"
            ref = max(model_dash_infid(n_label), TRACE_INFLOOR)
            push!(dash_refs, (ref, color))
        end
        @printf("%s traces=%d final median %.6g\n", N_LEGENDS[n_label], size(matrix, 1), q50[end])
    end
    for (ref, color) in dash_refs
        hlines!(ax, [ref]; color=(color, 0.98), linestyle=:dash, linewidth=1.8)
    end
    xlims!(ax, 0, MAX_ITER)
    ax.xticks = iteration_ticks()
    ylims!(ax, SELECTED_Y_FLOOR, 1.0)
    ax.yticks = decade_ticks(SELECTED_Y_FLOOR)
end

axis_label(axis::Symbol) = axis === :fcl ? L"\Delta f_{\mathrm{cl}}\;(\mathrm{kHz})" :
                           axis === :fsb ? L"\Delta f_{\mathrm{sb}}\;(\mathrm{kHz})" :
                           L"\Delta A/A_{\mathrm{opt}}"

function draw_figure_ab_vertical()
    fig = Figure(size=(750, 1000), figure_padding=(70, 40, 40, 40),
                 fontsize=PAPER_FIG_FONTSIZE)

    ax_scale = Axis(fig[1, 1];
        xlabel="",
        ylabel=L"1 - Q_{\mathrm{det}}(x_n^*)",
        yscale=log10,
    )
    style_log_axis!(ax_scale)
    draw_scale_trend!(ax_scale)
    ax_scale.xticksvisible = false
    ax_scale.xticklabelsvisible = false
    ax_scale.xlabelvisible = false
    axislegend(ax_scale; position=:rt, labelsize=PAPER_LEGEND_SIZE, framevisible=false,
               patchsize=(28, 12), rowgap=2.0, padding=(6, 6, 3, 3))

    ax_selected = Axis(fig[2, 1];
        xlabel=L"n",
        ylabel=L"1 - Q_{\mathrm{det}}(x_n^*)",
        yscale=log10,
    )
    style_log_axis!(ax_selected)
    draw_selected_n_trend!(ax_selected)
    axislegend(ax_selected; position=:rt, labelsize=PAPER_LEGEND_SIZE, framevisible=false,
               patchsize=(28, 12), rowgap=2.0, padding=(6, 6, 3, 3))

    rowgap!(fig.layout, 50)

    return fig
end

function draw_gp_overlay_axis!(ax, gp_lookup, true_lookup, axis::Symbol, gp_iters;
                               gp_ylim=(0.5, 1.0))
    style_gp_axis!(ax)
    ax.xticklabelpad = 8.0
    ax.yticklabelpad = 8.0

    true_x, true_y = true_lookup[axis]
    lines!(ax, true_x, true_y; color=:black, linewidth=2.3)

    iter_colors = (colorant"#EE6677", colorant"#4477AA", colorant"#228833")
    for (iter_i, iter) in enumerate(gp_iters)
        key = ("100", iter, axis)
        if haskey(gp_lookup, key)
            xs, mu, sigma = averaged_gp_curve(gp_lookup[key])
            color = iter_colors[iter_i]
            band!(ax, xs, clamp.(mu .- sigma, 0.0, 1.0), clamp.(mu .+ sigma, 0.0, 1.0);
                  color=(color, 0.08))
            lines!(ax, xs, mu; color=color, linewidth=2.2)
        end
    end

    ylims!(ax, gp_ylim...)
    ax.yticks = ([0.5, 1.0], ["0.5", "1.0"])
    if axis === :amp
        xlims!(ax, -0.1, 0.1)
        ax.xticks = ([-0.1, 0.0, 0.1], ["-0.1", "0", "0.1"])
    else
        xlims!(ax, -5.0, 5.0)
        ax.xticks = ([-5.0, 0.0, 5.0], ["-5", "0", "5"])
    end

    return ax
end

function full_l1_scores(rows)
    scores = Float64[]
    for row in rows
        p = clamp.([
            parse_float(row["gg"]),
            parse_float(row["ee"]),
            parse_float(row["odd"]),
        ], 0.0, 1.0)
        psum = sum(p)
        psum > 0.0 && (p ./= psum)

        target_gg = parse_float(row["nominal_gg"])
        target_ee = parse_float(row["nominal_ee"])
        q = clamp.([target_gg, target_ee, max(1.0 - target_gg - target_ee, 0.0)], 0.0, 1.0)
        qsum = sum(q)
        qsum > 0.0 && (q ./= qsum)

        push!(scores, clamp(1.0 - 0.5 * sum(abs.(p .- q)), 0.0, 1.0))
    end
    return scores
end

function sampling_edges(lims::Tuple{Float64,Float64})
    return collect(range(lims[1], lims[2]; length=SAMPLING_BINS + 1))
end

sampling_centers(edges) = [(edges[i] + edges[i + 1]) / 2 for i in 1:(length(edges) - 1)]

function bin_index(edges, value::Float64)
    value < edges[1] && return nothing
    value > edges[end] && return nothing
    idx = searchsortedlast(edges, value)
    return clamp(idx, 1, length(edges) - 1)
end

function gaussian_kernel(sigma::Float64)
    radius = max(1, ceil(Int, 3sigma))
    offsets = collect(-radius:radius)
    kernel = exp.(-0.5 .* (offsets ./ sigma) .^ 2)
    return offsets, kernel ./ sum(kernel)
end

function smooth_sampling_grid(grid::Matrix{Float64}; sigma::Float64=SAMPLING_SMOOTH_SIGMA)
    offsets, kernel = gaussian_kernel(sigma)
    nx, ny = size(grid)
    tmp = zeros(Float64, nx, ny)
    out = zeros(Float64, nx, ny)

    for ix in 1:nx, iy in 1:ny
        total = 0.0
        for (offset, weight) in zip(offsets, kernel)
            jx = clamp(ix + offset, 1, nx)
            total += weight * grid[jx, iy]
        end
        tmp[ix, iy] = total
    end

    for ix in 1:nx, iy in 1:ny
        total = 0.0
        for (offset, weight) in zip(offsets, kernel)
            jy = clamp(iy + offset, 1, ny)
            total += weight * tmp[ix, jy]
        end
        out[ix, iy] = total
    end

    return clamp.(out, 0.0, 1.0)
end

function sampling_frequency_grid(kind::Symbol)
    x_edges = sampling_edges(SAMPLING_OMEGA_LIMS)
    y_edges = sampling_edges(SAMPLING_FSB_LIMS)
    visits = zeros(Float64, SAMPLING_BINS, SAMPLING_BINS)
    paths = trace_paths(full_l1_trace_dir("100"))

    x_col = kind === :rec ? "x_rec_u3" : "x_acq_u3"
    y_col = kind === :rec ? "x_rec_u2" : "x_acq_u2"

    for path in paths
        cols = read_csv_columns(path)
        run_visits = falses(SAMPLING_BINS, SAMPLING_BINS)
        for row_i in (SAMPLING_SKIP_INITIAL + 1):length(cols[x_col])
            omega_ratio = 1.0 + SAMPLING_AMP_RATIO_SPAN * parse_float(cols[x_col][row_i])
            fsb_khz = SAMPLING_FREQ_SPAN_KHZ * parse_float(cols[y_col][row_i])
            ix = bin_index(x_edges, omega_ratio)
            iy = bin_index(y_edges, fsb_khz)
            (ix === nothing || iy === nothing) && continue
            run_visits[ix, iy] = true
        end
        visits .+= run_visits
    end

    freq = visits ./ max(length(paths), 1)
    return sampling_centers(x_edges), sampling_centers(y_edges),
        smooth_sampling_grid(freq)
end

function draw_sampling_contours!(ax)
    rows = read_csv_rows(VARMS2_RABI_SIDEBAND_CSV)
    data_xs = [parse_float(row["rabi_ratio"]) for row in rows]
    data_ys = [2π * parse_float(row["sideband_2pi_khz"]) for row in rows]
    scores = full_l1_scores(rows)

    xs = collect(range(SAMPLING_OMEGA_LIMS[1], SAMPLING_OMEGA_LIMS[2]; length=140))
    ys = collect(range(SAMPLING_FSB_LIMS[1], SAMPLING_FSB_LIMS[2]; length=140))
    z = Matrix{Float64}(undef, length(xs), length(ys))
    xscale = SAMPLING_OMEGA_LIMS[2] - SAMPLING_OMEGA_LIMS[1]
    yscale = SAMPLING_FSB_LIMS[2] - SAMPLING_FSB_LIMS[1]

    @inbounds for ix in eachindex(xs), iy in eachindex(ys)
        numerator = 0.0
        denominator = 0.0
        for k in eachindex(scores)
            dx = (data_xs[k] - xs[ix]) / xscale
            dy = (data_ys[k] - ys[iy]) / yscale
            d2 = dx * dx + dy * dy
            if d2 < 1e-14
                numerator = scores[k]
                denominator = 1.0
                break
            end
            weight = d2^-1.0
            numerator += weight * scores[k]
            denominator += weight
        end
        z[ix, iy] = numerator / denominator
    end

    contour!(ax, xs, ys, z; levels=0.1:0.2:0.9,
             color=(:gray45, 0.9), linewidth=1.0)
    contour!(ax, xs, ys, z; levels=[0.999],
             color=(:gray15, 0.95), linewidth=2.2)
    return ax
end

function style_sampling_axis!(ax)
    ax.xgridvisible = true
    ax.ygridvisible = true
    ax.xgridstyle = :dot
    ax.ygridstyle = :dot
    ax.xgridcolor = (:gray, 0.35)
    ax.ygridcolor = (:gray, 0.35)
    ax.xtickalign = 1
    ax.ytickalign = 1
    ax.xticksize = 10.0
    ax.yticksize = 10.0
    ax.xlabelsize = PAPER_AXIS_LABELSIZE
    ax.ylabelsize = PAPER_AXIS_LABELSIZE
    ax.xticklabelsize = PAPER_TICK_LABELSIZE
    ax.yticklabelsize = PAPER_TICK_LABELSIZE
    ax.xticklabelpad = 10.0
    ax.yticklabelpad = 14.0
    ax.titlesize = PAPER_TITLE_SIZE
    return ax
end

function draw_sampling_density_panel!(ax, kind::Symbol)
    xs, ys, freq = sampling_frequency_grid(kind)
    heatmap_plot = heatmap!(ax, xs, ys, max.(freq, 0.01);
        colormap=cgrad([:white, colorant"#a8c8ff", colorant"#4f83d1"]),
        colorrange=(0.01, 1.0),
        colorscale=log10)
    draw_sampling_contours!(ax)

    xlims!(ax, SAMPLING_OMEGA_LIMS...)
    ylims!(ax, SAMPLING_FSB_LIMS...)
    ax.xticks = ([0.9, 1.0, 1.1], ["0.9", "1.0", "1.1"])
    ax.yticks = ([-5, 0, 5], ["-5", "0", "5"])
    return heatmap_plot
end

function draw_gp_overlay_pair_n100()
    gp_lookup = load_gp_lookup()
    true_lookup = load_true_score_lookup()

    fig = Figure(size=(1320, 1360), figure_padding=(138, 92, 166, 190),
                 fontsize=PAPER_FIG_FONTSIZE)

    ax_fsb = Axis(fig[1, 1];
        xlabel=axis_label(:fsb),
        ylabel=L"\mathrm{Score}",
    )
    draw_gp_overlay_axis!(ax_fsb, gp_lookup, true_lookup, :fsb, GP_ITERS_ALT)

    ax_amp = Axis(fig[1, 2];
        xlabel=axis_label(:amp),
        ylabel="",
    )
    draw_gp_overlay_axis!(ax_amp, gp_lookup, true_lookup, :amp, GP_ITERS)
    ax_amp.yticklabelsvisible = false

    ax_rec = Axis(fig[2, 1];
        xlabel=L"\Omega/\Omega_{\mathrm{opt}}",
        ylabel=L"\Delta f_{\mathrm{sb}}\;(\mathrm{kHz})",
        title=L"\mathrm{estimated\;max}\;(x_n^*)",
    )
    style_sampling_axis!(ax_rec)
    sampling_plot = draw_sampling_density_panel!(ax_rec, :rec)

    ax_acq = Axis(fig[2, 2];
        xlabel=L"\Omega/\Omega_{\mathrm{opt}}",
        ylabel="",
        title=L"\mathrm{samples}\;(x_{n+1})",
    )
    style_sampling_axis!(ax_acq)
    draw_sampling_density_panel!(ax_acq, :acq)
    ax_acq.yticklabelsvisible = false

    Colorbar(fig[3, 1:2], sampling_plot;
        vertical=false,
        label="Sampling Frequency",
        ticks=([0.01, 0.1, 1.0], ["0.01", "0.1", "1"]),
        labelsize=PAPER_COLORBAR_LABELSIZE,
        ticklabelsize=PAPER_COLORBAR_TICKSIZE,
        width=420,
        height=22,
        halign=:center)

    colgap!(fig.layout, 56)
    rowgap!(fig.layout, 52)
    rowsize!(fig.layout, 1, Fixed(360))
    rowsize!(fig.layout, 2, Fixed(340))
    rowsize!(fig.layout, 3, Fixed(120))

    return fig
end

function main()
    mkpath(FIGURE_DIR)

    ab_stem = joinpath(FIGURE_DIR, "figure_ab_vertical")
    ab_fig = draw_figure_ab_vertical()
    save("$ab_stem.png", ab_fig; px_per_unit=PX_PER_UNIT)
    save("$ab_stem.pdf", ab_fig)
    println("Saved $ab_stem.png")
    println("Saved $ab_stem.pdf")

    pair_stem = joinpath(FIGURE_DIR, "figure_fcl_amp_n100_gpzoom")
    pair_fig = draw_gp_overlay_pair_n100()
    save("$pair_stem.png", pair_fig; px_per_unit=PX_PER_UNIT)
    save("$pair_stem.pdf", pair_fig)
    println("Saved $pair_stem.png")
    println("Saved $pair_stem.pdf")
end

main()
