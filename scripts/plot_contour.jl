# Generates paper figure:
#   figures/paper/mainpaper_contour_2ms.png
#   (via PLOT_SCORE_OUTPATH; see header env flags below)
#
# Combined 2MS/3MS/Jacobian contour figure using the full-population L1 target:
#
#   Score = 1 - 0.5 * sum(abs.([gg, ee, odd] - [target_gg, target_ee, target_odd]))
#
# Emits the direct Score contour figure used in the paper.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using CairoMakie
using LaTeXStrings

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO_ROOT, "data")
const FIGURE_DIR = get(ENV, "PLOT_FIGURE_DIR", joinpath(REPO_ROOT, "figures"))
const PX_PER_UNIT = parse(Float64, get(ENV, "PLOT_PX_PER_UNIT", "2"))
const SCORE_ONE_TOL = parse(Float64, get(ENV, "PLOT_SCORE_ONE_TOL", "1e-12"))

# Match figure_fcl_amp_n100_gpzoom paper typography
const PAPER_FIG_FONTSIZE = 36.0
const PAPER_AXIS_LABELSIZE = 34.0
const PAPER_TICK_LABELSIZE = 28.0
const PAPER_TITLE_SIZE = 32.0
const PAPER_COLORBAR_LABELSIZE = 26.0
const PAPER_COLORBAR_TICKSIZE = 22.0

env_bool(name::String, default::Bool) = begin
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    throw(ArgumentError("$name must be true/false, got '$raw'"))
end

const SHOW_ROW_LABELS = env_bool("PLOT_SHOW_ROW_LABELS", false)
const SHOW_PANEL_LABELS = env_bool("PLOT_SHOW_PANEL_LABELS", true)
const OMIT_PHASE = env_bool("PLOT_OMIT_PHASE", false)
const VERTICAL_LAYOUT = env_bool("PLOT_VERTICAL_LAYOUT", false)
const INTERPOLATE = env_bool("PLOT_INTERPOLATE", false)
const INTERP_METHOD = lowercase(get(ENV, "PLOT_INTERP_METHOD", "idw"))
const INTERP_N = parse(Int, get(ENV, "PLOT_INTERP_N", "220"))
const INTERP_POWER = parse(Float64, get(ENV, "PLOT_INTERP_POWER", "2.0"))
const SMOOTH_SIGMA_FRAC = parse(Float64, get(ENV, "PLOT_SMOOTH_SIGMA_FRAC", "0.055"))
const SMOOTH_CUTOFF_SIGMA = parse(Float64, get(ENV, "PLOT_SMOOTH_CUTOFF_SIGMA", "4.0"))
const SCORE_INTERP_SPACE = lowercase(get(ENV, "PLOT_SCORE_INTERP_SPACE", "score"))
const MAX_PRESERVE_BLEND = parse(Float64, get(ENV, "PLOT_MAX_PRESERVE_BLEND", "0.35"))
const SHORT_SCORE_LABEL = env_bool("PLOT_SHORT_SCORE_LABEL", false)

const ALL_PULSE_SPECS = (
    (
        name="varms_2",
        label="2MS",
        prefix_label="varms2",
        merged_prefix="varms_2_refined_merged_heatmap",
    ),
    (
        name="baseline_3ms",
        label="3MS",
        prefix_label="3ms",
        merged_prefix="baseline_3ms_refined_merged_heatmap",
    ),
    (
        name="jacobian_probe",
        label="Jacobian",
        prefix_label="jacobian",
        merged_prefix="jacobian_probe_refined_merged_heatmap",
    ),
)

function selected_pulse_specs()
    raw = strip(get(ENV, "PLOT_PULSES", "varms_2,baseline_3ms,jacobian_probe"))
    requested = strip.(split(raw, ","))
    specs = [spec for name in requested for spec in ALL_PULSE_SPECS if spec.name == name]
    length(specs) == length(requested) ||
        error("PLOT_PULSES contains unknown pulse(s): $raw")
    return Tuple(specs)
end

const PULSE_SPECS = selected_pulse_specs()
const OUT_PREFIX = get(
    ENV,
    "PLOT_OUT_PREFIX",
    join((spec.prefix_label for spec in PULSE_SPECS), "_") * "_full_l1",
)

const HEATMAP_PAIRS = (
    (:rabi, :sideband),
    (:rabi, :phase),
    (:rabi, :fcl),
    (:sideband, :phase),
    (:sideband, :fcl),
    (:phase, :fcl),
)

const ACTIVE_HEATMAP_PAIRS = OMIT_PHASE ?
    Tuple(pair for pair in HEATMAP_PAIRS if !(:phase in pair)) :
    HEATMAP_PAIRS
const N_PANEL_COLS = VERTICAL_LAYOUT ? length(PULSE_SPECS) : length(ACTIVE_HEATMAP_PAIRS)
const N_PANEL_ROWS = VERTICAL_LAYOUT ? length(ACTIVE_HEATMAP_PAIRS) : length(PULSE_SPECS)
const LOG_OUTPATH = get(
    ENV,
    "PLOT_LOG_OUTPATH",
    joinpath(FIGURE_DIR, "$(OUT_PREFIX)_logscore_$(N_PANEL_ROWS)x$(N_PANEL_COLS).png"),
)
const SCORE_OUTPATH = get(
    ENV,
    "PLOT_SCORE_OUTPATH",
    joinpath(FIGURE_DIR, "$(OUT_PREFIX)_score_$(N_PANEL_ROWS)x$(N_PANEL_COLS).png"),
)

function read_generated_csv(path::String)
    lines = readlines(path)
    isempty(lines) && error("CSV is empty: $path")
    header = split(lines[1], ',')
    rows = Dict{String,Vector{String}}(name => String[] for name in header)
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, ',')
        length(values) == length(header) || error("Malformed row in $path: $line")
        for (name, value) in zip(header, values)
            push!(rows[name], value)
        end
    end
    return rows
end

function float_column(rows, name::String)
    haskey(rows, name) || error("Missing column '$name'")
    values = Float64[]
    for value in rows[name]
        parsed = tryparse(Float64, value)
        parsed === nothing && error("Could not parse '$value' in column '$name'")
        push!(values, parsed)
    end
    return values
end

merged_heatmap_path(spec, axis_x::Symbol, axis_y::Symbol) =
    joinpath(DATA_DIR, "$(spec.merged_prefix)_$(axis_x)_$(axis_y).csv")

coordinate_column(axis::Symbol) = if axis === :rabi
    "rabi_ratio"
elseif axis === :phase
    "phase_pi"
elseif axis === :sideband
    "sideband_2pi_khz"
elseif axis === :fcl
    "fcl_2pi_khz"
else
    error("Unknown axis: $axis")
end

axis_label(axis::Symbol) = if axis === :rabi
    L"\Omega / \Omega_{\mathrm{opt}}"
elseif axis === :phase
    L"\Delta\phi / \pi"
elseif axis === :sideband
    L"\Delta\delta\;(\mathrm{kHz})"
elseif axis === :fcl
    L"\Delta\omega_{\mathrm{cl}}\;(\mathrm{kHz})"
else
    string(axis)
end

display_coordinate(axis::Symbol, values::Vector{Float64}) =
    axis in (:sideband, :fcl) ? values .* (2π) : values

function full_l1_scores(rows)
    gg = float_column(rows, "gg")
    ee = float_column(rows, "ee")
    odd = float_column(rows, "odd")
    target_gg = float_column(rows, "nominal_gg")
    target_ee = float_column(rows, "nominal_ee")

    scores = Vector{Float64}(undef, length(gg))
    for i in eachindex(scores)
        p = clamp.([gg[i], ee[i], odd[i]], 0.0, 1.0)
        psum = sum(p)
        psum > 0.0 && (p ./= psum)

        target_odd = max(1.0 - target_gg[i] - target_ee[i], 0.0)
        q = clamp.([target_gg[i], target_ee[i], target_odd], 0.0, 1.0)
        qsum = sum(q)
        qsum > 0.0 && (q ./= qsum)

        scores[i] = clamp(1.0 - 0.5 * sum(abs.(p .- q)), 0.0, 1.0)
    end
    return scores
end

function point_cloud(rows, axis_x::Symbol, axis_y::Symbol)
    xs = display_coordinate(axis_x, float_column(rows, coordinate_column(axis_x)))
    ys = display_coordinate(axis_y, float_column(rows, coordinate_column(axis_y)))
    return xs, ys, full_l1_scores(rows)
end

function shared_log_floor()
    best_nonone = -Inf
    for spec in PULSE_SPECS, (axis_x, axis_y) in ACTIVE_HEATMAP_PAIRS
        path = merged_heatmap_path(spec, axis_x, axis_y)
        isfile(path) || error("Missing merged CSV: $path")
        rows = read_generated_csv(path)
        for score in full_l1_scores(rows)
            if score < 1.0 - SCORE_ONE_TOL
                best_nonone = max(best_nonone, score)
            end
        end
    end
    isfinite(best_nonone) ||
        error("Could not find a score below 1.0 using SCORE_ONE_TOL=$SCORE_ONE_TOL")
    return log10(1.0 - best_nonone), best_nonone
end

log_score_values(scores::Vector{Float64}, log_floor::Float64) =
    clamp.(log10.(max.(1.0 .- clamp.(scores, 0.0, 1.0), 10.0^log_floor)), log_floor, 0.0)

score_from_log_values(values::AbstractArray{<:Real}) =
    clamp.(1.0 .- 10.0 .^ values, 0.0, 1.0)

function style_axis!(ax; show_xlabel::Bool)
    ax.xlabelsize = PAPER_AXIS_LABELSIZE
    ax.ylabelsize = PAPER_AXIS_LABELSIZE
    ax.xticklabelsize = PAPER_TICK_LABELSIZE
    ax.yticklabelsize = PAPER_TICK_LABELSIZE
    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.xlabelvisible = show_xlabel
    return ax
end

function panel_label!(ax, label::String)
    text!(ax, 0.035, 0.94;
          text=label,
          space=:relative,
          align=(:left, :top),
          fontsize=PAPER_TITLE_SIZE,
          font=:bold,
          color=:white)
    return ax
end

axis_limits(axis::Symbol, values::Vector{Float64}) =
    axis in (:sideband, :fcl) ? (-10.0, 10.0) :
    axis === :rabi ? (0.8, 1.2) :
    (minimum(values), maximum(values))

function set_axis_ticks!(ax, axis_x::Symbol, axis_y::Symbol)
    axis_x === :rabi && (ax.xticks = (0.8:0.1:1.2, ["0.8", "0.9", "1.0", "1.1", "1.2"]))
    axis_y === :rabi && (ax.yticks = (0.8:0.1:1.2, ["0.8", "0.9", "1.0", "1.1", "1.2"]))
    return ax
end

function idw_interpolated_grid(xs::Vector{Float64}, ys::Vector{Float64},
                               values::Vector{Float64}, axis_x::Symbol,
                               axis_y::Symbol)
    xmin, xmax = axis_limits(axis_x, xs)
    ymin, ymax = axis_limits(axis_y, ys)
    xgrid = collect(range(xmin, xmax; length=INTERP_N))
    ygrid = collect(range(ymin, ymax; length=INTERP_N))
    z = Matrix{Float64}(undef, length(xgrid), length(ygrid))

    xscale = max(xmax - xmin, eps(Float64))
    yscale = max(ymax - ymin, eps(Float64))
    tiny = 1.0e-14
    @inbounds for ix in eachindex(xgrid), iy in eachindex(ygrid)
        x = xgrid[ix]
        y = ygrid[iy]
        numerator = 0.0
        denominator = 0.0
        exact = false
        exact_value = 0.0
        for k in eachindex(values)
            dx = (xs[k] - x) / xscale
            dy = (ys[k] - y) / yscale
            d2 = dx * dx + dy * dy
            if d2 < tiny
                exact = true
                exact_value = values[k]
                break
            end
            w = d2^(-0.5 * INTERP_POWER)
            numerator += w * values[k]
            denominator += w
        end
        z[ix, iy] = exact ? exact_value : numerator / denominator
    end
    return xgrid, ygrid, z
end

function gaussian_smoothed_grid(xs::Vector{Float64}, ys::Vector{Float64},
                                values::Vector{Float64}, axis_x::Symbol,
                                axis_y::Symbol)
    xmin, xmax = axis_limits(axis_x, xs)
    ymin, ymax = axis_limits(axis_y, ys)
    xgrid = collect(range(xmin, xmax; length=INTERP_N))
    ygrid = collect(range(ymin, ymax; length=INTERP_N))
    z = Matrix{Float64}(undef, length(xgrid), length(ygrid))

    σx = max(SMOOTH_SIGMA_FRAC * (xmax - xmin), eps(Float64))
    σy = max(SMOOTH_SIGMA_FRAC * (ymax - ymin), eps(Float64))
    cutoff2 = SMOOTH_CUTOFF_SIGMA^2
    @inbounds for ix in eachindex(xgrid), iy in eachindex(ygrid)
        x = xgrid[ix]
        y = ygrid[iy]
        numerator = 0.0
        denominator = 0.0
        nearest_d2 = Inf
        nearest_value = values[1]
        for k in eachindex(values)
            dx = (xs[k] - x) / σx
            dy = (ys[k] - y) / σy
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2
                nearest_d2 = d2
                nearest_value = values[k]
            end
            d2 > cutoff2 && continue
            w = exp(-0.5 * d2)
            numerator += w * values[k]
            denominator += w
        end
        z[ix, iy] = denominator > 0.0 ? numerator / denominator : nearest_value
    end
    return xgrid, ygrid, clamp.(z, minimum(values), maximum(values))
end

function gaussian_max_preserving_grid(xs::Vector{Float64}, ys::Vector{Float64},
                                      values::Vector{Float64}, axis_x::Symbol,
                                      axis_y::Symbol)
    xgrid, ygrid, smooth = gaussian_smoothed_grid(xs, ys, values, axis_x, axis_y)
    z = similar(smooth)

    xmin, xmax = axis_limits(axis_x, xs)
    ymin, ymax = axis_limits(axis_y, ys)
    σx = max(SMOOTH_SIGMA_FRAC * (xmax - xmin), eps(Float64))
    σy = max(SMOOTH_SIGMA_FRAC * (ymax - ymin), eps(Float64))
    cutoff2 = SMOOTH_CUTOFF_SIGMA^2

    @inbounds for ix in eachindex(xgrid), iy in eachindex(ygrid)
        x = xgrid[ix]
        y = ygrid[iy]
        local_max = -Inf
        nearest_d2 = Inf
        nearest_value = values[1]
        for k in eachindex(values)
            dx = (xs[k] - x) / σx
            dy = (ys[k] - y) / σy
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2
                nearest_d2 = d2
                nearest_value = values[k]
            end
            d2 <= cutoff2 && (local_max = max(local_max, values[k]))
        end
        peak = isfinite(local_max) ? local_max : nearest_value
        z[ix, iy] = (1.0 - MAX_PRESERVE_BLEND) * smooth[ix, iy] + MAX_PRESERVE_BLEND * peak
    end

    return xgrid, ygrid, clamp.(z, minimum(values), maximum(values))
end

function draw_figure!(fig, values_mode::Symbol; log_floor::Float64)
    contour_plot = nothing
    for (pulse_idx, spec) in enumerate(PULSE_SPECS)
        if SHOW_ROW_LABELS && !VERTICAL_LAYOUT
            label_row = 2 * pulse_idx - 1
            Label(fig[label_row, 1:N_PANEL_COLS], spec.label;
                  fontsize=PAPER_AXIS_LABELSIZE,
                  font=:bold,
                  tellwidth=false)
        end
        for (pair_idx, (axis_x, axis_y)) in enumerate(ACTIVE_HEATMAP_PAIRS)
            axis_row = VERTICAL_LAYOUT ? pair_idx :
                (SHOW_ROW_LABELS ? 2 * pulse_idx : pulse_idx)
            axis_col = VERTICAL_LAYOUT ? pulse_idx : pair_idx
            ax = Axis(fig[axis_row, axis_col];
                xlabel=axis_label(axis_x),
                ylabel=axis_label(axis_y),
                xlabelpadding=6,
                ylabelpadding=6,
            )
            style_axis!(ax; show_xlabel=VERTICAL_LAYOUT || pulse_idx == length(PULSE_SPECS))

            path = merged_heatmap_path(spec, axis_x, axis_y)
            rows = read_generated_csv(path)
            xs, ys, scores = point_cloud(rows, axis_x, axis_y)
            interp_values = values_mode === :logscore ? log_score_values(scores, log_floor) :
                SCORE_INTERP_SPACE == "loginfid" ? log_score_values(scores, log_floor) :
                SCORE_INTERP_SPACE == "score" ? scores :
                error("PLOT_SCORE_INTERP_SPACE must be 'score' or 'loginfid', got '$SCORE_INTERP_SPACE'")
            plot_values = values_mode === :score && SCORE_INTERP_SPACE == "loginfid" ?
                score_from_log_values(interp_values) : interp_values
            levels = values_mode === :logscore ? range(log_floor, 0.0; length=21) :
                                                  range(0.0, 1.0; length=21)
            if INTERPOLATE
                xgrid, ygrid, z = INTERP_METHOD == "smooth" ?
                    gaussian_smoothed_grid(xs, ys, interp_values, axis_x, axis_y) :
                    INTERP_METHOD == "smooth_max" ?
                    gaussian_max_preserving_grid(xs, ys, interp_values, axis_x, axis_y) :
                    idw_interpolated_grid(xs, ys, interp_values, axis_x, axis_y)
                values_mode === :score && SCORE_INTERP_SPACE == "loginfid" &&
                    (z = score_from_log_values(z))
                contour_plot = contourf!(ax, xgrid, ygrid, z;
                                         levels=levels,
                                         colormap=:inferno)
            else
                contour_plot = tricontourf!(ax, xs, ys, plot_values;
                                            levels=levels,
                                            colormap=:inferno)
            end
            xlims!(ax, axis_limits(axis_x, xs)...)
            ylims!(ax, axis_limits(axis_y, ys)...)
            set_axis_ticks!(ax, axis_x, axis_y)

            if SHOW_PANEL_LABELS && pulse_idx == 1
                panel_label!(ax, "$(Char(Int('a') + pair_idx - 1)))")
            end
        end
    end

    cbar_label = values_mode === :logscore ?
        (SHORT_SCORE_LABEL ? L"\log_{10}(1-\mathrm{Score})" :
         L"\log_{10}(1-\mathrm{Score}_{\mathrm{full\;L1}})") :
        (SHORT_SCORE_LABEL ? L"\mathrm{Score}" :
         L"\mathrm{Score}_{\mathrm{full\;L1}}")
    cbar_rows = VERTICAL_LAYOUT ? (1:N_PANEL_ROWS) :
        (SHOW_ROW_LABELS ? (2:(2 * length(PULSE_SPECS))) : (1:length(PULSE_SPECS)))
    Colorbar(fig[cbar_rows, N_PANEL_COLS + 1], contour_plot;
             label=cbar_label,
             width=24,
             labelsize=PAPER_COLORBAR_LABELSIZE,
             ticklabelsize=PAPER_COLORBAR_TICKSIZE)

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 12)
    return fig
end

function main()
    CairoMakie.activate!(type="png")
    log_floor, best_nonone = shared_log_floor()
    println("Full-L1 shared log clamp floor = $log_floor from best score < 1: $best_nonone")

    mkpath(FIGURE_DIR)
    fig_height = if VERTICAL_LAYOUT
        340 * N_PANEL_ROWS
    elseif SHOW_ROW_LABELS
        500 * length(PULSE_SPECS) - 20
    else
        420 * length(PULSE_SPECS)
    end
    fig_width = 470 * N_PANEL_COLS + 260

    score_fig = Figure(size=(fig_width, fig_height), figure_padding=(30, 36, 20, 24),
                       fontsize=PAPER_FIG_FONTSIZE)
    draw_figure!(score_fig, :score; log_floor=log_floor)
    save(SCORE_OUTPATH, score_fig; px_per_unit=PX_PER_UNIT)
    println("Saved $SCORE_OUTPATH")

    return SCORE_OUTPATH
end

main()
