# scripts/contour_data_generation.jl
#
# Data generator for python_scripts/plot_contour.py.
# Writes deterministic (N=Inf) population-score heatmaps for the closed-loop
# 2×MS(π/2) sequence over the three axis pairs plot_contour.py reads:
#   data/varms_2_heatmap_rabi_sideband.csv
#   data/varms_2_heatmap_rabi_fcl.csv
#   data/varms_2_heatmap_sideband_fcl.csv
#
# Score = Q_infinity = p_ee (population in |DD>), the same infinite-shot fidelity
# used by main_opt.jl / gp_data_generation.jl (Q_varMS with N=Inf). varms_weights
# normalizes (p_ss, p_sd, p_ds, p_dd) to sum to 1, so this is exactly what
# Q_varMS(...; N=Inf) reduces to.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO_ROOT, "data")
const T_GATE = 100.0
const NUM_MS = 2
const TWO_PI_KHZ = 2pi * 1.0e3

function env_int(name::String, default::Int)
    value = parse(Int, get(ENV, name, string(default)))
    value >= 1 || throw(ArgumentError("$name must be >= 1, got $value"))
    return value
end

const N_WORKERS = env_int("VARMS_SCAN_WORKERS", max(1, Sys.CPU_THREADS - 1))
const N_HEATMAP = env_int("VARMS_HEATMAP_N", 31)

# Only the three axis pairs plot_contour.py (and plot_gp_figure.py's contour
# fallback) actually read; "phase" scans aren't consumed anywhere.
const HEATMAP_PAIRS = ((:rabi, :sideband), (:rabi, :fcl), (:sideband, :fcl))

if nprocs() == 1
    addprocs(N_WORKERS)
end

@everywhere begin
    import Pkg
    Pkg.activate(joinpath($REPO_ROOT); io=devnull)
    Base.include(Main, joinpath($REPO_ROOT, "src", "CalibrationCode.jl"))
end

function csv_field(value)
    if value isa AbstractFloat
        isnan(value) && return "NaN"
        isinf(value) && return value > 0 ? "Inf" : "-Inf"
        return @sprintf("%.17g", value)
    end
    return string(value)
end

function write_csv(path::String, columns::Vector{String}, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join((csv_field(getproperty(row, Symbol(col))) for col in columns), ","))
        end
    end
    return path
end

scan_range(start::Float64, stop::Float64, n::Int) =
    n == 1 ? [0.5 * (start + stop)] : collect(range(start, stop; length=n))

axis_values(axis::Symbol, n::Int) = if axis === :rabi
    scan_range(0.8, 1.2, n)
elseif axis === :sideband
    scan_range(-2.0, 2.0, n)
elseif axis === :fcl
    scan_range(-2.0, 2.0, n)
else
    error("Unknown scan axis: $axis")
end

nominal_parameters() = (rabi_ratio=1.0, sideband_2pi_khz=0.0, fcl_2pi_khz=0.0)

function with_axis_value(params, axis::Symbol, value::Float64)
    axis === :rabi && return merge(params, (rabi_ratio=value,))
    axis === :sideband && return merge(params, (sideband_2pi_khz=value,))
    axis === :fcl && return merge(params, (fcl_2pi_khz=value,))
    error("Unknown scan axis: $axis")
end

function make_heatmap_jobs(axis_x::Symbol, axis_y::Symbol, n::Int)
    jobs = NamedTuple[]
    xvals = axis_values(axis_x, n)
    yvals = axis_values(axis_y, n)
    for (j, y) in enumerate(yvals), (i, x) in enumerate(xvals)
        params = with_axis_value(with_axis_value(nominal_parameters(), axis_x, x), axis_y, y)
        push!(jobs, merge(params, (
            scan_kind="heatmap", x_axis=string(axis_x), y_axis=string(axis_y),
            x_index=i, y_index=j,
        )))
    end
    return jobs
end

@everywhere function _varms_eval_job(job)
    f_cl = _VARMS_F_CL0 + Float64(job.fcl_2pi_khz) * _VARMS_TWO_PI_KHZ
    f_sb = _VARMS_F_SB0 + Float64(job.sideband_2pi_khz) * _VARMS_TWO_PI_KHZ
    w = CalibrationCode.varms_weights(_VARMS_T_US, f_cl, f_sb, _VARMS_I_PI2 * Float64(job.rabi_ratio); numMS=_VARMS_NUM_MS)
    gg, eg, ge, ee = w
    return merge(job, (gg=gg, ee=ee, eg=eg, ge=ge, score=ee))
end

run_jobs(jobs) = pmap(_varms_eval_job, jobs; batch_size=1)

heatmap_path(axis_x::Symbol, axis_y::Symbol) =
    joinpath(DATA_DIR, "varms_$(NUM_MS)_heatmap_$(axis_x)_$(axis_y).csv")

const COLUMNS = [
    "scan_kind", "x_axis", "y_axis", "x_index", "y_index",
    "rabi_ratio", "sideband_2pi_khz", "fcl_2pi_khz",
    "gg", "ee", "eg", "ge", "score",
]

function main()
    t_us = T_GATE
    base = CalibrationCode.ideal(T_GATE)
    f_cl0, f_sb0, i_pi2 = Float64(base.f_cl), Float64(base.f_sb), Float64(base.A)

    @everywhere const _VARMS_T_US = $t_us
    @everywhere const _VARMS_F_CL0 = $f_cl0
    @everywhere const _VARMS_F_SB0 = $f_sb0
    @everywhere const _VARMS_I_PI2 = $i_pi2
    @everywhere const _VARMS_TWO_PI_KHZ = $TWO_PI_KHZ
    @everywhere const _VARMS_NUM_MS = $NUM_MS

    for (axis_x, axis_y) in HEATMAP_PAIRS
        outpath = heatmap_path(axis_x, axis_y)
        println("Evaluating $(axis_x)/$(axis_y) heatmap: $(N_HEATMAP * N_HEATMAP) jobs")
        rows = run_jobs(make_heatmap_jobs(axis_x, axis_y, N_HEATMAP))
        write_csv(outpath, COLUMNS, rows)
        println("Saved $outpath ($(length(rows)) rows)")
    end
end

try
    main()
finally
    rmprocs(workers())
end
