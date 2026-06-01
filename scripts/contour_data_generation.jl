# scripts/contour_data_generation.jl
#
# Upstream data generator for paper figure figures/paper/mainpaper_contour_2ms.png:
# produces the broad varms_2 probe heatmaps that refined_contour_data_generation.jl
# merges into data/varms_2_refined_merged_heatmap_*.csv.
#
# Export deterministic population-score heatmaps for repeated MS probes:
#   * Q_varMS-style 1 x MS0(pi/2)
#   * Q_varMS-style 2 x MS0(pi/2)
#
# The scans use the same axes and score convention as the Jacobian probe
# contour figure: score = 1 - |P_SS - P_SS,nom| - |P_DD - P_DD,nom|.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO_ROOT, "data")
const METADATA_PATH = joinpath(DATA_DIR, "jacobian_probe_metadata.csv")
const OUTPUT_METADATA_PATH = joinpath(DATA_DIR, "varms_probe_metadata.csv")
const TWO_PI_KHZ = 2pi * 1.0e3

function env_int(name::String, default::Int)
    raw = get(ENV, name, string(default))
    value = try
        parse(Int, raw)
    catch
        throw(ArgumentError("$name must be an integer, got '$raw'"))
    end
    value >= 1 || throw(ArgumentError("$name must be >= 1, got $value"))
    return value
end

const N_WORKERS = env_int("VARMS_SCAN_WORKERS", max(1, Sys.CPU_THREADS - 1))
const N_HEATMAP = env_int("VARMS_HEATMAP_N", 31)
const NUM_MS_VALUES = (1, 2)
const HEATMAP_PAIRS = (
    (:rabi, :sideband),
    (:rabi, :phase),
    (:rabi, :fcl),
    (:sideband, :phase),
    (:sideband, :fcl),
    (:phase, :fcl),
)

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
        if isnan(value)
            return "NaN"
        elseif isinf(value)
            return value > 0 ? "Inf" : "-Inf"
        end
        return @sprintf("%.17g", value)
    elseif value isa Integer || value isa Symbol
        return string(value)
    else
        text = string(value)
        if occursin('"', text) || occursin(',', text) || occursin('\n', text) || occursin('\r', text)
            return "\"" * replace(text, "\"" => "\"\"") * "\""
        end
        return text
    end
end

function write_csv(path::String, columns::Vector{String}, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join((csv_field(row[col]) for col in columns), ","))
        end
    end
    return path
end

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

function metadata_value(metadata, name::String)
    haskey(metadata, name) || error("Missing metadata column: $name")
    isempty(metadata[name]) && error("Empty metadata column: $name")
    return parse(Float64, first(metadata[name]))
end

scan_range(start::Float64, stop::Float64, n::Int) =
    n == 1 ? [0.5 * (start + stop)] : collect(range(start, stop; length=n))

axis_values(axis::Symbol, n::Int) = if axis === :rabi
    scan_range(0.8, 1.2, n)
elseif axis === :phase
    scan_range(-0.5, 0.5, n)
elseif axis === :sideband
    scan_range(-2.0, 2.0, n)
elseif axis === :fcl
    scan_range(-2.0, 2.0, n)
else
    error("Unknown scan axis: $axis")
end

nominal_parameters() = (
    rabi_ratio=1.0,
    phase_pi=0.0,
    sideband_2pi_khz=0.0,
    fcl_2pi_khz=0.0,
)

function with_axis_value(params, axis::Symbol, value::Float64)
    if axis === :rabi
        return merge(params, (rabi_ratio=value,))
    elseif axis === :phase
        return merge(params, (phase_pi=value,))
    elseif axis === :sideband
        return merge(params, (sideband_2pi_khz=value,))
    elseif axis === :fcl
        return merge(params, (fcl_2pi_khz=value,))
    end
    error("Unknown scan axis: $axis")
end

function with_axis_pair(axis_x::Symbol, value_x::Float64, axis_y::Symbol, value_y::Float64)
    return with_axis_value(with_axis_value(nominal_parameters(), axis_x, value_x), axis_y, value_y)
end

function make_heatmap_jobs(numMS::Int, axis_x::Symbol, axis_y::Symbol, n::Int)
    jobs = NamedTuple[]
    xvals = axis_values(axis_x, n)
    yvals = axis_values(axis_y, n)
    scan_name = string(axis_x, "_", axis_y)
    for (j, y) in enumerate(yvals), (i, x) in enumerate(xvals)
        params = with_axis_pair(axis_x, Float64(x), axis_y, Float64(y))
        push!(jobs, merge(params, (
            numMS=numMS,
            sequence_name="varms_$(numMS)",
            sequence_label="$(numMS) x MS0(pi/2)",
            scan_kind="heatmap",
            scan_name=scan_name,
            x_axis=string(axis_x),
            y_axis=string(axis_y),
            x_index=i,
            y_index=j,
        )))
    end
    return jobs
end

function nominal_job(numMS::Int)
    return merge(nominal_parameters(), (
        numMS=numMS,
        sequence_name="varms_$(numMS)",
        sequence_label="$(numMS) x MS0(pi/2)",
        scan_kind="nominal",
        scan_name="nominal",
        x_axis="",
        y_axis="",
        x_index=0,
        y_index=0,
    ))
end

@everywhere function _varms_normalized_populations(pops)
    weights = Float64[
        max(Float64(pops.gg), 0.0),
        max(Float64(pops.eg), 0.0),
        max(Float64(pops.ge), 0.0),
        max(Float64(pops.ee), 0.0),
    ]
    total = sum(weights)
    total > 0.0 || error("Population weights have non-positive total.")
    weights ./= total
    gg, eg, ge, ee = weights
    odd = eg + ge
    return (gg=gg, ee=ee, eg=eg, ge=ge, odd=odd, balance=gg - ee)
end

@everywhere function _varms_eval_job(job)
    f_cl = _VARMS_F_CL0 + Float64(job.fcl_2pi_khz) * _VARMS_TWO_PI_KHZ
    f_sb = _VARMS_F_SB0 + Float64(job.sideband_2pi_khz) * _VARMS_TWO_PI_KHZ
    subgates = [CalibrationCode.MSSubgate(pi / 2, 0.0) for _ in 1:Int(job.numMS)]
    pulses = CalibrationCode.build_closed_loop_ms_sequence(
        _VARMS_T_US,
        f_cl,
        f_sb,
        _VARMS_I_PI2,
        subgates;
        omega_ratio=Float64(job.rabi_ratio),
        relative_phase=Float64(job.phase_pi) * pi,
    )
    probs = _varms_normalized_populations(CalibrationCode.populations_ms_sequence(pulses))
    return merge(job, (
        gg=probs.gg,
        ee=probs.ee,
        eg=probs.eg,
        ge=probs.ge,
        odd=probs.odd,
        balance=probs.balance,
    ))
end

function run_jobs(jobs)
    results = pmap(_varms_eval_job, jobs; batch_size=1)
    return [Dict(string(key) => value for (key, value) in pairs(row)) for row in results]
end

optimizer_score(row, nominal) =
    clamp(1.0 - abs(row["gg"] - nominal["gg"]) - abs(row["ee"] - nominal["ee"]), 0.0, 1.0)

function add_score_columns!(rows, nominal)
    for row in rows
        row["nominal_gg"] = nominal["gg"]
        row["nominal_ee"] = nominal["ee"]
        row["score"] = optimizer_score(row, nominal)
    end
    return rows
end

heatmap_path(numMS::Int, axis_x::Symbol, axis_y::Symbol) =
    joinpath(DATA_DIR, "varms_$(numMS)_heatmap_$(axis_x)_$(axis_y).csv")

heatmap_pairs_string() =
    join((string(axis_x, "_", axis_y) for (axis_x, axis_y) in HEATMAP_PAIRS), ";")

heatmap_files_string(numMS::Int) =
    join((basename(heatmap_path(numMS, axis_x, axis_y)) for (axis_x, axis_y) in HEATMAP_PAIRS), ";")

const COLUMNS = [
    "scan_kind",
    "sequence_name",
    "sequence_label",
    "numMS",
    "scan_name",
    "x_axis",
    "y_axis",
    "x_index",
    "y_index",
    "rabi_ratio",
    "phase_pi",
    "sideband_2pi_khz",
    "fcl_2pi_khz",
    "gg",
    "ee",
    "eg",
    "ge",
    "odd",
    "balance",
    "nominal_gg",
    "nominal_ee",
    "score",
]

function main()
    isfile(METADATA_PATH) || error("Missing metadata file: $METADATA_PATH. Run scripts/export_jacobian_probe_data.jl first.")
    metadata = read_generated_csv(METADATA_PATH)
    t_us = metadata_value(metadata, "t_us")
    f_cl0 = metadata_value(metadata, "f_cl0")
    f_sb0 = metadata_value(metadata, "f_sb0")
    i_pi2 = metadata_value(metadata, "I_pi2_calibrated")

    @everywhere const _VARMS_T_US = $t_us
    @everywhere const _VARMS_F_CL0 = $f_cl0
    @everywhere const _VARMS_F_SB0 = $f_sb0
    @everywhere const _VARMS_I_PI2 = $i_pi2
    @everywhere const _VARMS_TWO_PI_KHZ = $TWO_PI_KHZ

    metadata_rows = Dict{String,Any}[]
    for numMS in NUM_MS_VALUES
        nominal = first(run_jobs([nominal_job(numMS)]))
        nominal["nominal_gg"] = nominal["gg"]
        nominal["nominal_ee"] = nominal["ee"]
        nominal["score"] = 1.0
        println("Nominal varMS=$numMS: gg=$(nominal["gg"]), ee=$(nominal["ee"]), odd=$(nominal["odd"])")

        for (axis_x, axis_y) in HEATMAP_PAIRS
            outpath = heatmap_path(numMS, axis_x, axis_y)
            println("Evaluating varMS=$numMS heatmap $(axis_x)/$(axis_y): $(N_HEATMAP * N_HEATMAP) jobs")
            rows = run_jobs(make_heatmap_jobs(numMS, axis_x, axis_y, N_HEATMAP))
            length(rows) == N_HEATMAP * N_HEATMAP ||
                error("Unexpected row count for varMS=$numMS $(axis_x)/$(axis_y): $(length(rows))")
            add_score_columns!(rows, nominal)
            write_csv(outpath, COLUMNS, rows)
            println("Saved $outpath ($(length(rows)) rows)")
        end

        push!(metadata_rows, Dict(
            "numMS" => numMS,
            "t_us" => t_us,
            "f_cl0" => f_cl0,
            "f_sb0" => f_sb0,
            "I_pi2_calibrated" => i_pi2,
            "nominal_gg" => nominal["gg"],
            "nominal_ee" => nominal["ee"],
            "nominal_odd" => nominal["odd"],
            "nominal_balance" => nominal["balance"],
            "heatmap_n" => N_HEATMAP,
            "heatmap_pairs" => heatmap_pairs_string(),
            "heatmap_files" => heatmap_files_string(numMS),
            "sideband_units" => "2pi_khz",
            "fcl_units" => "2pi_khz",
            "phase_units" => "pi",
            "rabi_units" => "omega_ratio",
        ))
    end

    metadata_columns = [
        "numMS",
        "t_us",
        "f_cl0",
        "f_sb0",
        "I_pi2_calibrated",
        "nominal_gg",
        "nominal_ee",
        "nominal_odd",
        "nominal_balance",
        "heatmap_n",
        "heatmap_pairs",
        "heatmap_files",
        "sideband_units",
        "fcl_units",
        "phase_units",
        "rabi_units",
    ]
    write_csv(OUTPUT_METADATA_PATH, metadata_columns, metadata_rows)
    println("Saved metadata CSV: $OUTPUT_METADATA_PATH")
end

try
    main()
finally
    rmprocs(workers())
end
