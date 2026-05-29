# Data generator for paper figure figures/paper/mainpaper_contour_2ms.png.
# Writes data/varms_2_refined_merged_heatmap_{rabi_sideband,rabi_fcl,sideband_fcl}.csv.
#
# Refined transition-region scans for the Jacobian probe and 2 x MS0(pi/2).
#
# This script keeps the existing broad heatmaps, adds higher-accuracy samples in
# a central transition window, and writes one merged point-cloud CSV per
# parameter pair and pulse. The companion plotting script uses triangulated
# interpolation, so the merged CSVs do not need to be rectangular grids.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO_ROOT, "data")
const SEARCH_RESULT_PATH = joinpath(DATA_DIR, "ms_sequence_search_result.jl")
const JAC_METADATA_PATH = joinpath(DATA_DIR, "jacobian_probe_metadata.csv")
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

function env_float(name::String, default::Float64)
    raw = get(ENV, name, string(default))
    value = try
        parse(Float64, raw)
    catch
        throw(ArgumentError("$name must be numeric, got '$raw'"))
    end
    isfinite(value) || throw(ArgumentError("$name must be finite, got $value"))
    return value
end

const N_WORKERS = env_int("REFINE_SCAN_WORKERS", min(max(Sys.CPU_THREADS - 1, 1), 5))
const REFINE_GRID_N = env_int("REFINE_GRID_N", 13)
const FOCK_CUTOFF = env_int("REFINE_FOCK_CUTOFF", 20)
const ABSTOL = env_float("REFINE_ABSTOL", 1e-11)
const RELTOL = env_float("REFINE_RELTOL", 1e-11)
const MIN_INFidelity = 1e-15

const RABI_SPAN = env_float("REFINE_RABI_SPAN", 0.12)
const PHASE_SPAN_PI = env_float("REFINE_PHASE_SPAN_PI", 0.25)
const SIDEBAND_SPAN_KHZ = env_float("REFINE_SIDEBAND_SPAN_KHZ", 1.0)
const FCL_SPAN_KHZ = env_float("REFINE_FCL_SPAN_KHZ", 1.0)

const HEATMAP_PAIRS = (
    (:rabi, :sideband),
    (:rabi, :phase),
    (:rabi, :fcl),
    (:sideband, :phase),
    (:sideband, :fcl),
    (:phase, :fcl),
)

const PULSE_SPECS = (
    (
        name="jacobian_probe",
        label="Jacobian",
        existing_prefix="jacobian_probe_heatmap",
        merged_prefix="jacobian_probe_refined_merged_heatmap",
    ),
    (
        name="varms_2",
        label="2 x MS0(pi/2)",
        existing_prefix="varms_2_heatmap",
        merged_prefix="varms_2_refined_merged_heatmap",
    ),
)

if nprocs() == 1
    addprocs(N_WORKERS)
end

@everywhere begin
    import Pkg
    Pkg.activate(joinpath($REPO_ROOT); io=devnull)
    Base.include(Main, joinpath($REPO_ROOT, "src", "CalibrationCode.jl"))
end

const CC = Main.CalibrationCode

function csv_field(value)
    if value isa AbstractFloat
        if isnan(value)
            return "NaN"
        elseif isinf(value)
            return value > 0 ? "Inf" : "-Inf"
        end
        return @sprintf("%.17g", value)
    end
    text = string(value)
    if occursin('"', text) || occursin(',', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
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

function maybe_float(rows, name::String, idx::Int, default::Float64=NaN)
    haskey(rows, name) || return default
    parsed = tryparse(Float64, rows[name][idx])
    parsed === nothing && return default
    return parsed
end

axis_span(axis::Symbol) = if axis === :rabi
    RABI_SPAN
elseif axis === :phase
    PHASE_SPAN_PI
elseif axis === :sideband
    SIDEBAND_SPAN_KHZ
elseif axis === :fcl
    FCL_SPAN_KHZ
else
    error("Unknown axis: $axis")
end

axis_center(axis::Symbol) = axis === :rabi ? 1.0 : 0.0

scan_range(center::Float64, span::Float64, n::Int) =
    n == 1 ? [center] : collect(range(center - span, center + span; length=n))

axis_values(axis::Symbol, n::Int) =
    scan_range(axis_center(axis), axis_span(axis), n)

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
    error("Unknown axis: $axis")
end

function with_axis_pair(axis_x::Symbol, value_x::Float64, axis_y::Symbol, value_y::Float64)
    return with_axis_value(with_axis_value(nominal_parameters(), axis_x, value_x), axis_y, value_y)
end

function make_refined_jobs(pulse_name::String, pulse_label::String, axis_x::Symbol, axis_y::Symbol)
    jobs = NamedTuple[]
    xvals = axis_values(axis_x, REFINE_GRID_N)
    yvals = axis_values(axis_y, REFINE_GRID_N)
    scan_name = string(axis_x, "_", axis_y)
    for (j, y) in enumerate(yvals), (i, x) in enumerate(xvals)
        params = with_axis_pair(axis_x, Float64(x), axis_y, Float64(y))
        push!(jobs, merge(params, (
            pulse_name=pulse_name,
            pulse_label=pulse_label,
            source="refined_transition",
            scan_kind="transition_refined",
            scan_name=scan_name,
            x_axis=string(axis_x),
            y_axis=string(axis_y),
            x_index=i,
            y_index=j,
        )))
    end
    return jobs
end

existing_heatmap_path(spec, axis_x::Symbol, axis_y::Symbol) =
    joinpath(DATA_DIR, "$(spec.existing_prefix)_$(axis_x)_$(axis_y).csv")

existing_jacobian_high_accuracy_path(axis_x::Symbol, axis_y::Symbol) =
    joinpath(DATA_DIR, "jacobian_probe_high_accuracy_heatmap_$(axis_x)_$(axis_y).csv")

merged_heatmap_path(spec, axis_x::Symbol, axis_y::Symbol) =
    joinpath(DATA_DIR, "$(spec.merged_prefix)_$(axis_x)_$(axis_y).csv")

function normalized_weights_from_rho(rho)
    weights = Float64[
        max(real(rho[1, 1]), 0.0),
        max(real(rho[2, 2]), 0.0),
        max(real(rho[3, 3]), 0.0),
        max(real(rho[4, 4]), 0.0),
    ]
    total = sum(weights)
    total > 0.0 || error("Population weights have non-positive total.")
    weights ./= total
    gg, eg, ge, ee = weights
    return (gg=gg, ee=ee, eg=eg, ge=ge, odd=eg + ge, balance=gg - ee)
end

score_from_target(gg::Float64, ee::Float64, nominal_gg::Float64, nominal_ee::Float64) =
    clamp(1.0 - abs(gg - nominal_gg) - abs(ee - nominal_ee), 0.0, 1.0)

score_infidelity(score::Float64) = clamp(1.0 - score, MIN_INFidelity, 1.0)

@everywhere function _refined_reduced_density(pulses)
    setup = CalibrationCode.build_chamber()
    CalibrationCode.modecutoff!(setup.chamber, _REFINE_FOCK_CUTOFF)
    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    state = CalibrationCode.tensor(ca["S"], ca["S"], mode[0])
    for pulse in pulses
        CalibrationCode.configure_lasers!(
            setup,
            pulse.f_cl,
            pulse[3],
            pulse.I;
            phi_1=pulse.phi_1,
            phi_2=pulse.phi_2,
        )
        h = CalibrationCode.hamiltonian(
            chamber;
            timescale=1e-6,
            lamb_dicke_order=1,
            rwa_cutoff=Inf,
        )
        _, sol = CalibrationCode.timeevolution.schroedinger_dynamic(
            Float64[0.0, pulse.t],
            state,
            h;
            alg=CalibrationCode.Vern7(),
            abstol=_REFINE_ABSTOL,
            reltol=_REFINE_RELTOL,
        )
        state = sol[end]
    end
    return CalibrationCode.ptrace(CalibrationCode.tensor(state, CalibrationCode.dagger(state)), [3]).data
end

@everywhere function _refined_eval_job(job)
    if String(job.pulse_name) == "jacobian_probe"
        subgates = [CalibrationCode.MSSubgate(item.theta, item.phi) for item in _REFINE_JAC_SUBGATES]
        nominal_gg = _REFINE_JAC_TARGET_GG
        nominal_ee = _REFINE_JAC_TARGET_EE
    elseif String(job.pulse_name) == "varms_2"
        subgates = [CalibrationCode.MSSubgate(pi / 2, 0.0), CalibrationCode.MSSubgate(pi / 2, 0.0)]
        nominal_gg = _REFINE_VARMS2_TARGET_GG
        nominal_ee = _REFINE_VARMS2_TARGET_EE
    else
        error("Unknown pulse_name: $(job.pulse_name)")
    end

    f_cl = _REFINE_F_CL0 + Float64(job.fcl_2pi_khz) * _REFINE_TWO_PI_KHZ
    f_sb = _REFINE_F_SB0 + Float64(job.sideband_2pi_khz) * _REFINE_TWO_PI_KHZ
    pulses = CalibrationCode.build_closed_loop_ms_sequence(
        _REFINE_T_US,
        f_cl,
        f_sb,
        _REFINE_I_PI2,
        subgates;
        omega_ratio=Float64(job.rabi_ratio),
        relative_phase=Float64(job.phase_pi) * pi,
    )
    rho = _refined_reduced_density(pulses)
    weights = Float64[
        max(real(rho[1, 1]), 0.0),
        max(real(rho[2, 2]), 0.0),
        max(real(rho[3, 3]), 0.0),
        max(real(rho[4, 4]), 0.0),
    ]
    total = sum(weights)
    total > 0.0 || error("Population weights have non-positive total.")
    weights ./= total
    gg, eg, ge, ee = weights
    score = clamp(1.0 - abs(gg - nominal_gg) - abs(ee - nominal_ee), 0.0, 1.0)
    score_inf = clamp(1.0 - score, _REFINE_MIN_INFIDELITY, 1.0)
    return merge(job, (
        gg=gg,
        ee=ee,
        eg=eg,
        ge=ge,
        odd=eg + ge,
        balance=gg - ee,
        nominal_gg=nominal_gg,
        nominal_ee=nominal_ee,
        score=score,
        score_infidelity=score_inf,
        log10_score_infidelity=log10(score_inf),
        fock_cutoff=_REFINE_FOCK_CUTOFF,
        abstol=_REFINE_ABSTOL,
        reltol=_REFINE_RELTOL,
    ))
end

function run_jobs(jobs)
    results = pmap(_refined_eval_job, jobs; batch_size=1)
    return [Dict(string(key) => value for (key, value) in pairs(row)) for row in results]
end

function normalize_existing_rows(path::String, spec, axis_x::Symbol, axis_y::Symbol,
                                 nominal_gg::Float64, nominal_ee::Float64,
                                 source::String)
    rows = read_generated_csv(path)
    n = length(first(values(rows)))
    out = Dict{String,Any}[]
    for idx in 1:n
        gg = maybe_float(rows, "gg", idx)
        ee = maybe_float(rows, "ee", idx)
        eg = maybe_float(rows, "eg", idx)
        ge = maybe_float(rows, "ge", idx)
        odd = isfinite(maybe_float(rows, "odd", idx)) ? maybe_float(rows, "odd", idx) : eg + ge
        balance = isfinite(maybe_float(rows, "balance", idx)) ? maybe_float(rows, "balance", idx) : gg - ee
        score = score_from_target(gg, ee, nominal_gg, nominal_ee)
        score_inf = score_infidelity(score)
        push!(out, Dict{String,Any}(
            "pulse_name" => spec.name,
            "pulse_label" => spec.label,
            "source" => source,
            "scan_kind" => get(rows, "scan_kind", fill("heatmap", n))[idx],
            "scan_name" => string(axis_x, "_", axis_y),
            "x_axis" => string(axis_x),
            "y_axis" => string(axis_y),
            "x_index" => Int(round(maybe_float(rows, "x_index", idx, 0.0))),
            "y_index" => Int(round(maybe_float(rows, "y_index", idx, 0.0))),
            "rabi_ratio" => maybe_float(rows, "rabi_ratio", idx, 1.0),
            "phase_pi" => maybe_float(rows, "phase_pi", idx, 0.0),
            "sideband_2pi_khz" => maybe_float(rows, "sideband_2pi_khz", idx, 0.0),
            "fcl_2pi_khz" => maybe_float(rows, "fcl_2pi_khz", idx, 0.0),
            "gg" => gg,
            "ee" => ee,
            "eg" => eg,
            "ge" => ge,
            "odd" => odd,
            "balance" => balance,
            "nominal_gg" => nominal_gg,
            "nominal_ee" => nominal_ee,
            "score" => score,
            "score_infidelity" => score_inf,
            "log10_score_infidelity" => log10(score_inf),
            "fock_cutoff" => maybe_float(rows, "fock_cutoff", idx, NaN),
            "abstol" => maybe_float(rows, "abstol", idx, NaN),
            "reltol" => maybe_float(rows, "reltol", idx, NaN),
        ))
    end
    return out
end

source_priority(source) = if source == "refined_transition"
    3
elseif source == "existing_high_accuracy"
    2
else
    1
end

coord_key(row) = join((
    @sprintf("%.12g", row["rabi_ratio"]),
    @sprintf("%.12g", row["phase_pi"]),
    @sprintf("%.12g", row["sideband_2pi_khz"]),
    @sprintf("%.12g", row["fcl_2pi_khz"]),
), "|")

function dedupe_rows(rows)
    best = Dict{String,Dict{String,Any}}()
    for row in rows
        key = coord_key(row)
        if !haskey(best, key) || source_priority(row["source"]) >= source_priority(best[key]["source"])
            best[key] = row
        end
    end
    out = collect(values(best))
    sort!(out, by=row -> (row["sideband_2pi_khz"], row["fcl_2pi_khz"], row["phase_pi"], row["rabi_ratio"]))
    for (idx, row) in enumerate(out)
        row["sample_index"] = idx
    end
    return out
end

const COLUMNS = [
    "pulse_name",
    "pulse_label",
    "source",
    "scan_kind",
    "scan_name",
    "x_axis",
    "y_axis",
    "sample_index",
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
    "score_infidelity",
    "log10_score_infidelity",
    "fock_cutoff",
    "abstol",
    "reltol",
]

function nominal_populations(t_us::Float64, f_cl0::Float64, f_sb0::Float64, i_pi2::Float64,
                             subgates)
    pulses = CC.build_closed_loop_ms_sequence(t_us, f_cl0, f_sb0, i_pi2, subgates)
    rho = _refined_reduced_density(pulses)
    return normalized_weights_from_rho(rho)
end

function main()
    isfile(SEARCH_RESULT_PATH) || error("Missing search result: $SEARCH_RESULT_PATH")
    isfile(JAC_METADATA_PATH) || error("Missing metadata: $JAC_METADATA_PATH")

    search_result = include(SEARCH_RESULT_PATH)
    metadata = read_generated_csv(JAC_METADATA_PATH)
    t_us = Float64(get(search_result, :t, metadata_value(metadata, "t_us")))
    f_cl0 = metadata_value(metadata, "f_cl0")
    f_sb0 = metadata_value(metadata, "f_sb0")
    i_pi2 = metadata_value(metadata, "I_pi2_calibrated")

    jac_subgates = [CC.MSSubgate(Float64(sg.theta), Float64(sg.phi))
                    for sg in search_result.best_overall.subgates]
    jac_subgate_literals = [(theta=sg.theta, phi=sg.phi) for sg in jac_subgates]
    varms2_subgates = [CC.MSSubgate(pi / 2, 0.0), CC.MSSubgate(pi / 2, 0.0)]

    @everywhere const _REFINE_T_US = $t_us
    @everywhere const _REFINE_F_CL0 = $f_cl0
    @everywhere const _REFINE_F_SB0 = $f_sb0
    @everywhere const _REFINE_I_PI2 = $i_pi2
    @everywhere const _REFINE_TWO_PI_KHZ = $TWO_PI_KHZ
    @everywhere const _REFINE_JAC_SUBGATES = $jac_subgate_literals
    @everywhere const _REFINE_FOCK_CUTOFF = $FOCK_CUTOFF
    @everywhere const _REFINE_ABSTOL = $ABSTOL
    @everywhere const _REFINE_RELTOL = $RELTOL
    @everywhere const _REFINE_MIN_INFIDELITY = $MIN_INFidelity

    jac_nominal = nominal_populations(t_us, f_cl0, f_sb0, i_pi2, jac_subgates)
    varms2_nominal = nominal_populations(t_us, f_cl0, f_sb0, i_pi2, varms2_subgates)

    @everywhere const _REFINE_JAC_TARGET_GG = $(jac_nominal.gg)
    @everywhere const _REFINE_JAC_TARGET_EE = $(jac_nominal.ee)
    @everywhere const _REFINE_VARMS2_TARGET_GG = $(varms2_nominal.gg)
    @everywhere const _REFINE_VARMS2_TARGET_EE = $(varms2_nominal.ee)

    println("Refined transition scans")
    println("workers = $N_WORKERS, grid_n = $REFINE_GRID_N, fock_cutoff = $FOCK_CUTOFF")
    println("abstol = $ABSTOL, reltol = $RELTOL")
    println("focus spans: rabi +/-$RABI_SPAN, phase +/-$PHASE_SPAN_PI pi, sideband +/-$SIDEBAND_SPAN_KHZ kHz, fcl +/-$FCL_SPAN_KHZ kHz")
    println("Jacobian target: gg=$(jac_nominal.gg), ee=$(jac_nominal.ee)")
    println("2MS target: gg=$(varms2_nominal.gg), ee=$(varms2_nominal.ee)")

    for spec in PULSE_SPECS
        nominal = spec.name == "jacobian_probe" ? jac_nominal : varms2_nominal
        for (axis_x, axis_y) in HEATMAP_PAIRS
            combined_rows = Dict{String,Any}[]
            base_path = existing_heatmap_path(spec, axis_x, axis_y)
            isfile(base_path) || error("Missing existing heatmap: $base_path")
            append!(combined_rows, normalize_existing_rows(
                base_path, spec, axis_x, axis_y, nominal.gg, nominal.ee, "existing_coarse"))

            if spec.name == "jacobian_probe"
                hi_path = existing_jacobian_high_accuracy_path(axis_x, axis_y)
                if isfile(hi_path)
                    append!(combined_rows, normalize_existing_rows(
                        hi_path, spec, axis_x, axis_y, nominal.gg, nominal.ee, "existing_high_accuracy"))
                end
            end

            jobs = make_refined_jobs(spec.name, spec.label, axis_x, axis_y)
            println("Evaluating $(spec.label) $(axis_x)/$(axis_y): $(length(jobs)) refined jobs")
            refined_rows = run_jobs(jobs)
            append!(combined_rows, refined_rows)

            merged_rows = dedupe_rows(combined_rows)
            outpath = merged_heatmap_path(spec, axis_x, axis_y)
            write_csv(outpath, COLUMNS, merged_rows)
            best_score = maximum(row["score"] for row in merged_rows)
            println("Saved $outpath ($(length(merged_rows)) rows, best score = $best_score)")
        end
    end
end

try
    main()
finally
    rmprocs(workers())
end
