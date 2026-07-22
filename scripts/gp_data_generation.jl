# scripts/gp_data_generation.jl
#
# Data generator for python_scripts/plot_gp_figure.py.
#
# For each N (shot count), picks the N_SEEDS_FOR_AVERAGE BO traces whose final
# Q_det is closest to the median across all seeds in
# data/traces_freqspan10_bound050_N<N>_..., then replays each trace's training
# set at several snapshot iterations and fits a GP at each snapshot. The
# python side averages these seeds' GP curves to smooth out seed-to-seed
# fitting variance. Writes:
#   data/gp_slices/scale05_selectedN_gp_slices_iters10_30_50.csv
#   data/gp_slices/scale05_selectedN_gp_slices_iters10_30_50_selected_seeds.csv
#   data/score_cache/scale05_true_2ms_slices_freqspan10_grid201.csv
#
# Matches run_main.sh: FREQ_SPAN_KHZ=10, BOUND_SCALE=0.5, N_INIT=12.
#
# Each (N, seed, iter) snapshot GP fit is independent, so this runs them via
# Distributed/pmap across N_WORKERS processes.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed
using Printf

function env_int(name::String, default::Int)
    value = parse(Int, get(ENV, name, string(default)))
    value >= 1 || throw(ArgumentError("$name must be >= 1, got $value"))
    return value
end

const N_WORKERS = env_int("GP_DATA_N_WORKERS", max(1, Sys.CPU_THREADS - 1))
if nprocs() == 1
    addprocs(N_WORKERS)
end

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

@everywhere begin
    import Pkg
    Pkg.activate(joinpath($REPO_ROOT); io=devnull)
    include(joinpath($REPO_ROOT, "src", "CalibrationCode.jl"))
    using Random
    using Statistics

    const DATA_DIR = joinpath($REPO_ROOT, "data")
    const SNAPSHOT_ITERS = [10, 30, 50]
    const AXES = (:fcl, :fsb, :amp)

    const N_INIT = 12
    const BOUND_SCALE = 0.5
    const GRID_N = parse(Int, get(ENV, "MEDIAN_GP_GRID_N", "201"))
    const GP_N_RESTARTS = parse(Int, get(ENV, "MEDIAN_GP_N_RESTARTS", "6"))

    const T_GATE = 100.0
    const BASE = CalibrationCode.ideal(T_GATE)
    const F_CL0 = Float64(BASE.f_cl)
    const F_SB0 = Float64(BASE.f_sb)
    const A0 = Float64(BASE.A)
    const FREQ_SPAN_HZ = 10.0e3
    const SPAN_FCL = FREQ_SPAN_HZ
    const SPAN_FSB = FREQ_SPAN_HZ
    const SPAN_A = 1.2 * A0 - A0

    parse_float(value::AbstractString) = parse(Float64, value)

    axis_name(axis::Symbol) = axis === :fcl ? "fcl" : axis === :fsb ? "fsb" : axis === :amp ? "amp" :
        error("Unsupported axis $axis")

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

    u_to_params(u) = (fcl=F_CL0 + SPAN_FCL * Float64(u[1]), fsb=F_SB0 + SPAN_FSB * Float64(u[2]), A=A0 + SPAN_A * Float64(u[3]))
    objective(u, N) = (p = u_to_params(u); CalibrationCode.Q_varMS(T_GATE, p.fcl, p.fsb, p.A; N=N))
    qdet(u) = objective(u, Inf)[1]
    n_value(n_label::String) = n_label == "Inf" ? Inf : parse(Int, n_label)

    trace_dir(n_label::String) = joinpath(DATA_DIR, "traces_freqspan10_bound050_N$(n_label)_nostop100_stream_100seeds")

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

    function read_trace(path::String)
        rows = read_csv(path)
        sort!(rows; by=row -> parse(Int, row["iter"]))
        return rows
    end

    # main_opt.jl's run_seed does `Random.seed!(seed)` then calls bayesopt_ucb(seed=seed),
    # which draws the initial Latin-hypercube design from a fresh MersenneTwister(seed)
    # (a stream independent of the global RNG) and then evaluates the objective at each
    # init point using the *global* RNG (untouched until then, since CENTER_JITTER_U_MAX=0
    # means no RNG draws happen first). Replaying that exact sequence reproduces the
    # initial design's observations bit-for-bit; every later iteration's (x_acq, y_acq)
    # is already stored in the trace, so only its (deterministic) noise std needs recomputing.
    function reconstruct_training(n_label::String, seed::Int, snapshot_iter::Int)
        N = n_value(n_label)
        lb = fill(-BOUND_SCALE, 3)
        ub = fill(BOUND_SCALE, 3)

        Random.seed!(seed)
        rng = MersenneTwister(seed)
        init_points = CalibrationCode._latin_hypercube_points(rng, lb, ub, N_INIT)

        n_cap = N_INIT + snapshot_iter
        X = Matrix{Float64}(undef, 3, n_cap)
        y = Vector{Float64}(undef, n_cap)
        sigma_y = Vector{Float64}(undef, n_cap)
        write_idx = 0
        for x in init_points
            write_idx += 1
            X[:, write_idx] = x
            y[write_idx], sigma_y[write_idx] = objective(x, N)
        end

        for row in read_trace(joinpath(trace_dir(n_label), "trace_seed$(seed)_ucb.csv"))
            iter = parse(Int, row["iter"])
            iter > snapshot_iter && break
            x = [parse_float(row["x_acq_u1"]), parse_float(row["x_acq_u2"]), parse_float(row["x_acq_u3"])]
            CalibrationCode._is_far_enough(x, X, write_idx) || continue
            write_idx += 1
            X[:, write_idx] = x
            y[write_idx] = parse_float(row["y_acq"])
            _, sigma_y[write_idx] = objective(x, N)
        end

        return X[:, 1:write_idx], y[1:write_idx], sigma_y[1:write_idx]
    end

    function fit_snapshot_gp(n_label::String, seed::Int, snapshot_iter::Int)
        X, y, sigma_y = reconstruct_training(n_label, seed, snapshot_iter)
        rng = MersenneTwister(seed + 100_000 * snapshot_iter)
        gp = CalibrationCode.fit_heterogp(X, y, sigma_y;
            n_restarts=GP_N_RESTARTS, jitter=1e-8, rng=rng,
            ℓ_bounds=(0.05, 1.5), σf_bounds=(0.3, 2.0), c_bounds=(0.05, 3.0))
        return gp, size(X, 2)
    end

    function predict_axis_rows(gp, n_label, seed, selected_rank, iter, n_train, axis)
        grid = collect(range(-BOUND_SCALE, BOUND_SCALE; length=GRID_N))
        rows = NamedTuple[]
        for x_u in grid
            mu, s2 = CalibrationCode.predict_latent(gp, axis_point(axis, Float64(x_u)))
            push!(rows, (
                N=n_label, seed=seed, selected_rank=selected_rank, iter=iter, n_train=n_train,
                axis=axis_name(axis), x_u=Float64(x_u),
                x_physical=axis_physical(axis, Float64(x_u)), physical_unit=axis_unit(axis),
                mu=mu, sigma=sqrt(max(s2, 0.0)),
            ))
        end
        return rows
    end

    # One independent unit of work: fit the GP for this (N, seed, iter) snapshot
    # and predict all 3 axes.
    function run_snapshot_job(job)
        gp, n_train = fit_snapshot_gp(job.n_label, job.seed, job.iter)
        rows = NamedTuple[]
        for axis in AXES
            append!(rows, predict_axis_rows(gp, job.n_label, job.seed, job.rank, job.iter, n_train, axis))
        end
        return rows
    end
end

const CACHE_DIR = joinpath(DATA_DIR, "score_cache")
const GP_SLICE_DIR = joinpath(DATA_DIR, "gp_slices")
const N_LABELS = ("100", "1000", "10000", "100000", "Inf")
const N_SEEDS_FOR_AVERAGE = parse(Int, get(ENV, "GP_AVG_SEEDS", "5"))

function csv_field(value)
    if value isa AbstractFloat
        isnan(value) && return "NaN"
        isinf(value) && return value > 0 ? "Inf" : "-Inf"
        return @sprintf("%.17g", value)
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

# ── True (N=Inf) score cache, shared across all N labels ──────────────────────

true_score_cache_path() = joinpath(CACHE_DIR, "scale05_true_2ms_slices_freqspan10_grid$(GRID_N).csv")

function load_or_write_true_score_cache()
    path = true_score_cache_path()
    expected_rows = length(AXES) * GRID_N
    if isfile(path) && length(read_csv(path)) == expected_rows
        println("Loaded true-score cache: $path")
        return
    end
    println("Computing true 2MS score slices for cache...")
    grid = collect(range(-BOUND_SCALE, BOUND_SCALE; length=GRID_N))
    rows = NamedTuple[]
    for axis in AXES, x_u in grid
        push!(rows, (
            axis=axis_name(axis), x_u=Float64(x_u),
            x_physical=axis_physical(axis, Float64(x_u)), physical_unit=axis_unit(axis),
            score=qdet(axis_point(axis, Float64(x_u))),
        ))
    end
    write_namedtuple_csv(path, rows)
    println("Saved true-score cache: $path")
end

# ── Trace loading + median-seed selection ──────────────────────────────────────

function seed_from_trace_path(path::String)
    m = match(r"trace_seed(\d+)_ucb\.csv$", basename(path))
    m === nothing && error("Could not parse seed from $path")
    return parse(Int, m.captures[1])
end

function trace_paths(dir::String)
    isdir(dir) || error("Missing trace directory: $dir")
    paths = sort(filter(p -> occursin(r"trace_seed\d+_ucb\.csv$", basename(p)), readdir(dir; join=true)))
    isempty(paths) && error("No trace files in $dir")
    return paths
end

final_qdet(path::String) = parse_float(read_trace(path)[end]["q_det_rec"])

# Picks the N_SEEDS_FOR_AVERAGE seeds whose final q_det is closest to the
# median across all seeds for this N. GP curves are averaged over these to
# smooth out the seed-to-seed GP-fitting variance (worst at small n_train).
function median_seeds(n_label::String)
    dir = trace_dir(n_label)
    rows = [(seed=seed_from_trace_path(p), path=p, q_det=final_qdet(p)) for p in trace_paths(dir)]
    med = median([r.q_det for r in rows])
    sort!(rows; by=r -> abs(r.q_det - med))
    selected = rows[1:min(N_SEEDS_FOR_AVERAGE, length(rows))]
    return selected, med, dir
end

function gp_csv_paths()
    stem = "scale05_selectedN_gp_slices_iters$(join(SNAPSHOT_ITERS, "_"))"
    return (gp=joinpath(GP_SLICE_DIR, "$stem.csv"), selected=joinpath(GP_SLICE_DIR, "$(stem)_selected_seeds.csv"))
end

function collect_gp_slice_rows()
    jobs = NamedTuple[]
    selected_rows = NamedTuple[]
    for n_label in N_LABELS
        selected, med, dir = median_seeds(n_label)
        @printf("N=%s: median final Q_det %.8f; selected seeds %s\n",
                n_label, med, join((s.seed for s in selected), ","))
        for (rank, info) in enumerate(selected)
            push!(selected_rows, (N=n_label, selected_rank=rank, seed=info.seed,
                                  final_q_det=info.q_det, median_q_det=med, trace_dir=dir))
            for iter in SNAPSHOT_ITERS
                push!(jobs, (n_label=n_label, seed=info.seed, rank=rank, iter=iter))
            end
        end
    end

    println("Fitting $(length(jobs)) GP snapshots across $(nworkers()) workers...")
    results = pmap(jobs; batch_size=1) do job
        println("Starting N=$(job.n_label) seed=$(job.seed) rank=$(job.rank) iter=$(job.iter)")
        flush(stdout)
        rows = run_snapshot_job(job)
        println("Finished N=$(job.n_label) seed=$(job.seed) rank=$(job.rank) iter=$(job.iter)")
        flush(stdout)
        return rows
    end
    output_rows = reduce(vcat, results)

    paths = gp_csv_paths()
    write_namedtuple_csv(paths.gp, output_rows)
    write_namedtuple_csv(paths.selected, selected_rows)
    println("Saved GP slice data: $(paths.gp)")
    println("Saved selected seed table: $(paths.selected)")
end

function main()
    load_or_write_true_score_cache()
    paths = gp_csv_paths()
    if isfile(paths.gp) && isfile(paths.selected) &&
       lowercase(get(ENV, "RECOMPUTE_SELECTEDN_GP_SLICES", "false")) ∉ ("1", "true", "yes", "on")
        println("Loaded GP slice data: $(paths.gp)")
    else
        collect_gp_slice_rows()
    end
end

try
    main()
finally
    rmprocs(workers())
end
