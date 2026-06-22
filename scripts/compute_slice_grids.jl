"""
Compute Q_det on three 2D slices through the optimum for background contour maps.
The slices are in normalized BO coordinates (u ∈ [-1, 1] per dimension).

Physical mapping:
  fcl = f_cl0 + span_fcl * u1   (u1 ∈ [-1,1])
  fsb = f_sb0 + span_fsb * u2   (u2 ∈ [-1,1])
  A   = A0    + span_A   * u3   (u3 ∈ [-1,1])

Slices are taken through a fixed point (u1_fix, u2_fix, u3_fix), defaulting to
the median final x_rec from the UCB trace data (≈ 0.001, -0.001, 0.030).

Override via ENV:
  SLICE_U1_FIX   — fixed u1 for the u2-u3 slice
  SLICE_U2_FIX   — fixed u2 for the u1-u3 slice
  SLICE_U3_FIX   — fixed u3 for the u1-u2 slice
  SLICE_N_GRID   — grid points per axis (default 71)
  SLICE_METRIC   — "q_varMS" (default) or "q_det"
  SLICE_N_NOISY  — shots for q_varMS (default 100000; std ≈ 0.003 at Q=0.999, minimal noise)
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Distributed
using Printf
using Statistics

if nprocs() == 1
    addprocs(max(1, Sys.CPU_THREADS - 1))
end

@everywhere begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
    include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
    using LinearAlgebra
    using StatsBase
    BLAS.set_num_threads(1)
end

# ── Settings ──────────────────────────────────────────────────────────────────
const N_GRID   = parse(Int,     get(ENV, "SLICE_N_GRID",   "71"))
const METRIC   =                get(ENV, "SLICE_METRIC",   "q_varMS")
const N_NOISY  = parse(Int,     get(ENV, "SLICE_N_NOISY",  "100000"))

# Fixed slice values: median final x_rec from UCB traces ≈ (0.001, -0.001, 0.030)
const U1_FIX   = parse(Float64, get(ENV, "SLICE_U1_FIX",  "0.0"))
const U2_FIX   = parse(Float64, get(ENV, "SLICE_U2_FIX",  "-0.0"))
const U3_FIX   = parse(Float64, get(ENV, "SLICE_U3_FIX",  "0.0"))

t = 100.0
base = CalibrationCode.ideal(t)
f_cl0, f_sb0, A0 = base.f_cl, base.f_sb, base.A

span_kHz = 2.0
span_fcl = span_kHz * 1e3 * 2π
span_fsb = span_kHz * 1e3 * 2π
span_A   = 1.2 * A0 - A0

@everywhere const _t       = $t
@everywhere const _f_cl0   = $f_cl0
@everywhere const _f_sb0   = $f_sb0
@everywhere const _A0      = $A0
@everywhere const _sfcl    = $span_fcl
@everywhere const _sfsb    = $span_fsb
@everywhere const _sA      = $span_A
@everywhere const _metric  = $METRIC
@everywhere const _N_noisy = $N_NOISY

println("=== Slice grid computation ===")
println("  Metric  : $METRIC$(METRIC == "q_det" ? " (deterministic)" : " (N=$N_NOISY shots)")")
println("  Grid    : $(N_GRID)×$(N_GRID) per slice")
println("  Fix pt  : u1=$U1_FIX  u2=$U2_FIX  u3=$U3_FIX")
println("  Workers : $(nworkers())")
println()

# ── Evaluation function ───────────────────────────────────────────────────────
@everywhere function eval_metric(u1, u2, u3)
    fcl = _f_cl0 + _sfcl * u1
    fsb = _f_sb0 + _sfsb * u2
    A   = _A0    + _sA   * u3
    if _metric == "q_det"
        return clamp(CalibrationCode.Q_det(_t, fcl, fsb, A), 0.0, 1.0)
    else  # q_varMS
        return clamp(CalibrationCode.Q_noisy(_t, fcl, fsb, A; phi_1=0.0, phi_2=0.0, N=_N_noisy), 0.0, 1.0)
    end
end

# ── Grid axes ─────────────────────────────────────────────────────────────────
u_axis = LinRange(-1.0, 1.0, N_GRID)

# ── Slice 1: u1-u2 plane at fixed u3 ─────────────────────────────────────────
println("Computing u1-u2 slice (fixed u3=$U3_FIX) …")
flush(stdout)
pts_12 = [(u1, u2) for u1 in u_axis, u2 in u_axis][:]
vals_12 = pmap(pts_12; batch_size=4) do (u1, u2)
    eval_metric(u1, u2, U3_FIX)
end
grid_12 = reshape(vals_12, N_GRID, N_GRID)   # [u1_idx, u2_idx]

# ── Slice 2: u1-u3 plane at fixed u2 ─────────────────────────────────────────
println("Computing u1-u3 slice (fixed u2=$U2_FIX) …")
flush(stdout)
pts_13 = [(u1, u3) for u1 in u_axis, u3 in u_axis][:]
vals_13 = pmap(pts_13; batch_size=4) do (u1, u3)
    eval_metric(u1, U2_FIX, u3)
end
grid_13 = reshape(vals_13, N_GRID, N_GRID)   # [u1_idx, u3_idx]

# ── Slice 3: u2-u3 plane at fixed u1 ─────────────────────────────────────────
println("Computing u2-u3 slice (fixed u1=$U1_FIX) …")
flush(stdout)
pts_23 = [(u2, u3) for u2 in u_axis, u3 in u_axis][:]
vals_23 = pmap(pts_23; batch_size=4) do (u2, u3)
    eval_metric(U1_FIX, u2, u3)
end
grid_23 = reshape(vals_23, N_GRID, N_GRID)   # [u2_idx, u3_idx]

# ── Statistics ────────────────────────────────────────────────────────────────
println()
@printf("u1-u2 slice: min=%.4f  max=%.4f  mean=%.4f\n",
        minimum(grid_12), maximum(grid_12), mean(grid_12))
@printf("u1-u3 slice: min=%.4f  max=%.4f  mean=%.4f\n",
        minimum(grid_13), maximum(grid_13), mean(grid_13))
@printf("u2-u3 slice: min=%.4f  max=%.4f  mean=%.4f\n",
        minimum(grid_23), maximum(grid_23), mean(grid_23))

# Point at the fix location
q_at_fix = eval_metric(U1_FIX, U2_FIX, U3_FIX)
@printf("Metric at fix point (%.4f, %.4f, %.4f): %.6f\n",
        U1_FIX, U2_FIX, U3_FIX, q_at_fix)

# ── Save CSVs ─────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "data", "slices")
mkpath(outdir)

u_vals = collect(u_axis)

# Slice u1-u2: rows = u1, cols = u2
open(joinpath(outdir, "slice_u1u2.csv"), "w") do io
    println(io, "# u1-u2 slice at u3=$(U3_FIX)  metric=$(METRIC)  grid=$(N_GRID)")
    println(io, "# axis_a=u1  axis_b=u2")
    println(io, "u1,u2,q")
    for (i, u1) in enumerate(u_vals), (j, u2) in enumerate(u_vals)
        @printf(io, "%.6f,%.6f,%.6f\n", u1, u2, grid_12[i, j])
    end
end

# Slice u1-u3: rows = u1, cols = u3
open(joinpath(outdir, "slice_u1u3.csv"), "w") do io
    println(io, "# u1-u3 slice at u2=$(U2_FIX)  metric=$(METRIC)  grid=$(N_GRID)")
    println(io, "# axis_a=u1  axis_b=u3")
    println(io, "u1,u3,q")
    for (i, u1) in enumerate(u_vals), (j, u3) in enumerate(u_vals)
        @printf(io, "%.6f,%.6f,%.6f\n", u1, u3, grid_13[i, j])
    end
end

# Slice u2-u3: rows = u2, cols = u3
open(joinpath(outdir, "slice_u2u3.csv"), "w") do io
    println(io, "# u2-u3 slice at u1=$(U1_FIX)  metric=$(METRIC)  grid=$(N_GRID)")
    println(io, "# axis_a=u2  axis_b=u3")
    println(io, "u2,u3,q")
    for (i, u2) in enumerate(u_vals), (j, u3) in enumerate(u_vals)
        @printf(io, "%.6f,%.6f,%.6f\n", u2, u3, grid_23[i, j])
    end
end

# Metadata file (used by the Python plotting script)
open(joinpath(outdir, "slice_meta.txt"), "w") do io
    println(io, "metric=$METRIC")
    println(io, "n_grid=$N_GRID")
    println(io, "u1_fix=$U1_FIX")
    println(io, "u2_fix=$U2_FIX")
    println(io, "u3_fix=$U3_FIX")
    println(io, "n_noisy=$N_NOISY")
    println(io, "q_at_fix=$q_at_fix")
end

println()
println("Saved to scripts/data/slices/:")
println("  slice_u1u2.csv  (u1-u2 at u3=$U3_FIX)")
println("  slice_u1u3.csv  (u1-u3 at u2=$U2_FIX)")
println("  slice_u2u3.csv  (u2-u3 at u1=$U1_FIX)")
println("  slice_meta.txt")

rmprocs(workers())
