import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using PyPlot
using Statistics

# ── Parse benchmark result files ─────────────────────────────────────────────

"""
    parse_benchmark(path) -> (total_shots, q_det, iterations)

Read a benchmark result file and extract TotalShots, Q_det, and Iterations
from the tab-separated results table.
"""
function parse_benchmark(path::String)
    lines = readlines(path)
    header_idx = findfirst(l -> startswith(l, "Sim\t"), lines)
    header_idx === nothing && error("No results table found in $path")

    total_shots = Int[]
    q_det = Float64[]
    iterations = Int[]
    for i in (header_idx+1):length(lines)
        line = lines[i]
        isempty(strip(line)) && continue
        cols = split(line, '\t')
        length(cols) < 6 && continue
        push!(iterations, parse(Int, cols[3]))        # Iterations
        push!(total_shots, parse(Int, cols[4]))        # TotalShots
        push!(q_det, parse(Float64, cols[6]))          # Q_det
    end
    return (total_shots, q_det, iterations)
end

# ── Collect data ─────────────────────────────────────────────────────────────

data_dir = joinpath(@__DIR__, "data")

fixed_N_values = [50, 100, 250, 500, 1000, 2500]
fixed_files = [joinpath(data_dir, "benchmark_fixedN$(N).txt") for N in fixed_N_values]

for f in fixed_files
    isfile(f) || error("Missing benchmark file: $f")
end

# Filter out non-converged trials (those that ran all 120 iterations)
fixed_data = []
for f in fixed_files
    shots_raw, qdet_raw, iters_raw = parse_benchmark(f)
    keep = iters_raw .< 120
    push!(fixed_data, (shots_raw[keep], qdet_raw[keep]))
end

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = subplots(figsize=(10, 7))

# Individual fixed-N trials (light blue, all pooled)
let first_static = true
    for (shots, qdet) in fixed_data
        infidelity = max.(1.0 .- qdet, 1e-6)
        lbl = first_static ? "Static trials" : nothing
        ax.scatter(log10.(shots), log10.(infidelity),
                   c="lightskyblue", alpha=0.35, s=20, zorder=2,
                   label=lbl, edgecolors="none")
        first_static = false
    end
end

# Mean +/- sigma for each fixed-N group (large blue circles with error bars)
let first_static_mean = true
    for (shots, qdet) in fixed_data
        infidelity = max.(1.0 .- qdet, 1e-6)
        log_shots = log10.(Float64.(shots))
        log_infid = log10.(infidelity)

        mx = mean(log_shots)
        sx = std(log_shots)
        my = mean(log_infid)
        sy = std(log_infid)

        lbl = first_static_mean ? "Static noise mean \$\\pm \\sigma\$" : nothing
        ax.errorbar(mx, my, xerr=sx, yerr=sy,
                    fmt="o", color="royalblue", markersize=10,
                    markeredgecolor="darkblue", markeredgewidth=1.0,
                    capsize=5, capthick=1.5, elinewidth=1.5, zorder=5,
                    label=lbl)
        first_static_mean = false
    end
end

ax.set_xlabel(L"\log_{10}(\mathrm{Total\;Shots})", fontsize=14)
ax.set_ylabel(L"\log_{10}(1 - \mathrm{Fidelity})", fontsize=14)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(true, which="both", linestyle=":", alpha=0.5)
ax.tick_params(labelsize=12)

tight_layout()

out_path = joinpath(@__DIR__, "plots", "benchmark_comparison.png")
savefig(out_path, dpi=200)
println("Plot saved to: $out_path")
close(fig)
