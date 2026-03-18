# scripts/compare_benchmarks.jl
# Compare multiple benchmark result files

using Plots
using StatsPlots
using Statistics
using LinearAlgebra
using Printf

"""
    parse_benchmark_file(filepath)

Parse a benchmark results file and extract key statistics.
If individual simulation rows are available, summary statistics are recomputed
from converged runs only (`iterations < max_iterations`).
Returns a dictionary with:
- name: filename
- avg_qdet: average Q_det value
- std_qdet: standard deviation of Q_det
- avg_iterations: average iterations
- std_iterations: standard deviation of iterations
- n_sims: number of simulations
- min_qdet, max_qdet, median_qdet
"""
function parse_benchmark_file(filepath; max_iterations::Int=120)
    lines = readlines(filepath)

    safe_std(v::Vector{Float64}) = length(v) > 1 ? std(v) : 0.0

    result = Dict{String,Any}(
        "name" => basename(filepath),
        "avg_qdet" => NaN,
        "std_qdet" => NaN,
        "avg_iterations" => NaN,
        "std_iterations" => NaN,
        "n_sims" => 0,
        "min_qdet" => NaN,
        "max_qdet" => NaN,
        "median_qdet" => NaN,
        "avg_shots" => NaN,
        "std_shots" => NaN
    )

    # Track if we're in the individual results section
    in_results_section = false
    sigma_levels = Float64[]
    shots_per_sim = Float64[]
    converged_qdet = Float64[]
    converged_iters = Float64[]
    converged_shots = Float64[]

    for (i, line) in enumerate(lines)
        # Parse average Q_det with standard deviation
        if contains(line, "Average Q_det")
            m = match(r"Average Q_det = ([\d.]+) ± ([\d.]+)", line)
            if m !== nothing
                result["avg_qdet"] = parse(Float64, m.captures[1])
                result["std_qdet"] = parse(Float64, m.captures[2])
            end
        end

        # Parse average iterations with standard deviation
        if contains(line, "Average iterations")
            m = match(r"Average iterations = ([\d.]+) ± ([\d.]+)", line)
            if m !== nothing
                result["avg_iterations"] = parse(Float64, m.captures[1])
                result["std_iterations"] = parse(Float64, m.captures[2])
            end
        end

        # Parse number of simulations
        if contains(line, "Number of simulations")
            m = match(r"Number of simulations[:\s=]+(\d+)", line)
            if m !== nothing
                result["n_sims"] = parse(Int, m.captures[1])
            end
        end

        # Parse min/max/median Q_det
        if contains(line, "Min Q_det")
            m = match(r"Min Q_det = ([\d.]+)", line)
            if m !== nothing
                result["min_qdet"] = parse(Float64, m.captures[1])
            end
        end

        if contains(line, "Max Q_det")
            m = match(r"Max Q_det = ([\d.]+)", line)
            if m !== nothing
                result["max_qdet"] = parse(Float64, m.captures[1])
            end
        end

        if contains(line, "Median Q_det")
            m = match(r"Median Q_det = ([\d.]+)", line)
            if m !== nothing
                result["median_qdet"] = parse(Float64, m.captures[1])
            end
        end

        # Detect the individual results header to extract sigma levels
        if contains(line, "INDIVIDUAL RESULTS") || contains(line, "All Results")
            in_results_section = true
            # Next line should be the header
            if i < length(lines)
                header_line = lines[i+1]
                # Extract sigma values from header (e.g., "σ=0.1", "σ=0.05")
                sigma_matches = collect(eachmatch(r"σ=([\d.]+)", header_line))
                sigma_levels = [parse(Float64, m.captures[1]) for m in sigma_matches]
            end
            continue
        end

        # Parse individual simulation results
        if in_results_section && !isempty(sigma_levels)
            # Skip header line and empty lines
            if contains(line, "Sim\t") || strip(line) == ""
                continue
            end

            # Try to parse a data line
            parts = split(line, "\t")
            if length(parts) >= 4 + length(sigma_levels)
                try
                    iter = parse(Int, strip(parts[3]))
                    q_det = parse(Float64, strip(parts[4]))

                    # Extract calls for each sigma level (last columns)
                    total_shots = 0.0
                    for (j, sigma) in enumerate(sigma_levels)
                        calls = parse(Int, strip(parts[4+j]))
                        shots_per_level = calls * round(Int, 1 / sigma^2)
                        total_shots += shots_per_level
                    end

                    if iter < max_iterations
                        push!(converged_iters, float(iter))
                        push!(converged_qdet, q_det)
                        push!(converged_shots, total_shots)
                    end

                    push!(shots_per_sim, total_shots)
                catch
                    # Skip lines that can't be parsed
                    continue
                end
            end
        end
    end

    if !isempty(converged_qdet)
        result["avg_qdet"] = mean(converged_qdet)
        result["std_qdet"] = safe_std(converged_qdet)
        result["min_qdet"] = minimum(converged_qdet)
        result["max_qdet"] = maximum(converged_qdet)
        result["median_qdet"] = median(converged_qdet)

        result["avg_iterations"] = mean(converged_iters)
        result["std_iterations"] = safe_std(converged_iters)
    end

    # Calculate average and std of shots
    if !isempty(converged_shots)
        result["avg_shots"] = mean(converged_shots)
        result["std_shots"] = safe_std(converged_shots)
    elseif !isempty(shots_per_sim)
        result["avg_shots"] = mean(shots_per_sim)
        result["std_shots"] = safe_std(shots_per_sim)
    else
        # Fallback: infer shots/iter from filename (preferred) or sigma-style filename
        filename = basename(filepath)
        shots_match = match(r"onelevel_[Nn](\d+)", filename)
        sigma_match = match(r"onelevel_(\d+)", filename)
        if !isnan(result["avg_iterations"]) && shots_match !== nothing
            shots_per_iter = parse(Int, shots_match.captures[1])
            result["avg_shots"] = result["avg_iterations"] * shots_per_iter
            result["std_shots"] = result["std_iterations"] * shots_per_iter
        elseif !isnan(result["avg_iterations"]) && sigma_match !== nothing
            sigma_str = sigma_match.captures[1]
            sigma = parse(Float64, "0." * sigma_str)
            shots_per_iter = round(Int, 1 / sigma^2)
            result["avg_shots"] = result["avg_iterations"] * shots_per_iter
            result["std_shots"] = result["std_iterations"] * shots_per_iter
        end
    end

    return result
end


"""
    plot_benchmark_comparison(files; labels=nothing, output_file=nothing)

Create comparison plots for multiple benchmark result files.

# Arguments
- `files`: Array of file paths to compare
- `labels`: Optional custom labels for each file (default: use filenames)
- `output_file`: Optional path to save the plot (default: display only)
"""
function plot_benchmark_comparison(files::Vector{String};
    labels::Union{Vector{String},Nothing}=nothing,
    output_file::Union{String,Nothing}=nothing)

    # Parse all files
    results = [parse_benchmark_file(f) for f in files]

    # Create labels if not provided
    if labels === nothing
        labels = [replace(r["name"], ".txt" => "", "_" => " ") for r in results]
    end

    n_files = length(results)
    x_positions = 1:n_files

    # Create subplots
    p1 = plot(
        title="Average Q_det Comparison",
        xlabel="Benchmark",
        ylabel="Q_det",
        legend=false,
        size=(800, 400),
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3
    )

    # Plot average Q_det with error bars
    avg_qdet = [r["avg_qdet"] for r in results]
    std_qdet = [r["std_qdet"] for r in results]

    # Truncated (bounded) error bars for fidelity in [0, 1]
    # Use asymmetric errors so bars never exceed physical bounds.
    lower_err_qdet = [min(s, max(a, 0.0)) for (a, s) in zip(avg_qdet, std_qdet)]
    upper_err_qdet = [min(s, max(1.0 - a, 0.0)) for (a, s) in zip(avg_qdet, std_qdet)]

    # Calculate dynamic y-axis range for better visibility (zoomed from bottom)
    min_val = minimum(avg_qdet .- lower_err_qdet)
    y_padding = 0.02

    bar!(p1, x_positions, avg_qdet,
        yerr=(lower_err_qdet, upper_err_qdet),
        fillcolor=:lightblue,
        linecolor=:blue,
        xticks=(x_positions, labels),
        xrotation=45,
        ylabel="Q_det",
        ylim=(max(0.0, min_val - y_padding), 1.00))

    # Add data labels on top of bars (4 sig figs), positioned to always be visible
    for (i, (avg, up_err)) in enumerate(zip(avg_qdet, upper_err_qdet))
        label_y = min(avg + up_err + 0.005, 0.995)
        annotate!(p1, i, label_y,
            text(string(round(avg, sigdigits=4)), 8, :center))
    end

    # Plot iterations with error bars
    p2 = plot(
        title="Average Iterations Comparison",
        xlabel="Benchmark",
        ylabel="Iterations",
        legend=false,
        size=(800, 400),
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3
    )

    avg_iter = [r["avg_iterations"] for r in results]
    std_iter = [r["std_iterations"] for r in results]

    bar!(p2, x_positions, avg_iter,
        yerr=std_iter,
        fillcolor=:lightgreen,
        linecolor=:green,
        xticks=(x_positions, labels),
        xrotation=45,
        ylabel="Iterations")

    # Add data labels on top of bars
    for (i, (avg, std)) in enumerate(zip(avg_iter, std_iter))
        annotate!(p2, i, avg + std + max(avg * 0.05, 2),
            text(string(round(avg, digits=1)), 8, :center))
    end

    # Plot average shots
    p3 = plot(
        title="Average Shots per Simulation",
        xlabel="Benchmark",
        ylabel="Avg Shots",
        legend=false,
        size=(800, 400),
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3
    )

    avg_shots = [r["avg_shots"] for r in results]
    std_shots = [r["std_shots"] for r in results]
    plot_shots = [isnan(s) ? NaN : s for s in avg_shots]
    plot_std_shots = [isnan(s) ? NaN : std_shots[i] for (i, s) in enumerate(avg_shots)]

    bar!(p3, x_positions, plot_shots,
        yerr=plot_std_shots,
        fillcolor=:coral,
        linecolor=:red,
        xticks=(x_positions, labels),
        xrotation=45,
        ylabel="Avg Shots")

    # Add data labels on top of bars (skip missing values)
    for (i, (shots, std)) in enumerate(zip(avg_shots, std_shots))
        if !isnan(shots)
            annotate!(p3, i, shots + std + max(shots * 0.05, 100),
                text(string(round(Int, shots)), 8, :center))
        end
    end

    # Combine plots
    p = plot(p1, p2, p3, layout=(3, 1), size=(900, 1200),
        margin=10Plots.mm, bottom_margin=15Plots.mm)

    # Print summary statistics
    println("\n" * "="^70)
    println("BENCHMARK COMPARISON SUMMARY")
    println("="^70)
    println()

    for (i, (r, label)) in enumerate(zip(results, labels))
        println("[$i] $label")
        println("    Simulations:     $(r["n_sims"])")
        println("    Avg Q_det:       $(round(r["avg_qdet"], digits=4)) ± $(round(r["std_qdet"], digits=4))")
        println("    Min/Med/Max:     $(round(r["min_qdet"], digits=4)) / $(round(r["median_qdet"], digits=4)) / $(round(r["max_qdet"], digits=4))")
        println("    Avg Iterations:  $(round(r["avg_iterations"], digits=2)) ± $(round(r["std_iterations"], digits=2))")
        if !isnan(r["avg_shots"])
            println("    Avg Shots:       $(round(Int, r["avg_shots"])) ± $(round(Int, r["std_shots"]))")
        end
        println()
    end

    println("="^70)

    # Display or save plot
    if output_file !== nothing
        savefig(p, output_file)
        println("Plot saved to: $output_file")
    else
        display(p)
    end

    return p, results
end


"""
    plot_detailed_comparison(files; labels=nothing, output_file=nothing)

Create a more detailed comparison with multiple metrics side-by-side.
"""
function plot_detailed_comparison(files::Vector{String};
    labels::Union{Vector{String},Nothing}=nothing,
    output_file::Union{String,Nothing}=nothing)

    # Parse all files
    results = [parse_benchmark_file(f) for f in files]

    # Create labels if not provided
    if labels === nothing
        labels = [replace(r["name"], ".txt" => "", "_" => " ") for r in results]
    end

    n_files = length(results)
    x_positions = 1:n_files

    # Q_det statistics
    p1 = plot(title="Q_det Statistics", legend=:bottomright, size=(600, 400))
    avg_qdet = [r["avg_qdet"] for r in results]
    min_qdet = [r["min_qdet"] for r in results]
    max_qdet = [r["max_qdet"] for r in results]
    median_qdet = [r["median_qdet"] for r in results]

    plot!(p1, x_positions, avg_qdet,
        marker=:circle, markersize=8, linewidth=2, label="Average")
    plot!(p1, x_positions, median_qdet,
        marker=:square, markersize=6, linewidth=2, label="Median")
    plot!(p1, x_positions, min_qdet,
        marker=:dtriangle, markersize=6, linewidth=1, linestyle=:dash, label="Min")
    plot!(p1, x_positions, max_qdet,
        marker=:utriangle, markersize=6, linewidth=1, linestyle=:dash, label="Max")

    plot!(p1, xticks=(x_positions, labels), xrotation=45,
        ylabel="Q_det", ylim=(0, 1.1), grid=true)

    # Coefficient of variation (std/mean) for Q_det
    p2 = plot(title="Q_det Variability (CV)", legend=false, size=(600, 400))
    cv_qdet = [r["std_qdet"] / r["avg_qdet"] for r in results]
    bar!(p2, x_positions, cv_qdet,
        fillcolor=:orange, linecolor=:darkorange,
        xticks=(x_positions, labels), xrotation=45,
        ylabel="Coefficient of Variation")

    # Iterations comparison
    p3 = plot(title="Average Iterations", legend=false, size=(600, 400))
    avg_iter = [r["avg_iterations"] for r in results]
    std_iter = [r["std_iterations"] for r in results]
    bar!(p3, x_positions, avg_iter, yerr=std_iter,
        fillcolor=:lightgreen, linecolor=:green,
        xticks=(x_positions, labels), xrotation=45,
        ylabel="Iterations")

    # Average shots comparison
    p4 = plot(title="Average Shots per Simulation", legend=false, size=(600, 400))
    avg_shots = [r["avg_shots"] for r in results]
    std_shots = [r["std_shots"] for r in results]
    valid_indices = [i for (i, s) in enumerate(avg_shots) if !isnan(s)]

    if !isempty(valid_indices)
        valid_shots = [avg_shots[i] for i in valid_indices]
        valid_std_shots = [std_shots[i] for i in valid_indices]
        valid_labels_p4 = [labels[i] for i in valid_indices]
        valid_positions_p4 = 1:length(valid_indices)

        bar!(p4, valid_positions_p4, valid_shots,
            yerr=valid_std_shots,
            fillcolor=:coral, linecolor=:red,
            xticks=(valid_positions_p4, valid_labels_p4), xrotation=45,
            ylabel="Shots per Simulation")
    end

    # Efficiency metric (Q_det per shot)
    p5 = plot(title="Efficiency (Q_det / 1000 Shots)", legend=false, size=(600, 400))
    efficiency_shots = [!isnan(r["avg_shots"]) ? 1000.0 * r["avg_qdet"] / r["avg_shots"] : NaN for r in results]
    valid_indices_eff = [i for (i, e) in enumerate(efficiency_shots) if !isnan(e)]

    if !isempty(valid_indices_eff)
        valid_eff = [efficiency_shots[i] for i in valid_indices_eff]
        valid_labels_eff = [labels[i] for i in valid_indices_eff]
        valid_positions_eff = 1:length(valid_indices_eff)

        bar!(p5, valid_positions_eff, valid_eff,
            fillcolor=:purple, linecolor=:darkviolet,
            xticks=(valid_positions_eff, valid_labels_eff), xrotation=45,
            ylabel="Q_det per 1000 Shots")
    end

    # Efficiency metric per iteration (for comparison)
    p6 = plot(title="Efficiency (Q_det / Iteration)", legend=false, size=(600, 400))
    efficiency_iter = [r["avg_qdet"] / r["avg_iterations"] for r in results]
    bar!(p6, x_positions, efficiency_iter,
        fillcolor=:lightblue, linecolor=:blue,
        xticks=(x_positions, labels), xrotation=45,
        ylabel="Q_det per Iteration")

    # Combine plots
    p = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1400, 1200),
        margin=10Plots.mm, bottom_margin=15Plots.mm)

    # Display or save plot
    if output_file !== nothing
        savefig(p, output_file)
        println("Detailed plot saved to: $output_file")
    else
        display(p)
    end

    return p, results
end


"""
    get_single_level_sigma(filepath)

Extract single-level sigma from filename convention or file contents.
Supports both `onelevel_N{shots}` and legacy `onelevel_{sigmadigits}` naming.
"""
function get_single_level_sigma(filepath::String)
    filename = basename(filepath)

    shots_match = match(r"onelevel_[Nn](\d+)", filename)
    if shots_match !== nothing
        shots_per_call = parse(Int, shots_match.captures[1])
        return sqrt(1.0 / shots_per_call)
    end

    sigma_match = match(r"onelevel_(\d+)", filename)
    if sigma_match !== nothing
        return parse(Float64, "0." * sigma_match.captures[1])
    end

    for line in readlines(filepath)
        m = match(r"σ=([\d.]+)", line)
        if m !== nothing
            return parse(Float64, m.captures[1])
        end
    end

    return NaN
end


"""
    benchmark_label_from_file(filepath)

Create a display label with one-level naming based on shots per call N = Int(1/σ²).
"""
function benchmark_label_from_file(filepath::String)
    filename = lowercase(basename(filepath))

    if occursin("fourlevels", filename)
        return "Four-Level"
    elseif occursin("onelevel_", filename)
        sigma = get_single_level_sigma(filepath)
        if !isnan(sigma)
            shots_per_call = round(Int, 1 / sigma^2)
            return "One-Level N=$(shots_per_call)"
        end
        return "One-Level"
    end

    return replace(replace(basename(filepath), ".txt" => ""), "_" => " ")
end


"""
    parse_single_level_points(filepath)

Parse individual Q_det points from a single-level benchmark file.
Returns `(sigma, fidelities, iterations)`.
"""
function parse_single_level_points(filepath::String)
    lines = readlines(filepath)

    sigma = get_single_level_sigma(filepath)

    fidelities = Float64[]
    iterations = Int[]
    in_results_section = false

    for line in lines
        if contains(line, "=== All Results ===") || contains(line, "INDIVIDUAL RESULTS")
            in_results_section = true
            continue
        end

        if !in_results_section || strip(line) == "" || contains(line, "Sim\t")
            continue
        end

        parts = split(line, "\t")
        if length(parts) >= 4
            try
                iter = parse(Int, strip(parts[3]))
                q_det = parse(Float64, strip(parts[4]))
                push!(iterations, iter)
                push!(fidelities, q_det)
            catch
                continue
            end
        end
    end

    return sigma, fidelities, iterations
end


"""
    parse_four_level_effective_points(filepath; max_iterations=120)

Parse converged four-level points and map each simulation to an effective
shots-per-iteration value:
`N_eff = total_shots / iterations`, with `total_shots = Σ calls_i * round(1/σ_i^2)`.
Returns `(log10_N_eff, fidelities)`.
"""
function parse_four_level_effective_points(filepath::String; max_iterations::Int=120)
    lines = readlines(filepath)

    sigma_levels = Float64[]
    in_results_section = false

    x_vals = Float64[]
    y_vals = Float64[]

    for (i, line) in enumerate(lines)
        if contains(line, "INDIVIDUAL RESULTS") || contains(line, "=== All Results ===")
            in_results_section = true
            if i < length(lines)
                header_line = lines[i+1]
                sigma_matches = collect(eachmatch(r"σ=([\d.]+)", header_line))
                sigma_levels = [parse(Float64, m.captures[1]) for m in sigma_matches]
            end
            continue
        end

        if !in_results_section || isempty(sigma_levels) || strip(line) == "" || contains(line, "Sim\t")
            continue
        end

        parts = split(line, "\t")
        if length(parts) < 4 + length(sigma_levels)
            continue
        end

        try
            iter = parse(Int, strip(parts[3]))
            q_det = parse(Float64, strip(parts[4]))

            if iter == max_iterations
                continue
            end

            total_shots = 0.0
            for (j, sigma) in enumerate(sigma_levels)
                calls = parse(Int, strip(parts[4+j]))
                total_shots += calls * round(Int, 1 / sigma^2)
            end

            if iter > 0 && total_shots > 0
                n_eff = total_shots / iter
                push!(x_vals, log10(n_eff))
                push!(y_vals, q_det)
            end
        catch
            continue
        end
    end

    return x_vals, y_vals
end


"""
    plot_single_level_fidelity_vs_noise(files; output_file=nothing)

Create a box plot of fidelity vs. log10(shots per call) for single-level benchmark files,
using only converged points (excluding `iterations == max_iterations`).
Box plots show the distribution (median, quartiles, and outliers) for each noise level.
"""
function plot_single_level_fidelity_vs_noise(files::Vector{String};
    output_file::Union{String,Nothing}=nothing,
    max_iterations::Int=120)
    single_level_files = [f for f in files if occursin("onelevel_", lowercase(basename(f)))]
    four_level_files = [f for f in files if occursin("fourlevels", lowercase(basename(f)))]

    if isempty(single_level_files)
        error("No single-level benchmark files found in provided file list.")
    end

    x_inliers = Float64[]
    y_inliers = Float64[]

    for file in single_level_files
        sigma, fidelities, iters = parse_single_level_points(file)

        if isnan(sigma)
            continue
        end

        shots_per_call = round(Int, 1 / sigma^2)
        log_shots_per_call = log10(shots_per_call)

        for (fidelity, iter) in zip(fidelities, iters)
            if iter != max_iterations
                push!(x_inliers, log_shots_per_call)
                push!(y_inliers, fidelity)
            end
        end
    end

    p_main = plot(
        xlabel="Log(N)",
        ylabel="Fidelity",
        legend=false,
        size=(1200, 800),
        dpi=300,
        framestyle=:box,
        yformatter=y -> @sprintf("%.2f", y),
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3
    )

    # Group data by x-level for box plots
    if !isempty(x_inliers)
        level_map = Dict{Float64,Vector{Float64}}()
        for (x, y) in zip(x_inliers, y_inliers)
            if !haskey(level_map, x)
                level_map[x] = Float64[]
            end
            push!(level_map[x], y)
        end

        x_levels = sort(collect(keys(level_map)))

        # Create box plots for each noise level
        for (i, x) in enumerate(x_levels)
            boxplot!(p_main,
                fill(x, length(level_map[x])),
                level_map[x],
                fillcolor=:dodgerblue,
                fillalpha=0.5,
                linecolor=:blue,
                linewidth=1.5,
                whisker_width=:match,
                markercolor=:blue,
                markerstrokecolor=:blue,
                label="",
                bar_width=if length(x_levels) > 1
                    max(0.08, 0.4 * minimum(diff(x_levels)))
                else
                    0.15
                end)
        end
    end

    if !isempty(x_inliers)
        x_min = minimum(x_inliers)
        x_max = maximum(x_inliers)
        x_pad = max((x_max - x_min) * 0.1, 0.002)

        # Keep y-range physically valid in [0, 1] while adding modest visual padding.
        y_min = max(0.0, minimum(y_inliers) - 0.01)
        y_max = min(1.0, maximum(y_inliers) + 0.01)
        y_pad = max((y_max - y_min) * 0.12, 0.001)

        plot!(p_main, xlim=(x_min - x_pad, x_max + x_pad),
            ylim=(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad)))
    end

    plot!(p_main, ylim=(0.95, 1.0), yticks=[0.95, 0.96, 0.97, 0.98, 0.99, 1.0])

    p_multilevel = nothing

    if !isempty(four_level_files)
        x4, y4 = parse_four_level_effective_points(four_level_files[1], max_iterations=max_iterations)

        if !isempty(x4)
            p_inset = plot(
                xlabel="Log(N)",
                ylabel="Fidelity",
                legend=false,
                size=(900, 650),
                dpi=300,
                framestyle=:box,
                yformatter=y -> @sprintf("%.3f", y),
                grid=true,
                gridstyle=:dot,
                gridalpha=0.25,
                tickfontsize=6,
                guidefontsize=8
            )

            x4_mean = mean(x4)
            x4_min = minimum(x4)
            x4_max = maximum(x4)
            x4_pad = max((x4_max - x4_min) * 0.12, 0.002)
            x_box_left = x4_min - x4_pad
            x_box_right = x4_max + x4_pad

            # Create box plot for four-level data in inset
            boxplot!(p_inset,
                fill(x4_mean, length(y4)),
                y4,
                fillcolor=:red,
                fillalpha=0.5,
                linecolor=:darkred,
                linewidth=1.5,
                whisker_width=:match,
                markercolor=:red,
                markerstrokecolor=:darkred,
                label="",
                bar_width=max((x4_max - x4_min) * 0.8, 0.15))

            # Also overlay multi-level box plot on the main single-level plot
            boxplot!(p_main,
                fill(x4_mean, length(y4)),
                y4,
                fillcolor=:red,
                fillalpha=0.5,
                linecolor=:darkred,
                linewidth=1.5,
                whisker_width=:match,
                markercolor=:red,
                markerstrokecolor=:darkred,
                label="",
                bar_width=max((x4_max - x4_min) * 0.8, 0.15))

            plot!(p_inset,
                xlim=(x_box_left, x_box_right),
                ylim=(0.990, 1.0),
                yticks=[0.990, 0.995, 1.0])

            p_multilevel = p_inset
        end
    end

    if output_file !== nothing
        savefig(p_main, output_file)
        println("Single-level box plot saved to: $output_file")

        if p_multilevel !== nothing
            base, ext = splitext(output_file)
            output_file_multilevel = string(base, "_multilevel", ext)
            savefig(p_multilevel, output_file_multilevel)
            println("Multi-level box plot saved to: $output_file_multilevel")
        end
    else
        display(p_main)
        if p_multilevel !== nothing
            display(p_multilevel)
        end
    end

    return p_main
end


"""
    plot_shots_vs_fidelity_scatter(files; output_file=nothing, max_iterations=120)

Create a scatter plot of log₁₀(total shots) vs fidelity for all provided files.
Each file contributes one point at (mean log₁₀(shots), mean fidelity), with
asymmetric error bars for the standard deviation clamped to physical bounds
(fidelity ∈ [0, 1], log-shots > 0 log-space of at least 1 shot).

Static-noise (one-level) files are shown in blue; variable-noise (multi-level)
files are shown in red. Runs that did not converge (iterations == max_iterations)
are excluded.
"""
function plot_shots_vs_fidelity_scatter(files::Vector{String};
    output_file::Union{String,Nothing}=nothing,
    max_iterations::Int=120)

    # ---- collect per-file statistics AND individual trial points ------------
    static_x = Float64[]
    static_y = Float64[]
    static_xerr_lo = Float64[]
    static_xerr_hi = Float64[]
    static_yerr_lo = Float64[]
    static_yerr_hi = Float64[]

    variable_x = Float64[]
    variable_y = Float64[]
    variable_xerr_lo = Float64[]
    variable_xerr_hi = Float64[]
    variable_yerr_lo = Float64[]
    variable_yerr_hi = Float64[]

    # Individual trial raw points (all converged runs across all files)
    static_trials_x = Float64[]
    static_trials_y = Float64[]
    variable_trials_x = Float64[]
    variable_trials_y = Float64[]

    for filepath in files
        fname = lowercase(basename(filepath))
        is_static = occursin("onelevel_", fname)
        is_variable = occursin("fourlevels", fname)

        if !is_static && !is_variable
            continue   # unknown type – skip
        end

        lines = readlines(filepath)
        sigma_levels = Float64[]
        in_results = false
        log_shots_vec = Float64[]
        fidelity_vec = Float64[]

        if is_static
            sigma_single = get_single_level_sigma(filepath)
            if isnan(sigma_single)
                continue
            end
            shots_per_call = round(Int, 1 / sigma_single^2)

            for line in lines
                if contains(line, "=== All Results ===") || contains(line, "INDIVIDUAL RESULTS")
                    in_results = true
                    continue
                end
                if !in_results || strip(line) == "" || contains(line, "Sim\t")
                    continue
                end
                parts = split(line, "\t")
                if length(parts) >= 4
                    try
                        iter = parse(Int, strip(parts[3]))
                        q_det = parse(Float64, strip(parts[4]))
                        if iter != max_iterations
                            total_shots = iter * shots_per_call
                            lx = log10(Float64(total_shots))
                            push!(log_shots_vec, lx)
                            push!(fidelity_vec, q_det)
                            push!(static_trials_x, lx)
                            push!(static_trials_y, 1.0 - q_det)
                        end
                    catch
                        continue
                    end
                end
            end

        else  # variable / multi-level
            for (i, line) in enumerate(lines)
                if contains(line, "INDIVIDUAL RESULTS") || contains(line, "=== All Results ===")
                    in_results = true
                    if i < length(lines)
                        header_line = lines[i+1]
                        sm = collect(eachmatch(r"σ=(\d+\.?\d*)", header_line))
                        sigma_levels = [parse(Float64, m.captures[1]) for m in sm]
                    end
                    continue
                end
                if !in_results || isempty(sigma_levels) || strip(line) == "" || contains(line, "Sim\t")
                    continue
                end
                parts = split(line, "\t")
                if length(parts) < 4 + length(sigma_levels)
                    continue
                end
                try
                    iter = parse(Int, strip(parts[3]))
                    q_det = parse(Float64, strip(parts[4]))
                    if iter == max_iterations
                        continue
                    end
                    total_shots = 0.0
                    for (j, sigma) in enumerate(sigma_levels)
                        calls = parse(Int, strip(parts[4+j]))
                        total_shots += calls * round(Int, 1 / sigma^2)
                    end
                    if total_shots > 0
                        lx = log10(total_shots)
                        push!(log_shots_vec, lx)
                        push!(fidelity_vec, q_det)
                        push!(variable_trials_x, lx)
                        push!(variable_trials_y, 1.0 - q_det)
                    end
                catch
                    continue
                end
            end
        end

        if length(log_shots_vec) < 1
            continue  # no converged runs – skip this file
        end

        mx = mean(log_shots_vec)
        sx = length(log_shots_vec) > 1 ? std(log_shots_vec) : 0.0
        mf = mean(fidelity_vec)          # mean fidelity (used for clamping)
        sy = length(fidelity_vec) > 1 ? std(fidelity_vec) : 0.0
        my = 1.0 - mf                    # mean infidelity (y-value to plot)

        # Clamp error bars: infidelity in (0, 1], log scale requires lower bound > 0
        xerr_lo = min(sx, mx)              # log10(shots) >= 0  →  shots >= 1
        xerr_hi = sx
        yerr_lo = min(sy, my * 0.9999)     # keep lower bar strictly above 0
        yerr_hi = min(sy, max(1.0 - my, 0.0))   # can't exceed infidelity = 1

        if is_static
            push!(static_x, mx)
            push!(static_y, my)
            push!(static_xerr_lo, xerr_lo)
            push!(static_xerr_hi, xerr_hi)
            push!(static_yerr_lo, yerr_lo)
            push!(static_yerr_hi, yerr_hi)
        else
            push!(variable_x, mx)
            push!(variable_y, my)
            push!(variable_xerr_lo, xerr_lo)
            push!(variable_xerr_hi, xerr_hi)
            push!(variable_yerr_lo, yerr_lo)
            push!(variable_yerr_hi, yerr_hi)
        end
    end

    # ---- build plot ---------------------------------------------------------
    p = plot(
        xlabel="log₁₀(Total Shots)",
        ylabel="log₁₀(1 − Fidelity)",
        title="log(Infidelity) vs. Total Shots",
        legend=:topright,
        size=(800, 600),
        dpi=300,
        framestyle=:box,
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3,
        yscale=:log10
    )

    # --- individual trial points (plotted first, behind the means) -----------
    if !isempty(static_trials_x)
        scatter!(p, static_trials_x, static_trials_y,
            color=:dodgerblue,
            alpha=0.25,
            markersize=4,
            markerstrokewidth=0,
            label="Static trials")
    end

    if !isempty(variable_trials_x)
        scatter!(p, variable_trials_x, variable_trials_y,
            color=:crimson,
            alpha=0.25,
            markersize=4,
            markerstrokewidth=0,
            label="Variable trials")
    end

    # --- per-file mean ± std points (plotted on top) -------------------------
    if !isempty(static_x)
        scatter!(p, static_x, static_y,
            xerr=(static_xerr_lo, static_xerr_hi),
            yerr=(static_yerr_lo, static_yerr_hi),
            color=:dodgerblue,
            markersize=8,
            markerstrokecolor=:blue,
            markerstrokewidth=1.5,
            label="Static noise mean ± σ")
    end

    if !isempty(variable_x)
        scatter!(p, variable_x, variable_y,
            xerr=(variable_xerr_lo, variable_xerr_hi),
            yerr=(variable_yerr_lo, variable_yerr_hi),
            color=:crimson,
            markersize=8,
            markerstrokecolor=:darkred,
            markerstrokewidth=1.5,
            label="Variable noise mean ± σ")
    end

    # Set axis limits using all raw trial infidelity points.
    # On a log scale the lower bound must be strictly > 0; use half the minimum.
    # Upper limit: 95th percentile × 3 so extreme high-noise outliers don't
    # dominate while the interesting region stays well visible.
    all_x = vcat(static_trials_x, variable_trials_x)
    all_y = vcat(static_trials_y, variable_trials_y)   # these are infidelities

    if !isempty(all_x)
        x_pad = max((maximum(all_x) - minimum(all_x)) * 0.08, 0.1)
        y_lower = minimum(all_y) / 2.0
        y_upper = quantile(all_y, 0.95) * 3.0
        plot!(p,
            xlim=(max(0.0, minimum(all_x) - x_pad), maximum(all_x) + x_pad),
            ylim=(y_lower, y_upper))
    end

    if output_file !== nothing
        savefig(p, output_file)
        println("Shots-vs-fidelity scatter plot saved to: $output_file")
    else
        display(p)
    end

    return p
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

"""
    main()

Run the benchmark comparison with default files.
Call this function when using include() in the REPL.
"""
function main()
    # Define benchmark files to compare
    script_dir = @__DIR__
    directory = "data/"
    files = [
        joinpath(script_dir, directory, "benchmark_results_fourlevels_oldOmega.txt"),
        joinpath(script_dir, directory, "benchmark_results_onelevel_N10000_old.txt"),
        joinpath(script_dir, directory, "benchmark_results_onelevel_N2500_old.txt"),
        joinpath(script_dir, directory, "benchmark_results_onelevel_N400_old.txt"),
        joinpath(script_dir, directory, "benchmark_results_onelevel_N277.txt"),
        joinpath(script_dir, directory, "benchmark_results_onelevel_N100_old.txt"),
        joinpath(script_dir, directory, "benchmark_results_onelevel_N51_old.txt")
    ]

    labels = [benchmark_label_from_file(f) for f in files]

    # Create comparison plots
    println("Creating benchmark comparison plots...")
    p1, results = plot_benchmark_comparison(files,
        labels=labels,
        output_file=joinpath(script_dir, "..", "figures", "benchmark_comparison.png"))

    println("\nCreating detailed comparison plots...")
    p2, _ = plot_detailed_comparison(files,
        labels=labels,
        output_file=joinpath(script_dir, "..", "figures", "benchmark_detailed_comparison.png"))

    println("\nCreating single-level fidelity vs noise box plot...")
    p3 = plot_single_level_fidelity_vs_noise(files,
        output_file=joinpath(script_dir, "..", "figures", "singlelevel_fidelity_vs_noise_fit.png"))

    println("\nCreating shots-vs-fidelity scatter plot...")
    p4 = plot_shots_vs_fidelity_scatter(files,
        output_file=joinpath(script_dir, "..", "figures", "shots_vs_fidelity_scatter.png"))

    println("\nComparison complete!")
    return p1, p2, p3, p4
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
