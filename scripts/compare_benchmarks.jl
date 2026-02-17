# scripts/compare_benchmarks.jl
# Compare multiple benchmark result files

using Plots
using Statistics

"""
    parse_benchmark_file(filepath)

Parse a benchmark results file and extract key statistics.
Returns a dictionary with:
- name: filename
- avg_qdet: average Q_det value
- std_qdet: standard deviation of Q_det
- avg_iterations: average iterations
- std_iterations: standard deviation of iterations
- n_sims: number of simulations
- min_qdet, max_qdet, median_qdet
"""
function parse_benchmark_file(filepath)
    lines = readlines(filepath)
    
    result = Dict{String, Any}(
        "name" => basename(filepath),
        "avg_qdet" => NaN,
        "std_qdet" => NaN,
        "avg_iterations" => NaN,
        "std_iterations" => NaN,
        "n_sims" => 0,
        "min_qdet" => NaN,
        "max_qdet" => NaN,
        "median_qdet" => NaN
    )
    
    for line in lines
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
                                  labels::Union{Vector{String}, Nothing}=nothing,
                                  output_file::Union{String, Nothing}=nothing)
    
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
    
    # Plot Q_det with error bars
    avg_qdet = [r["avg_qdet"] for r in results]
    std_qdet = [r["std_qdet"] for r in results]
    
    bar!(p1, x_positions, avg_qdet, 
         yerr=std_qdet,
         fillcolor=:lightblue,
         linecolor=:blue,
         xticks=(x_positions, labels),
         xrotation=45,
         ylabel="Q_det",
         ylim=(0, 1.1))
    
    # Add data labels on top of bars
    for (i, (avg, std)) in enumerate(zip(avg_qdet, std_qdet))
        annotate!(p1, i, avg + std + 0.05, 
                 text(string(round(avg, digits=3)), 8, :center))
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
    
    # Combine plots
    p = plot(p1, p2, layout=(2, 1), size=(900, 800), 
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
                                 labels::Union{Vector{String}, Nothing}=nothing,
                                 output_file::Union{String, Nothing}=nothing)
    
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
    
    # Efficiency metric (Q_det / iterations)
    p4 = plot(title="Efficiency (Q_det / Iteration)", legend=false, size=(600, 400))
    efficiency = [r["avg_qdet"] / r["avg_iterations"] for r in results]
    bar!(p4, x_positions, efficiency,
         fillcolor=:purple, linecolor=:darkviolet,
         xticks=(x_positions, labels), xrotation=45,
         ylabel="Q_det per Iteration")
    
    # Combine plots
    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900),
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Define benchmark files to compare
    script_dir = @__DIR__
    files = [
        joinpath(script_dir, "benchmark_results_onelevel.txt"),
        joinpath(script_dir, "benchmark_results.txt"),
        joinpath(script_dir, "mf_benchmark_results.txt")
    ]
    
    # Custom labels for clarity
    labels = [
        "One-Level",
        "Standard",
        "Multi-Fidelity"
    ]
    
    # Create comparison plots
    println("Creating benchmark comparison plots...")
    p1, results = plot_benchmark_comparison(files, 
                                           labels=labels,
                                           output_file=joinpath(script_dir, "..", "figures", "benchmark_comparison.png"))
    
    println("\nCreating detailed comparison plots...")
    p2, _ = plot_detailed_comparison(files,
                                    labels=labels, 
                                    output_file=joinpath(script_dir, "..", "figures", "benchmark_detailed_comparison.png"))
    
    println("\nComparison complete!")
end
