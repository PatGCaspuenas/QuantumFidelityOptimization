using Statistics
using Plots

results_file = joinpath(@__DIR__, "benchmark_results.txt")
output_file = joinpath(@__DIR__, "benchmark_histogram.png")

if !isfile(results_file)
	error("Results file not found: $results_file")
end

lines = readlines(results_file)

start_idx = findfirst(==("=== All Results ==="), lines)
if start_idx === nothing
	error("Could not find results table in $results_file")
end

data_lines = lines[(start_idx + 2):end]  # skip header line after table title
data_lines = filter(l -> !isempty(strip(l)), data_lines)

Q_det_values = Float64[]
for line in data_lines
	parts = split(strip(line), '\t')
	if length(parts) < 3
		continue
	end
	push!(Q_det_values, parse(Float64, parts[3]))
end

if isempty(Q_det_values)
	error("No Q_det values found in $results_file")
end

println("Loaded $(length(Q_det_values)) results")
println("Q_det mean = ", mean(Q_det_values))
println("Q_det median = ", median(Q_det_values))

histogram(
	Q_det_values;
	bins=:auto,
	xlabel="Q_det",
	ylabel="Count",
	title="Benchmark Q_det Histogram",
	legend=false
)

savefig(output_file)
println("Histogram saved to: $output_file")
