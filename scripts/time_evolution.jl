import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

include(joinpath(@__DIR__, "..", "src", "CalibrationCode.jl"))
using .CalibrationCode
using Printf

# If IonSim/QuantumOptics functions aren't exported by CalibrationCode, 
# you may need to explicitly load them here:
using IonSim, QuantumOptics

# ==========================================
# 1. Setup Parameters
# ==========================================
const t_gate = 100.0
const steps = 250
const tout = collect(range(0.0, t_gate, length=steps))

println("Calculating ideal baseline parameters...")
const base = CalibrationCode.ideal(t_gate)
const f_cl_ideal = base.f_cl
const f_sb_ideal = base.f_sb
const A_ideal = base.A

# Create a bad calibration by shifting the sideband detuning by 1.5 kHz
const f_sb_bad = f_sb_ideal + (1.5 * 1e3 * 2π)

# ==========================================
# 2. Dynamics Extraction Function
# ==========================================
function get_population_dynamics(t_gate, f_cl, f_sb, A, tout)
    setup = CalibrationCode.build_chamber()
    CalibrationCode.configure_lasers!(setup, f_cl, f_sb, A)

    ca, chamber, mode = setup.ca, setup.chamber, setup.mode
    
    # Hamiltonian setup
    h = CalibrationCode.hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
    
    # Initial state: |S, S, n=0>
    psi0 = ca["S"] ⊗ ca["S"] ⊗ mode[0]
    
    # Evolve over the full tout array
    _, sol = CalibrationCode.timeevolution.schroedinger_dynamic(tout, psi0, h)

    # Projectors
    proj_SS = CalibrationCode.ionprojector(chamber, "S", "S")
    proj_DD = CalibrationCode.ionprojector(chamber, "D", "D")
    proj_SD = CalibrationCode.ionprojector(chamber, "S", "D")
    proj_DS = CalibrationCode.ionprojector(chamber, "D", "S")

    # expect() automatically broadcasts over the array of states in 'sol'
    SS = real.(CalibrationCode.expect(proj_SS, sol))
    DD = real.(CalibrationCode.expect(proj_DD, sol))
    SD = real.(CalibrationCode.expect(proj_SD, sol))
    DS = real.(CalibrationCode.expect(proj_DS, sol))

    return SS, DD, SD, DS
end

# ==========================================
# 3. Run and Save
# ==========================================
println("Simulating High Fidelity dynamics...")
SS_id, DD_id, SD_id, DS_id = get_population_dynamics(t_gate, f_cl_ideal, f_sb_ideal, A_ideal, tout)

println("Simulating Low Fidelity dynamics...")
SS_bad, DD_bad, SD_bad, DS_bad = get_population_dynamics(t_gate, f_cl_ideal, f_sb_bad, A_ideal, tout)

output_file = joinpath(@__DIR__, "..", "data", "population_dynamics.txt")
println("Writing results to $output_file ...")

open(output_file, "w") do io
    # Header
    println(io, "time,SS_ideal,DD_ideal,SD_ideal,DS_ideal,SS_bad,DD_bad,SD_bad,DS_bad")
    
    # Rows
    for i in 1:steps
        @printf(io, "%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            tout[i],
            SS_id[i], DD_id[i], SD_id[i], DS_id[i],
            SS_bad[i], DD_bad[i], SD_bad[i], DS_bad[i])
    end
end

println("Done!")