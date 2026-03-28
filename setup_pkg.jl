# Automated Julia package setup for QuantumFidelityOptimization
# Called by setup.sh — do not run directly unless Julia 1.11.6 is active and cwd is repo root.

using Pkg

project_dir = @__DIR__
ionsim_dev  = joinpath(homedir(), ".julia", "dev", "IonSim")
manifest    = joinpath(project_dir, "Manifest.toml")
manifest_bak = joinpath(project_dir, "Manifest.toml.bak")

# ── helpers ───────────────────────────────────────────────────────────────────
function step(n, msg)
    println("\n[$(n)] $(msg)")
    flush(stdout)
end

# ── 1. Activate ───────────────────────────────────────────────────────────────
step("1/6", "Activating project at $project_dir")
Pkg.activate(project_dir)

# ── 2. Backup the author's Manifest before Pkg.develop modifies it ────────────
step("2/6", "Backing up Manifest.toml → Manifest.toml.bak")
cp(manifest, manifest_bak; force=true)

# ── 3. Develop IonSim (clones to ~/.julia/dev/IonSim, modifies Manifest) ──────
step("3/6", "Developing IonSim (downloads to $ionsim_dev)...")
Pkg.develop("IonSim")

# ── 4. Pin to v0.5.1 and apply patch ─────────────────────────────────────────
step("4/6", "Checking out IonSim v0.5.1 and applying patch...")

run(`git -C $ionsim_dev checkout v0.5.1`)

iontraps = joinpath(ionsim_dev, "src", "iontraps.jl")
isfile(iontraps) || error("iontraps.jl not found at: $iontraps")

content = read(iontraps, String)
replacement = "Optim.Options(g_tol=1e-6, x_abstol=1e-12, x_reltol=1e-6,\n              f_abstol=1e-12, f_reltol=1e-6)"
patched = replace(content, r"Optim\.Options\(.*?\)"s => replacement)

if patched == content
    @warn "Optim.Options pattern not found — iontraps.jl was NOT patched. Check manually."
else
    write(iontraps, patched)
    println("  Patch applied to iontraps.jl.")
end

# ── 5. Restore author's Manifest, then re-develop so the dev path is injected ─
# This is the critical step: using the locked Manifest ensures exact package
# versions, while re-running Pkg.develop updates only the IonSim entry to the
# local dev path.
step("5/6", "Restoring original Manifest and re-instantiating with locked versions...")
cp(manifest_bak, manifest; force=true)
Pkg.develop("IonSim")   # re-adds the local dev path to the restored Manifest
Pkg.instantiate()

# ── 6. Build IonSim ──────────────────────────────────────────────────────────
step("6/6", "Building IonSim...")
Pkg.build("IonSim")

# ── Verify patch ──────────────────────────────────────────────────────────────
println("\n--- Verification ---")
final = read(iontraps, String)
if occursin("g_tol=1e-6", final)
    println("PASS  Optim.Options patch is active in IonSim.")
else
    println("FAIL  Patch not detected — check iontraps.jl manually.")
end

println("""
Setup complete.

To run the project:
  julia --project=. src/main.jl

To undo the IonSim patch later:
  git -C ~/.julia/dev/IonSim checkout -- src/iontraps.jl
""")
