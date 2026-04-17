# src/ms_sequence_search.jl
#
# Search utilities for MS-only calibration probes built from closed-loop
# `MS_φ(θ)` subgates. The objective is shot-noise aware and favors sequences
# whose nominal readout stays non-saturated enough to remain practical at low
# shot count, without strictly enforcing a 0.5/0.5 operating point.

const MS_SEARCH_TWO_PULSE_ANGLE_GRID = Float64[π / 4, 3π / 8, π / 2, 5π / 8]
const MS_SEARCH_TWO_PULSE_PHASE_GRID = Float64[0.0, π / 8, π / 4, 3π / 8, π / 2]
const MS_SEARCH_THREE_PULSE_OUTER_GRID = Float64[π / 4, 3π / 8, π / 2]
const MS_SEARCH_THREE_PULSE_MIDDLE_GRID = Float64[π / 4, 3π / 8, π / 2, 5π / 8]
const MS_SEARCH_THREE_PULSE_PHASE_GRID = Float64[0.0, π / 8, π / 4, 3π / 8, π / 2]
const MS_SEARCH_THREE_PULSE_EPS_GRID = Float64[0.0, π / 16, π / 8]
const MS_SEARCH_MIN_EVEN_PREFILTER = 0.0
const MS_SEARCH_MIN_EVEN_READOUT = 0.12

const MS_SEARCH_PARAMETER_SCALES = (
    omega_ratio=1.0,
    centerline=1.0e3,
    sideband=1.0e3,
    phase_error=π,
)

const MS_SEARCH_PARAMETER_STEPS = (
    omega_ratio=0.01,
    centerline=250.0,
    sideband=250.0,
    phase_error=π / 24,
)

function closed_loop_ms_candidate_sequences()
    candidates = NamedTuple[]

    for θ1 in MS_SEARCH_TWO_PULSE_ANGLE_GRID,
        θ2 in MS_SEARCH_TWO_PULSE_ANGLE_GRID,
        φ2 in MS_SEARCH_TWO_PULSE_PHASE_GRID
        push!(candidates, (
            family=:two_pulse,
            subgates=[ms_subgate(θ1, 0.0), ms_subgate(θ2, φ2)],
        ))
    end

    for θ_outer in MS_SEARCH_THREE_PULSE_OUTER_GRID,
        θ_mid in MS_SEARCH_THREE_PULSE_MIDDLE_GRID,
        φ_mid in MS_SEARCH_THREE_PULSE_PHASE_GRID,
        ϵ in MS_SEARCH_THREE_PULSE_EPS_GRID
        push!(candidates, (
            family=:three_pulse,
            subgates=[
                ms_subgate(θ_outer, 0.0),
                ms_subgate(θ_mid, φ_mid),
                ms_subgate(θ_outer, -ϵ),
            ],
        ))
    end

    return candidates
end

function sequence_nominal_summary(pops)
    probs = trinary_ms_probabilities(pops)
    return (
        gg=probs.gg,
        ee=probs.ee,
        odd=probs.odd,
        balance=probs.gg - probs.ee,
    )
end

function ideal_even_subspace_probabilities(subgates::AbstractVector{<:MSSubgate})
    state = ComplexF64[1.0, 0.0]
    for subgate in subgates
        s, c = sincos(subgate.theta / 2.0)
        phase = cis(2.0 * subgate.phi)
        U = ComplexF64[
            c -1im * conj(phase) * s;
            -1im * phase * s c
        ]
        state = U * state
    end
    state ./= norm(state)
    return (gg=abs2(state[1]), ee=abs2(state[2]), odd=0.0)
end

function passes_nominal_prefilter(subgates::AbstractVector{<:MSSubgate})
    probs = ideal_even_subspace_probabilities(subgates)
    return min(probs.gg, probs.ee) >= MS_SEARCH_MIN_EVEN_PREFILTER
end

function candidate_measurement(t, f_cl, Δ, I_pi2, subgates;
                               omega_ratio::Float64=1.0,
                               centerline::Float64=f_cl,
                               sideband::Float64=Δ,
                               phase_error::Float64=0.0)
    pulses = build_closed_loop_ms_sequence(
        t, centerline, sideband, I_pi2, subgates;
        phase_error=phase_error, omega_ratio=omega_ratio)
    pops = populations_ms_sequence(pulses)
    probs = sequence_nominal_summary(pops)
    obs = ms_balance_and_odd(pops)
    return (pops=pops, probs=probs, obs=obs)
end

function max_pairwise_column_correlation(M::AbstractMatrix{<:Real})
    ncols = size(M, 2)
    max_corr = 0.0
    for i in 1:(ncols - 1)
        vi = @view M[:, i]
        ni = norm(vi)
        for j in (i + 1):ncols
            vj = @view M[:, j]
            nj = norm(vj)
            corr = (ni == 0.0 || nj == 0.0) ? 1.0 : abs(dot(vi, vj) / (ni * nj))
            max_corr = max(max_corr, corr)
        end
    end
    return max_corr
end

function candidate_jacobian(t, f_cl, Δ, I_pi2, subgates)
    base = candidate_measurement(t, f_cl, Δ, I_pi2, subgates)
    J = Matrix{Float64}(undef, 2, 4)

    perturb_specs = (
        (name=:omega_ratio, actual=MS_SEARCH_PARAMETER_STEPS.omega_ratio,
         normalized=MS_SEARCH_PARAMETER_STEPS.omega_ratio / MS_SEARCH_PARAMETER_SCALES.omega_ratio),
        (name=:centerline, actual=MS_SEARCH_PARAMETER_STEPS.centerline,
         normalized=MS_SEARCH_PARAMETER_STEPS.centerline / MS_SEARCH_PARAMETER_SCALES.centerline),
        (name=:sideband, actual=MS_SEARCH_PARAMETER_STEPS.sideband,
         normalized=MS_SEARCH_PARAMETER_STEPS.sideband / MS_SEARCH_PARAMETER_SCALES.sideband),
        (name=:phase_error, actual=MS_SEARCH_PARAMETER_STEPS.phase_error,
         normalized=MS_SEARCH_PARAMETER_STEPS.phase_error / MS_SEARCH_PARAMETER_SCALES.phase_error),
    )

    for (col_idx, spec) in enumerate(perturb_specs)
        plus = if spec.name === :omega_ratio
            candidate_measurement(t, f_cl, Δ, I_pi2, subgates; omega_ratio=1.0 + spec.actual)
        elseif spec.name === :centerline
            candidate_measurement(t, f_cl, Δ, I_pi2, subgates; centerline=f_cl + spec.actual)
        elseif spec.name === :sideband
            candidate_measurement(t, f_cl, Δ, I_pi2, subgates; sideband=Δ + spec.actual)
        else
            candidate_measurement(t, f_cl, Δ, I_pi2, subgates; phase_error=spec.actual)
        end

        minus = if spec.name === :omega_ratio
            candidate_measurement(t, f_cl, Δ, I_pi2, subgates; omega_ratio=1.0 - spec.actual)
        elseif spec.name === :centerline
            candidate_measurement(t, f_cl, Δ, I_pi2, subgates; centerline=f_cl - spec.actual)
        elseif spec.name === :sideband
            candidate_measurement(t, f_cl, Δ, I_pi2, subgates; sideband=Δ - spec.actual)
        else
            candidate_measurement(t, f_cl, Δ, I_pi2, subgates; phase_error=-spec.actual)
        end

        J[:, col_idx] = (plus.obs - minus.obs) ./ (2.0 * spec.normalized)
    end

    return (base=base, J=J)
end

function score_ms_candidate(t, f_cl, Δ, I_pi2, candidate)
    jac = candidate_jacobian(t, f_cl, Δ, I_pi2, candidate.subgates)
    probs = jac.base.probs
    even_floor = min(probs.gg, probs.ee)

    if probs.odd > 0.02
        return merge(candidate, (
            nominal=probs,
            score=-Inf,
            info_area=0.0,
            balance_score=0.0,
            readout_weight=0.0,
            max_corr=1.0,
            singular_values=Float64[],
            fisher_eigs=Float64[],
            column_norms=(omega_ratio=0.0, centerline=0.0, sideband=0.0, phase_error=0.0),
        ))
    end

    cov_pops = (gg=probs.gg, ee=probs.ee, eg=probs.odd, ge=0.0)
    Σ = ms_observable_covariance(cov_pops)
    J = jac.J
    W = cholesky(Hermitian(Σ + 1e-9I)).L
    Jw = W \ J

    singular_values = svdvals(Jw)
    info_area = prod(singular_values)
    column_norms = vec(norm.(eachcol(Jw)))
    max_corr = max_pairwise_column_correlation(Jw)
    balance_score = isempty(column_norms) ? 0.0 : minimum(column_norms) / max(maximum(column_norms), eps())
    readout_weight = 1.0
    corr_weight = max(1.0 - max_corr^2, 0.0)
    complexity_penalty = sqrt(length(candidate.subgates))
    total_score = info_area * sqrt(max(balance_score, 0.0)) * corr_weight * readout_weight / complexity_penalty

    F = Symmetric(transpose(Jw) * Jw)
    return merge(candidate, (
        nominal=probs,
        score=isfinite(total_score) ? total_score : -Inf,
        info_area=info_area,
        balance_score=balance_score,
        readout_weight=readout_weight,
        max_corr=max_corr,
        singular_values=singular_values,
        fisher_eigs=sort(collect(eigvals(F)); rev=true),
        column_norms=(
            omega_ratio=column_norms[1],
            centerline=column_norms[2],
            sideband=column_norms[3],
            phase_error=column_norms[4],
        ),
    ))
end

function baseline_sequence_results(t, f_cl, Δ, I_pi2)
    return Dict(
        spec.name => score_ms_candidate(
            t, f_cl, Δ, I_pi2, (family=:baseline, subgates=spec.subgates))
        for spec in default_ms_sequence_specs()
    )
end

function best_distinct_candidate(results, baselines)
    for result in results
        if all(!same_ms_subgates(result.subgates, baselines[name].subgates) for name in keys(baselines))
            return result
        end
    end
    return nothing
end

function search_ms_calibration_sequence(t::Float64;
                                        omega_ratio_grid::AbstractVector{<:Real}=collect(range(0.995, 1.03; length=71)))
    init = ideal(t)
    omega_center = refine_bell_ms_sequence_omega_ratio(
        t, init.f_cl, init.f_sb, init.A, sequence_A_subgates();
        ratio_grid=omega_ratio_grid)
    I_center = init.A * omega_center.ratio^2

    candidates = closed_loop_ms_candidate_sequences()
    filtered_candidates = [candidate for candidate in candidates if passes_nominal_prefilter(candidate.subgates)]
    isempty(filtered_candidates) && error("No MS candidates passed the nominal balance prefilter.")
    @info "Scoring closed-loop MS candidates" total=length(candidates) filtered=length(filtered_candidates)
    results = NamedTuple[]
    for (idx, candidate) in enumerate(filtered_candidates)
        push!(results, score_ms_candidate(t, init.f_cl, init.f_sb, I_center, candidate))
        if idx == 1 || idx % 10 == 0 || idx == length(filtered_candidates)
            @info "Scored MS candidates" completed=idx total=length(filtered_candidates)
        end
    end
    sort!(results; by=result -> result.score, rev=true)

    baselines = baseline_sequence_results(t, init.f_cl, init.f_sb, I_center)
    best_overall = first(results)
    baseline_specs = default_ms_sequence_specs()
    matched_idx = findfirst(spec -> same_ms_subgates(best_overall.subgates, baselines[spec.name].subgates),
                            baseline_specs)
    best_new = best_distinct_candidate(results, baselines)

    return (
        init=init,
        omega_center=omega_center,
        I_center=I_center,
        baselines=baselines,
        candidate_count=length(candidates),
        filtered_candidate_count=length(filtered_candidates),
        top_results=results[1:min(length(results), 10)],
        best_overall=best_overall,
        best_new=best_new,
        distinct=isnothing(matched_idx),
        matching_baseline=isnothing(matched_idx) ? nothing : baseline_specs[matched_idx].name,
    )
end
