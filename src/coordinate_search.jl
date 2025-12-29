# src/coordinate_search.jl

"""
    coordinate_search(qfun, init;
                      tolerance=5e-3, offset=2e3, step=1e2, max_iter=50, verbose=false)

Coordinate-wise search over a 3-parameter tuple using a black-box objective
`qfun(p1, p2, p3) -> Real` that is maximized.

`init` must be a 3-tuple: (p1, p2, p3).

Returns a named tuple with the best value and parameters. Designed to be unit-tested
without IonSim / physics dependencies.
"""
function coordinate_search(qfun, init;
                           tolerance=5e-3, offset=2e3, step=1e2, max_iter=50,
                           verbose::Bool=false)

    max_iter ≥ 1 || throw(ArgumentError("max_iter must be ≥ 1"))
    step > 0     || throw(ArgumentError("step must be > 0"))
    offset ≥ 0   || throw(ArgumentError("offset must be ≥ 0"))
    tolerance ≥ 0 || throw(ArgumentError("tolerance must be ≥ 0"))

    best_params = init
    best_val = qfun(best_params...)
    iter = 0
    diff = Inf

    while diff > tolerance && iter < max_iter
        iter += 1
        prev = best_val

        p1, p2, p3 = best_params

        for v1 in (p1 - offset):step:(p1 + offset)
            v = qfun(v1, p2, p3)
            if v > best_val
                best_val = v
                best_params = (v1, p2, p3)
            end
        end

        p1, p2, p3 = best_params
        for v2 in (p2 - offset):step:(p2 + offset)
            v = qfun(p1, v2, p3)
            if v > best_val
                best_val = v
                best_params = (p1, v2, p3)
            end
        end

        p1, p2, p3 = best_params
        for v3 in (p3 - offset):step:(p3 + offset)
            v = qfun(p1, p2, v3)
            if v > best_val
                best_val = v
                best_params = (p1, p2, v3)
            end
        end

        diff = abs(best_val - prev)
        if verbose
            @info "iter=$iter best=$best_val params=$best_params diff=$diff"
        end
    end

    return (val=best_val, p1=best_params[1], p2=best_params[2], p3=best_params[3], iters=iter)
end

"""
    trial(t, f_cl, f_sb, A; method=:det, ...)

Backwards-compatible wrapper around `coordinate_search` using `Q_det` or `Q_noisy`.
"""
function trial(t, f_cl, f_sb, A;
               method::Symbol=:det,
               tolerance=5e-3, offset=2e3, step=1e2, max_iter=50,
               verbose::Bool=false)

    qfun = method === :noisy ?
        ((fcl, fsb, Aval) -> Q_noisy(t, fcl, fsb, Aval; N=100)) :
        ((fcl, fsb, Aval) -> Q_det(t, fcl, fsb, Aval))

    res = coordinate_search(qfun, (f_cl, f_sb, A);
                            tolerance=tolerance, offset=offset, step=step,
                            max_iter=max_iter, verbose=verbose)

    return (fid=res.val, t=t, f_cl=res.p1, f_sb=res.p2, A=res.p3, iters=res.iters)
end
