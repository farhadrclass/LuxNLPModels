import NLPModels: hprod!

"""
    hprod!(nlp, x, v, hv; obj_weight=1)

Hessian-vector product via a forward finite-difference of the gradient:

    ∇²f(x) v  ≈  (∇f(x + ε v) − ∇f(x)) / ε,   ε = √eps(T)

This approach is GPU-compatible because it never stores `ForwardDiff.Dual`
values in device arrays (which is unsupported by CUDA / AMDGPU).
It calls the internal `_compute_grad!` helper twice to avoid inflating the
`neval_grad` counter.
"""
function hprod!(
    nlp        :: LuxNLPModel{T},
    x          :: AbstractVector{T},
    v          :: AbstractVector{T},
    hv         :: AbstractVector{T};
    obj_weight :: T = one(T),
) where T
    NLPModels.increment!(nlp, :neval_hprod)

    ε   = sqrt(eps(T))
    xpv = x .+ ε .* v

    # ∇f(x + ε v) → written into hv
    _compute_grad!(nlp, xpv, hv)
    # ∇f(x)       → written into hv_cache
    _compute_grad!(nlp, x, nlp.hv_cache)

    # Forward finite difference
    @. hv = (hv - nlp.hv_cache) / ε

    obj_weight != one(T) && (hv .*= obj_weight)
    return hv
end