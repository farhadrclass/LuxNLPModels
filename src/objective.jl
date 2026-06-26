# src/objective.jl

import NLPModels: obj, grad!, objgrad!

# ---------------------------------------------------------------------------
# Internal helper — same AD logic as grad! but without touching the counters.
# Used by hprod! to avoid inflating neval_grad.
# ---------------------------------------------------------------------------
function _compute_grad!(
    nlp :: LuxNLPModel{T},
    x   :: AbstractVector{T},
    g   :: AbstractVector{T},
) where T
    # Transfer the flat CPU parameter vector to the target device and give it
    # the ComponentArray layout the model expects.
    cx = ComponentVector(x, nlp.axes) |> nlp.dev
    X_batch, y_batch = nlp.current_batch

    # Zygote differentiates w.r.t. cx.  The result is a ComponentVector that
    # lives on the same device as cx; getdata extracts the backing array.
    g_val = Zygote.gradient(cx) do p
        y_pred, _ = Lux.apply(nlp.model, X_batch, p, nlp.st)
        nlp.loss_fn(y_pred, y_batch)
    end |> first

    if isnothing(g_val)
        fill!(g, zero(T))
    else
        # Move gradient back to CPU (where the solver lives).
        g .= cpu_device()(getdata(g_val))
    end
    return g
end

# ---------------------------------------------------------------------------
# NLPModels public interface
# ---------------------------------------------------------------------------

function obj(nlp::LuxNLPModel{T}, x::AbstractVector{T}) where T
    NLPModels.increment!(nlp, :neval_obj)
    cx = ComponentVector(x, nlp.axes) |> nlp.dev
    X_batch, y_batch = nlp.current_batch
    y_pred, _ = Lux.apply(nlp.model, X_batch, cx, nlp.st)
    return T(nlp.loss_fn(y_pred, y_batch))
end

function grad!(nlp::LuxNLPModel{T}, x::AbstractVector{T}, g::AbstractVector{T}) where T
    NLPModels.increment!(nlp, :neval_grad)
    return _compute_grad!(nlp, x, g)
end

"""
    objgrad!(nlp, x, g)

Compute the objective value and gradient in a single forward/backward pass
(one `Zygote.pullback` call).  This is roughly 2× faster than calling `obj`
and `grad!` separately and is the hot path used by most JSOSolvers routines.
"""
function objgrad!(nlp::LuxNLPModel{T}, x::AbstractVector{T}, g::AbstractVector{T}) where T
    NLPModels.increment!(nlp, :neval_obj)
    NLPModels.increment!(nlp, :neval_grad)
    cx = ComponentVector(x, nlp.axes) |> nlp.dev
    X_batch, y_batch = nlp.current_batch

    # pullback returns (forward_value, backward_closure).
    # The closure accepts a cotangent scalar and returns a tuple of gradients,
    # one per positional argument after the anonymous function.
    val, back = Zygote.pullback(cx) do p
        y_pred, _ = Lux.apply(nlp.model, X_batch, p, nlp.st)
        nlp.loss_fn(y_pred, y_batch)
    end

    g_val = back(one(T))[1]   # gradient w.r.t. cx
    if isnothing(g_val)
        fill!(g, zero(T))
    else
        g .= cpu_device()(getdata(g_val))
    end
    return T(val), g
end