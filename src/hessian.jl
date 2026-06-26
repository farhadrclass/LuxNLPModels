import NLPModels: hprod!

function hprod!(nlp::LuxNLPModel{T}, x::AbstractVector{T}, v::AbstractVector{T}, hv::AbstractVector{T}) where T
    # Define exact Dual layout
    TagType = typeof(ForwardDiff.Tag(LuxNLPModelTag(), T))
    DualType = ForwardDiff.Dual{TagType, T, 1}

    # 1. Mutate cached array with new parameters + directional vector
    nlp.x_dual_cache .= DualType.(x, ForwardDiff.Partials{1, T}.(tuple.(v)))
    
    # 2. View generation
    cx_dual = ComponentVector(nlp.x_dual_cache, nlp.axes)
    
    X_batch, y_batch = nlp.current_batch

    # 3. Zygote Backward Pass over ForwardDiff Dual Types
    grads = Zygote.gradient(cx_dual) do p
        y_pred, _ = nlp.model(X_batch, p, nlp.st)
        return nlp.loss_fn(y_pred, y_batch)
    end

    g_dual = first(grads)
    
    # 4. Extract Epsilon Partials
    if isnothing(g_dual)
        fill!(hv, zero(T))
    else
        hv .= ForwardDiff.partials.(getdata(g_dual), 1)
    end
    
    return hv
end