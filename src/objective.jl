# src/objective.jl

import NLPModels: obj, grad!

function obj(nlp::LuxNLPModel{T}, x::AbstractVector{T}) where T
    cx = ComponentVector(x, nlp.axes)
    X_batch, y_batch = nlp.current_batch 

    y_pred, _ = nlp.model(X_batch, cx, nlp.st) 
    return nlp.loss_fn(y_pred, y_batch)
end

function grad!(nlp::LuxNLPModel{T}, x::AbstractVector{T}, g::AbstractVector{T}) where T
    cx = ComponentVector(x, nlp.axes)
    X_batch, y_batch = nlp.current_batch

    grads = Zygote.gradient(cx) do p
        y_pred, _ = nlp.model(X_batch, p, nlp.st)
        return nlp.loss_fn(y_pred, y_batch)
    end

    g_val = first(grads)
    isnothing(g_val) ? fill!(g, zero(T)) : g .= getdata(g_val)
    return g
end