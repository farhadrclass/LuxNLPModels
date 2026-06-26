# src/types.jl

struct LuxNLPModelTag end

mutable struct LuxNLPModel{T, V <: AbstractVector{T}, M, S, D, I, B, L, AX, VD <: AbstractVector} <: AbstractNLPModel{T, V}
    meta::NLPModelMeta{T, V}
    counters::Counters
    
    model::M             
    st::S                
    data_loader::D       
    iter_state::I        
    current_batch::B     
    loss_fn::L           
    axes::AX             # Caches structural layout metadata
    
    g_cache::V           
    hv_cache::V          
    x_dual_cache::VD     
end

function LuxNLPModel(model, ps::ComponentVector{T}, st, data_loader, loss_fn) where T
    # Extract flat raw data buffer and structural axes metadata
    flat_ps = getdata(ps)
    axes = getaxes(ps)
    n = length(flat_ps)
    
    # NLPModelMeta now receives a standard flat array. Eliminates constructor conflict.
    meta = NLPModelMeta(n, x0 = flat_ps, name = "LuxDeepLearningModel")
    
    iter = iterate(data_loader)
    iter === nothing && error("DataLoader is empty")
    current_batch, iter_state = iter

    g_cache = similar(flat_ps)
    hv_cache = similar(flat_ps)
    
    TagType = typeof(ForwardDiff.Tag(LuxNLPModelTag(), T))
    DualType = ForwardDiff.Dual{TagType, T, 1}
    x_dual_cache = similar(flat_ps, DualType)
    
    return LuxNLPModel(
        meta, Counters(), model, st, data_loader, 
        iter_state, current_batch, loss_fn, axes,
        g_cache, hv_cache, x_dual_cache
    )
end