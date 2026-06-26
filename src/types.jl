# src/types.jl

# Type tag kept for potential user-side ForwardDiff extensions.
struct LuxNLPModelTag end

# V is always a plain CPU vector (Vector{T}).  JSOSolvers allocates all its
# internal state from meta.x0, so keeping V on CPU is required for solver
# compatibility regardless of where the model and data live.
mutable struct LuxNLPModel{T, V <: AbstractVector{T}, M, S, D, I, B, L, AX} <: AbstractNLPModel{T, V}
    meta::NLPModelMeta{T, V}
    counters::Counters

    model::M          # Lux AbstractLuxLayer (device-agnostic)
    st::S             # Lux model state — lives on `dev`
    data_loader::D    # MLUtils.DataLoader or similar
    iter_state::I     # internal DataLoader iterator state
    current_batch::B  # (X, y) tuple — lives on `dev`
    loss_fn::L        # (ŷ, y) -> scalar
    axes::AX          # ComponentArray structural axes (CPU metadata, no data)
    dev::Any          # Lux device: cpu_device() or gpu_device()

    g_cache::V        # scratch buffer for gradient (always CPU)
    hv_cache::V       # scratch buffer for Hv finite difference (always CPU)
end

"""
    LuxNLPModel(model, ps, st, data_loader, loss_fn; dev=cpu_device())

Wrap a Lux model as an `NLPModels.AbstractNLPModel`.

# Arguments
- `model`       : a Lux `AbstractLuxLayer`.
- `ps`          : a `ComponentVector{T}` of model parameters.  May already live
                  on the target device; `x0` will always be copied to CPU.
- `st`          : Lux model state returned by `Lux.setup`.  Should already be
                  moved to `dev` by the caller.
- `data_loader` : any iterable that yields `(X_batch, y_batch)` pairs.
                  Batches must already live on `dev`.
- `loss_fn`     : a function `(ŷ, y) -> scalar` compatible with Zygote AD.
- `dev`         : Lux device handle (`cpu_device()` or `gpu_device()`).
                  Defaults to `cpu_device()`.

# Notes
- `meta.x0` (and all solver-side vectors) are always `Vector{T}` on CPU.
- Inside `obj`/`grad!` the flat parameter vector `x` is transferred to `dev`
  on every call.  Use `objgrad!` to halve the number of AD passes.
- Stateful layers (BatchNorm, Dropout) require the caller to manage `nlp.st`
  between epochs; the state returned by `Lux.apply` is intentionally discarded
  inside `obj`/`grad!` to keep the solver-facing interface side-effect-free.
"""
function LuxNLPModel(
    model, ps::ComponentVector{T}, st, data_loader, loss_fn;
    dev = cpu_device(),
) where T
    # Always copy parameter data to a plain CPU Vector so that NLPModelMeta
    # (and every JSOSolvers internal buffer) stays on CPU.
    flat_ps_cpu = Array(getdata(ps))   # Array() is a no-op for CPU, copies for GPU
    axes = getaxes(ps)
    n    = length(flat_ps_cpu)

    meta = NLPModelMeta(n; x0 = flat_ps_cpu, name = "LuxNLPModel")

    iter = iterate(data_loader)
    iter === nothing && error("DataLoader is empty")
    current_batch, iter_state = iter

    g_cache  = similar(flat_ps_cpu)
    hv_cache = similar(flat_ps_cpu)

    return LuxNLPModel(
        meta, Counters(),
        model, st, data_loader,
        iter_state, current_batch, loss_fn, axes, dev,
        g_cache, hv_cache,
    )
end