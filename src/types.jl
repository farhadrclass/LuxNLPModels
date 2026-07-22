# src/types.jl

# Type tag kept for potential user-side ForwardDiff extensions.
struct LuxNLPModelTag end

# V is allocated on `dev`.  Solver implementations that support device arrays
# therefore retain their parameters, gradients, and work vectors on the GPU.
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

    g_cache::V        # scratch buffer for gradient, allocated on `dev`
    hv_cache::V       # scratch buffer for Hv finite difference, allocated on `dev`
end

"""
    LuxNLPModel(model, ps, st, data_loader, loss_fn; dev=cpu_device())

Wrap a Lux model as an `NLPModels.AbstractNLPModel`.

# Arguments
- `model`       : a Lux `AbstractLuxLayer`.
- `ps`          : a `ComponentVector{T}` of model parameters.  May already live
                  on the target device; `x0` is copied to `dev`.
- `st`          : Lux model state returned by `Lux.setup`.  Should already be
                  moved to `dev` by the caller.
- `data_loader` : any iterable that yields `(X_batch, y_batch)` pairs.
                  Batches must already live on `dev`.
- `loss_fn`     : a function `(ŷ, y) -> scalar` compatible with Zygote AD.
- `dev`         : Lux device handle (`cpu_device()` or `gpu_device()`).
                  Defaults to `cpu_device()`.

# Notes
- `meta.x0` (and solver-side vectors allocated from it) live on `dev`.  The
    solver must therefore support the chosen array type, such as `CuArray`.
- `obj`/`grad!` expect `x` and `g` to live on `dev`; `objgrad!` computes both
    in one reverse-mode pass.
- Stateful layers (BatchNorm, Dropout) require the caller to manage `nlp.st`
  between epochs; the state returned by `Lux.apply` is intentionally discarded
  inside `obj`/`grad!` to keep the solver-facing interface side-effect-free.
"""
function LuxNLPModel(
    model, ps::ComponentVector{T}, st, data_loader, loss_fn;
    dev = cpu_device(),
) where T
    # Solver state is allocated from x0, so its device determines where the
    # complete optimization state lives.  `copy` prevents aliasing caller ps.
    flat_ps = copy(dev(getdata(ps)))
    axes    = getaxes(ps)
    n       = length(flat_ps)

    meta = NLPModelMeta(n; x0 = flat_ps, name = "LuxNLPModel")

    iter = iterate(data_loader)
    iter === nothing && error("DataLoader is empty")
    current_batch, iter_state = iter

    g_cache  = similar(flat_ps)
    hv_cache = similar(flat_ps)

    return LuxNLPModel(
        meta, Counters(),
        model, st, data_loader,
        iter_state, current_batch, loss_fn, axes, dev,
        g_cache, hv_cache,
    )
end