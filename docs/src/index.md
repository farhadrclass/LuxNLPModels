# LuxNLPModels.jl

## Compatibility
Julia ≥ 1.9.

## How to install

```julia
pkg> add LuxNLPModels
```

## Synopsis

`LuxNLPModels` exposes [Lux.jl](https://github.com/LuxDL/Lux.jl) neural
networks as nonlinear optimization problems conforming to the
[NLPModels API](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
This makes it possible to train neural networks with any solver from
[JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl)
(R2, L-BFGS, trunk, …) without modification.

A `LuxNLPModel` exposes:
- `obj(nlp, x)` — loss evaluated at the flat parameter vector `x`
- `grad!(nlp, x, g)` — in-place gradient via Zygote
- `objgrad!(nlp, x, g)` — combined forward+backward pass (half the AD cost)
- `hprod!(nlp, x, v, hv)` — Hessian-vector product (finite-difference, GPU-safe)
- `minibatch_next_train!(nlp)` — advance to the next mini-batch

Parameters are stored as a flat `Vector{T}` on CPU (JSOSolvers
compatibility).  GPU execution is supported: pass `dev=gpu_device()` and
move your data and Lux state to the device — the CPU/GPU transfer is
handled transparently inside each function call.

## Bug reports

Please open an [issue](https://github.com/Farhad-Phd/LuxNLPModels.jl/issues).

