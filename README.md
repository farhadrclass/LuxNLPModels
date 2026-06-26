# LuxNLPModels.jl

[![Build Status](https://github.com/Farhad-Phd/LuxNLPModels.jl/workflows/CI/badge.svg)](https://github.com/Farhad-Phd/LuxNLPModels.jl/actions)

An [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) interface for [Lux.jl](https://github.com/LuxDL/Lux.jl) neural networks.

`LuxNLPModels` wraps any Lux model as an `AbstractNLPModel`, letting you train neural networks with gradient-based solvers from [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) (R2, L-BFGS, trunk, …) without changing a single line of solver code.

## Features

- Drop-in `AbstractNLPModel` wrapper around any `Lux.AbstractLuxLayer`
- Provides `obj`, `grad!`, `objgrad!` (single AD pass), and `hprod!`
- GPU-compatible: pass `dev=gpu_device()` and keep solver vectors on CPU
- Mini-batch training via any `DataLoader`-style iterable
- Parameters handled as `ComponentVector` — axes preserved for `Lux.apply`

## Installation

```julia
pkg> add LuxNLPModels
```

## Quick start

```julia
using Lux, ComponentArrays, MLUtils
using NLPModels, JSOSolvers
using LuxNLPModels
using Random, Statistics

rng = Random.MersenneTwister(0)

# 1. Data
x = randn(rng, Float32, 1, 128)
y = evalpoly.(x, Ref((0f0, -2f0, 1f0))) .+ 0.1f0 .* randn(rng, Float32, size(x))
loader = DataLoader((x, y); batchsize=128, shuffle=false)

# 2. Lux model
model = Chain(Dense(1 => 16, relu), Dense(16 => 1))
ps, st = Lux.setup(rng, model)

# 3. Wrap as NLPModel
nlp = LuxNLPModel(model, ComponentArray(ps), st, loader,
                  (ŷ, y) -> mean(abs2, ŷ .- y))

# 4. Solve with any JSOSolvers solver
stats = lbfgs(nlp; max_iter=300, atol=1f-6)
println("Final loss: ", stats.objective)
```

For a GPU run, pass `dev=gpu_device()` and move your data and `st` to the
device with `|> gpu_device()` before constructing the model.  The solver
vectors remain on CPU automatically.

## Mini-batch training

Call `minibatch_next_train!(nlp)` in a solver callback to advance to the next
batch after each iteration:

```julia
cb = (nlp, solver, stats) -> minibatch_next_train!(nlp)
stats = R2(nlp; max_iter=1000, callback=cb)
```

## How to Cite

If you use LuxNLPModels.jl in your work, please cite it using the reference
in [`CITATION.bib`](CITATION.bib).

## Bug Reports and Discussions

Please open an [issue](https://github.com/Farhad-Phd/LuxNLPModels.jl/issues)
for bugs or feature requests.
