## vision_benchmark.jl
##
## Compares Adam and TADAM on FashionMNIST image classification.
## This proves Algorithm 5's convergence on deep non-convex landscapes.

using Pkg
Pkg.activate("lux_test_env_2")

using Lux
using LuxNLPModels
using ComponentArrays
using NLPModels, JSOSolvers
using Optimisers
using MLDatasets
using MLUtils
using OneHotArrays
using NNlib: logitcrossentropy
using Random
using Statistics: mean
using Printf
using CairoMakie

# -----------------------------------------------------------------------
# 1. Data Preparation (FashionMNIST)
# -----------------------------------------------------------------------
@info "Loading FashionMNIST Data..."

train_x, train_y = FashionMNIST.traindata(Float32)

# Reshape into (W, H, Channels, Batch) and limit to 10,000 samples
train_x = reshape(train_x, 28, 28, 1, :)[:, :, :, 1:10000]
train_y = onehotbatch(train_y[1:10000], 0:9)

batch_size = 1000
data_loader = DataLoader((train_x, train_y); batchsize=batch_size, shuffle=true)

# -----------------------------------------------------------------------
# 2. Neural Network Model
# -----------------------------------------------------------------------
using NNlib: logsoftmax

rng = MersenneTwister(42)

# Simple Multi-Layer Perceptron (MLP) for Vision
model = Chain(
    FlattenLayer(),
    Dense(28*28 => 128, relu),
    Dense(128 => 64, relu),
    Dense(64 => 10)
)

ps, st = Lux.setup(rng, model)
ps_cv  = ComponentArray(ps)

# This avoids the NNlib import warning and works on all versions.
function loss_fn(ŷ, y)
    return mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))
end

make_nlp() = LuxNLPModel(model, ps_cv, st, data_loader, loss_fn)

@printf "Total Parameters : %d\n" length(ps_cv)
@printf "Initial Loss     : %.6f\n\n" obj(make_nlp(), ps_cv)

# -----------------------------------------------------------------------
# 3. Standard Adam Baseline
# -----------------------------------------------------------------------
function run_adam!(nlp; lr::Float32=0.001f0, max_iter::Int=100)
    x = copy(nlp.meta.x0)
    g = similar(x)
    losses = Float32[]
    
    opt_state = Optimisers.setup(Optimisers.Adam(lr), x)
    
    @info "Starting Adam Training..."
    for i in 1:max_iter
        f, _ = objgrad!(nlp, x, g)
        push!(losses, f)
        
        if i % 10 == 0
            @printf "Adam Iter %3d | Loss: %.4f | GradNorm: %.4f\n" i f norm(g)
        end
        
        opt_state, x = Optimisers.update(opt_state, x, g)
    end
    return x, losses
end

x_adam, adam_loss = run_adam!(make_nlp(); lr=0.001f0, max_iter=1500)

# -----------------------------------------------------------------------
# 4. TADAM Benchmark
# -----------------------------------------------------------------------
tadam_loss  = Float32[]
nlp_tadam   = make_nlp()
_prev_ngrad = Ref(-1)   # sentinel: -1 means "not yet initialised"

function tadam_callback(nlp, solver, stats)
    push!(tadam_loss, stats.objective)
    solver.step_accepted && minibatch_next_train!(nlp)
end


@info "Starting TADAM Training..."
stats_tadam = tadam(
    nlp_tadam;
    max_iter = 1500,
    atol     = 1f-8,
    rtol     = 1f-8,
    # atol     = 1f-5,
    # rtol     = 0f0,
    callback = tadam_callback,
    η2       = 0.75f0,  # Allows trust-region expansion on rough models
    γ3       = 0.9f0,   # Prevents over-penalizing momentum on rejected steps
    β2       = 0.99f0,  # Makes the diagonal Hessian approximation more agile
    ϵ_v      = 1e-6,  # Float32 stability
    verbose  = 10  # Print every 10 iterations
)

# -----------------------------------------------------------------------
# 5. Plot Results
# -----------------------------------------------------------------------
fig = Figure(size=(800, 500))
ax = CairoMakie.Axis(fig[1, 1];
    xlabel = "Iteration", ylabel = "Cross Entropy Loss",
    title  = "Adam vs TADAM (FashionMNIST)", yscale = log10)

lines!(ax, adam_loss;  label="Adam (lr=1e-3)", color=:gray50,     linewidth=2.5)
lines!(ax, tadam_loss; label="TADAM",          color=:dodgerblue, linewidth=2.5)

axislegend(ax; position=:rt)
display(fig)