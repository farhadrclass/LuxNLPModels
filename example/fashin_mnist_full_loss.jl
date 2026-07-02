using Pkg
Pkg.activate("lux_test_env_2")

using Lux
using LuxNLPModels
using LinearAlgebra: norm
using ComponentArrays
using NLPModels, JSOSolvers
using Optimisers
using MLDatasets
using MLUtils
using OneHotArrays
using NNlib: logsoftmax
using Random
using Statistics: mean
using Printf
using CairoMakie

# -----------------------------------------------------------------------
# 1. Data Preparation
# -----------------------------------------------------------------------
train_x, train_y = FashionMNIST.traindata(Float32)

train_x = reshape(train_x, 28, 28, 1, :)[:, :, :, 1:10000]
train_y = onehotbatch(train_y[1:10000], 0:9)

batch_size = 1000
data_loader = DataLoader((train_x, train_y); batchsize=batch_size, shuffle=true)

# -----------------------------------------------------------------------
# 2. Model Setup
# -----------------------------------------------------------------------
rng = MersenneTwister(42)

model = Chain(
    FlattenLayer(),
    Dense(28*28 => 128, relu),
    Dense(128 => 64, relu),
    Dense(64 => 10)
)

ps, st = Lux.setup(rng, model)
ps_cv  = ComponentArray(ps)

function loss_fn(ŷ, y)
    return mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))
end

make_nlp() = LuxNLPModel(model, ps_cv, st, data_loader, loss_fn)

# Full dataset evaluator. Wraps data in a single-element array.
full_data = [(train_x, train_y)]
nlp_eval = LuxNLPModel(model, ps_cv, st, full_data, loss_fn)

@printf "Total Parameters : %d\n" length(ps_cv)
@printf "Initial Loss     : %.6f\n\n" obj(nlp_eval, ps_cv)

# -----------------------------------------------------------------------
# 3. Adam Baseline
# -----------------------------------------------------------------------
function run_adam!(nlp, nlp_eval; lr::Float32=0.001f0, max_iter::Int=100, eval_freq::Int=100)
    x = copy(nlp.meta.x0)
    g = similar(x)
    epoch_losses = Float32[]
    
    opt_state = Optimisers.setup(Optimisers.Adam(lr), x)
    
    for i in 1:max_iter
        if i % eval_freq == 0 || i == 1
            push!(epoch_losses, obj(nlp_eval, x))
            @printf "Adam Epoch %3d | Full Train Loss: %.4f\n" (i ÷ eval_freq) epoch_losses[end]
        end

        _, _ = objgrad!(nlp, x, g)
        opt_state, x = Optimisers.update(opt_state, x, g)
    end
    return x, epoch_losses
end

x_adam, adam_epoch_loss = run_adam!(make_nlp(), nlp_eval; lr=0.001f0, max_iter=1500, eval_freq=100)

# -----------------------------------------------------------------------
# 4. TADAM Benchmark
# -----------------------------------------------------------------------
tadam_epoch_loss  = Float32[]
nlp_tadam   = make_nlp()

const success_count = Ref(0)
const batch_update_freq = 1

function tadam_callback(nlp, solver, stats)
    if stats.iter % 100 == 0 || stats.iter == 1
        push!(tadam_epoch_loss, obj(nlp_eval, solver.x))
        @printf "TADAM Epoch %3d | Full Train Loss: %.4f\n" (stats.iter ÷ 100) tadam_epoch_loss[end]
    end

    if solver.step_accepted
        success_count[] += 1
        if success_count[] >= batch_update_freq
            minibatch_next_train!(nlp)
            success_count[] = 0
        end
        stats.objective = obj(nlp, solver.x)
    end

    if stats.status == :small_step
        norm_gx = norm(solver.gx)
        solver.Δ = max(norm_gx / (2^round(log2(norm_gx + 1f0))), 1f-5)
        stats.status = :unknown
    end
end

stats_tadam = tadam(
    nlp_tadam;
    max_iter = 1500,
    atol     = 1f-8,
    rtol     = 1f-5,
    callback = tadam_callback,
    η2       = 0.550f0,
    γ3       = 0.02f0, 
    β1       = 0.90f0,
    β2       = 0.990f0,
    ϵ_v      = 1e-7,
    verbose  = 0
)

# -----------------------------------------------------------------------
# 5. Plot Results
# -----------------------------------------------------------------------
fig = Figure(size=(800, 500))
ax = CairoMakie.Axis(fig[1, 1];
    xlabel = "Epoch", ylabel = "Full Cross Entropy Loss",
    title  = "Adam vs TADAM Empirical Risk (FashionMNIST)", yscale = log10)

epochs_adam = 0:(length(adam_epoch_loss)-1)
epochs_tadam = 0:(length(tadam_epoch_loss)-1)

lines!(ax, epochs_adam, adam_epoch_loss;  label="Adam (lr=1e-3)", color=:gray50,     linewidth=2.5)
lines!(ax, epochs_tadam, tadam_epoch_loss; label="TADAM",         color=:dodgerblue, linewidth=2.5)

axislegend(ax; position=:rt)
display(fig)