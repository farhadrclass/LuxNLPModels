using Pkg
Pkg.activate("lux_test_env_2")

using Lux
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

# Assuming LuxNLPModels is available in your environment
using LuxNLPModels 

# -----------------------------------------------------------------------
# 1. Data Preparation (Train & Test)
# -----------------------------------------------------------------------
train_x, train_y = FashionMNIST.traindata(Float32)
test_x, test_y   = FashionMNIST.testdata(Float32)

# Subsample for speed during benchmarking
n_train, n_test = 10000, 2000
train_x = reshape(train_x, 28, 28, 1, :)[:, :, :, 1:n_train]
train_y = onehotbatch(train_y[1:n_train], 0:9)

test_x = reshape(test_x, 28, 28, 1, :)[:, :, :, 1:n_test]
test_y = onehotbatch(test_y[1:n_test], 0:9)

batch_size = 1000
batches_per_epoch = n_train ÷ batch_size
data_loader = DataLoader((train_x, train_y); batchsize=batch_size, shuffle=true)

# -----------------------------------------------------------------------
# 2. Model & Evaluation Setup
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

function calc_accuracy(ŷ, y)
    preds = [x.I[1] for x in argmax(ŷ, dims=1)]
    trues = [x.I[1] for x in argmax(y, dims=1)]
    return mean(preds .== trues)
end

make_nlp() = LuxNLPModel(model, copy(ps_cv), st, data_loader, loss_fn)

# -----------------------------------------------------------------------
# 3. History Tracking System
# -----------------------------------------------------------------------
mutable struct History
    iters::Vector{Int}
    batches_seen::Vector{Int} # Fair comparison metric
    times::Vector{Float64}
    train_loss::Vector{Float32}
    train_acc::Vector{Float32}
    test_loss::Vector{Float32}
    test_acc::Vector{Float32}
    
    # TADAM diagnostics
    deltas::Vector{Float64}
    rolling_rejections::Vector{Float64} 
    
    active_time::Float64
    timer_start::UInt64
    n_accepted::Int
    n_rejected::Int
end

function History()
    History(Int[], Int[], Float64[], Float32[], Float32[], Float32[], Float32[], 
            Float64[], Float64[], 0.0, time_ns(), 0, 0)
end

function evaluate_and_log!(hist::History, iter::Int, current_batches_seen::Int, ps_vec, delta::Real=0.0)
    # Stop timer
    hist.active_time += (time_ns() - hist.timer_start) / 1e9
    
    # Restructure flat parameters for Lux
    ps_structured = ComponentArray(ps_vec, getaxes(ps_cv))
    
    # Evaluate
    ŷ_train, _ = Lux.apply(model, train_x, ps_structured, st)
    tr_loss = loss_fn(ŷ_train, train_y)
    tr_acc  = calc_accuracy(ŷ_train, train_y)
    
    ŷ_test, _  = Lux.apply(model, test_x, ps_structured, st)
    te_loss = loss_fn(ŷ_test, test_y)
    te_acc  = calc_accuracy(ŷ_test, test_y)
    
    # Log metrics
    push!(hist.iters, iter)
    push!(hist.batches_seen, current_batches_seen)
    push!(hist.times, hist.active_time)
    push!(hist.train_loss, tr_loss)
    push!(hist.train_acc, tr_acc)
    push!(hist.test_loss, te_loss)
    push!(hist.test_acc, te_acc)
    
    # Diagnostics
    push!(hist.deltas, delta)
    rej_rate = (hist.n_accepted + hist.n_rejected) == 0 ? 0.0 : hist.n_rejected / (hist.n_accepted + hist.n_rejected)
    push!(hist.rolling_rejections, rej_rate)
    
    @printf "Iter %4d | Batches: %4d | Time: %5.1fs | Tr Loss: %.4f | Te Acc: %.2f%%\n" iter current_batches_seen hist.active_time tr_loss (te_acc*100)
    
    # Restart timer
    hist.timer_start = time_ns()
end

# -----------------------------------------------------------------------
# 4. Adam Baseline
# -----------------------------------------------------------------------
function run_adam!(nlp; lr::Float32=0.001f0, max_iter::Int=1500, eval_freq::Int=50)
    x = copy(nlp.meta.x0)
    g = similar(x)
    hist = History()
    batches_seen = 0
    
    opt_state = Optimisers.setup(Optimisers.Adam(lr), x)
    
    hist.timer_start = time_ns()
    for i in 0:max_iter
        if i % eval_freq == 0 || i == max_iter
            evaluate_and_log!(hist, i, batches_seen, x)
        end
        if i == max_iter break end

        _, _ = objgrad!(nlp, x, g)
        opt_state, x = Optimisers.update(opt_state, x, g)
        
        minibatch_next_train!(nlp)
        batches_seen += 1
    end
    return x, hist
end

println("--- Starting Adam ---")
nlp_adam = make_nlp()
x_adam, hist_adam = run_adam!(nlp_adam; lr=0.001f0, max_iter=1500, eval_freq=50)

# -----------------------------------------------------------------------
# 5. TADAM Benchmark
# -----------------------------------------------------------------------
function run_tadam!(nlp; max_iter::Int=1500, eval_freq::Int=50, kwargs...)
    hist = History()
    success_count = Ref(0)
    batches_seen = Ref(0)
    
    function tadam_callback(nlp, solver, stats)
        # Log at defined frequency
        if stats.iter % eval_freq == 0 || stats.iter == 0
            evaluate_and_log!(hist, stats.iter, batches_seen[], solver.x, solver.Δ)
        end

        if solver.step_accepted
            hist.n_accepted += 1
            success_count[] += 1
            
            # Timer paused during objective re-evaluation on new batch
            hist.active_time += (time_ns() - hist.timer_start) / 1e9
            if success_count[] >= 1 # batch_update_freq
                minibatch_next_train!(nlp)
                batches_seen[] += 1
                success_count[] = 0
            end
            stats.objective = obj(nlp, solver.x)
            hist.timer_start = time_ns()
            
        elseif stats.iter > 0
            hist.n_rejected += 1
        end

        if stats.status == :small_step
            norm_gx = norm(solver.gx)
            solver.Δ = max(norm_gx / (2^round(log2(norm_gx + 1f0))), 1f-5)
            stats.status = :unknown
        end
    end

    hist.timer_start = time_ns()
    stats = tadam(
        nlp;
        max_iter = max_iter,
        atol     = 1f-8,
        rtol     = 1f-5,
        callback = tadam_callback,
        verbose  = 0,
        kwargs...
    )
    
    if hist.iters[end] != stats.iter
        evaluate_and_log!(hist, stats.iter, batches_seen[], stats.solution, hist.deltas[end])
    end
    rej_rate = hist.n_rejected / max(1, (hist.n_accepted + hist.n_rejected))
    @printf "TADAM Final Rejection Rate: %.1f%%\n" (rej_rate * 100)
    
    return stats, hist
end

println("\n--- Starting TADAM ---")
nlp_tadam = make_nlp()
stats_tadam, hist_tadam = run_tadam!(
    nlp_tadam; 
    max_iter = 1500, 
    eval_freq = 50,
    θ2 = 1.0f0,    # Added explicit override
    η2 = 0.550f0,
    γ3 = 0.02f0, 
    β1 = 0.90f0,
    β2 = 0.990f0,
    ϵ_v = 1e-7
)

# -----------------------------------------------------------------------
# 6. Publication-Ready Dashboard
# -----------------------------------------------------------------------
fig = Figure(size=(1200, 900))
c_adam  = :gray50
c_tadam = :dodgerblue

# Row 1: Loss
ax1 = CairoMakie.Axis(fig[1, 1], xlabel = "Minibatches Seen", ylabel = "Train Loss", yscale = log10, title = "Loss vs Data Consumption")
ax2 = CairoMakie.Axis(fig[1, 2], xlabel = "Active Time (s)", ylabel = "Train Loss", yscale = log10, title = "Loss vs Time")

lines!(ax1, hist_adam.batches_seen,  hist_adam.train_loss,  label="Adam",  color=c_adam,  linewidth=2.5)
lines!(ax1, hist_tadam.batches_seen, hist_tadam.train_loss, label="TADAM", color=c_tadam, linewidth=2.5)

lines!(ax2, hist_adam.times,  hist_adam.train_loss,  label="Adam",  color=c_adam,  linewidth=2.5)
lines!(ax2, hist_tadam.times, hist_tadam.train_loss, label="TADAM", color=c_tadam, linewidth=2.5)

# Row 2: Test Accuracy
ax3 = CairoMakie.Axis(fig[2, 1], xlabel = "Minibatches Seen", ylabel = "Test Accuracy", title = "Test Acc vs Data Consumption")
ax4 = CairoMakie.Axis(fig[2, 2], xlabel = "Active Time (s)", ylabel = "Test Accuracy", title = "Test Acc vs Time")

lines!(ax3, hist_adam.batches_seen,  hist_adam.test_acc,  color=c_adam,  linewidth=2.5)
lines!(ax3, hist_tadam.batches_seen, hist_tadam.test_acc, color=c_tadam, linewidth=2.5)

lines!(ax4, hist_adam.times,  hist_adam.test_acc,  color=c_adam,  linewidth=2.5)
lines!(ax4, hist_tadam.times, hist_tadam.test_acc, color=c_tadam, linewidth=2.5)

# Row 3: Diagnostics (TADAM Only)
ax5 = CairoMakie.Axis(fig[3, 1], xlabel = "Iterations", ylabel = "Trust Region Radius (Δ)", yscale = log10, title = "Trust Region Dynamics")
ax6 = CairoMakie.Axis(fig[3, 2], xlabel = "Iterations", ylabel = "Rejection Rate", title = "Cumulative Rejection Rate")

lines!(ax5, hist_tadam.iters, hist_tadam.deltas, color=c_tadam, linewidth=2.0)
lines!(ax6, hist_tadam.iters, hist_tadam.rolling_rejections, color=c_tadam, linewidth=2.0)
ylims!(ax6, 0.0, 1.0)

axislegend(ax1; position=:rt)
display(fig)