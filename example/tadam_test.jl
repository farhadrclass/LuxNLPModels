## adversarial_regression.jl
##
## Compares three optimisers on an ill-conditioned, heavy-tailed regression task:
##   • SGD   – Optimisers.Descent (small LR required for stability)
##   • Adam  – Optimisers.Adam (diverges due to momentum poisoning)
##   • TADAM – Trust-region embedded Adam (converges safely)

using Pkg
Pkg.activate("lux_test_env_2")
# Pkg.develop(path=".")
# Pkg.add(["Lux", "ComponentArrays", "NLPModels", "JSOSolvers", "MLDatasets", "MLUtils", "OneHotArrays", "NNlib", "Optimisers", "CairoMakie"])

using Lux
using LuxNLPModels
using ComponentArrays
using NLPModels, JSOSolvers
using Optimisers
using MLUtils
using Random
using LinearAlgebra: norm
using Statistics: mean
using Printf
using CairoMakie

# -----------------------------------------------------------------------
# 1. Data (Ill-Conditioned + Cauchy Noise)
# -----------------------------------------------------------------------
function generate_adversarial_data(rng, n_features=50, n_samples=5000)
    # Ill-conditioned features (Eigenvalues span 1 to 100)
    scales = Float32.(10.0 .^ range(0, 2, length=n_features))
    x = randn(rng, Float32, n_features, n_samples) .* scales
    
    # True weights
    w_true = randn(rng, Float32, 1, n_features)
    
    # Cauchy noise (Heavy-tailed, undefined variance)
    noise = (randn(rng, Float32, 1, n_samples) ./ randn(rng, Float32, 1, n_samples))
    noise .*= 5.0 
    
    y = w_true * x .+ noise
    return x, y, vec(w_true)
end

rng = MersenneTwister()
Random.seed!(rng, 12345)

n_features = 50
x_data, y_data, w_true = generate_adversarial_data(rng, n_features, 5000)
data_loader = DataLoader((x_data, y_data); batchsize=5000, shuffle=false)

# -----------------------------------------------------------------------
# 2. Model setup
# -----------------------------------------------------------------------
model = Dense(n_features => 1, use_bias=false)
ps, st = Lux.setup(rng, model)
ps_cv  = ComponentArray(ps)

loss_fn(ŷ, y) = mean(abs2, ŷ .- y)

make_nlp() = LuxNLPModel(model, ps_cv, st, data_loader, loss_fn)

@printf "Parameters : %d\n"   length(ps_cv)
@printf "Loss (x₀)  : %.6f\n\n" obj(make_nlp(), ps_cv)

# -----------------------------------------------------------------------
# 3. Unified Optimisers.jl Loop (For SGD and Adam)
# -----------------------------------------------------------------------
function run_optimiser!(nlp, opt_rule; max_iter::Int=300)
    x = copy(nlp.meta.x0)
    g = similar(x)
    losses = Float32[]
    dists  = Float32[]
    
    # Setup Optimisers.jl state
    opt_state = Optimisers.setup(opt_rule, x)
    
    for _ in 1:max_iter
        f, _ = objgrad!(nlp, x, g)
        push!(losses, f)
        push!(dists, norm(x - w_true))
        
        # Unified update step
        opt_state, x = Optimisers.update(opt_state, x, g)
    end
    return x, losses, dists
end

# Run SGD
x_sgd, sgd_loss, sgd_dist = run_optimiser!(make_nlp(), Optimisers.Descent(1e-7); max_iter=300)
@printf "SGD   final loss : %.6f  (Distance to w_true: %.4f)\n" sgd_loss[end] sgd_dist[end]

# Run Adam
x_adam, adam_loss, adam_dist = run_optimiser!(make_nlp(), Optimisers.Adam(0.1f0); max_iter=300)
@printf "Adam  final loss : %.6f  (Distance to w_true: %.4f)\n" adam_loss[end] adam_dist[end]

# -----------------------------------------------------------------------
# 4. TADAM (JSOSolvers)
# -----------------------------------------------------------------------
tadam_loss = Float32[]
tadam_dist = Float32[]
nlp_tadam  = make_nlp()

function tadam_callback(nlp, solver, stats)
    push!(tadam_loss, stats.objective)
    push!(tadam_dist, norm(solver.x - w_true))
end

stats_tadam = tadam(
    nlp_tadam;
    max_iter = 300,
    atol     = 1f-5,
    rtol     = 0f0,
    callback = tadam_callback,
    verbose  = 0
)
@printf "TADAM final loss : %.6f  (Distance to w_true: %.4f)\n" stats_tadam.objective tadam_dist[end]

# -----------------------------------------------------------------------
# 5. Plot Results
# -----------------------------------------------------------------------
fig = Figure(size=(1100, 440))

# Left Panel
ax_loss = CairoMakie.Axis(fig[1, 1];
    xlabel = "Iteration", ylabel = "MSE Loss",
    title  = "Loss Curves (Cauchy Noise)", yscale = log10)

lines!(ax_loss, sgd_loss;   label="SGD (lr=1e-7)", color=:tomato,      linewidth=2)
lines!(ax_loss, adam_loss;  label="Adam (lr=0.1)", color=:gray50,      linewidth=2)
lines!(ax_loss, tadam_loss; label="TADAM",         color=:dodgerblue,  linewidth=2.5)
axislegend(ax_loss; position=:rt)

# Right Panel
ax_dist = CairoMakie.Axis(fig[1, 2];
    xlabel = "Iteration", ylabel = "||w - w_true||",
    title  = "Distance to True Weights")

lines!(ax_dist, sgd_dist;   label="SGD",   color=:tomato,     linewidth=2)
lines!(ax_dist, adam_dist;  label="Adam",  color=:gray50,     linewidth=2)
lines!(ax_dist, tadam_dist; label="TADAM", color=:dodgerblue, linewidth=2.5)
axislegend(ax_dist; position=:rt)

display(fig)