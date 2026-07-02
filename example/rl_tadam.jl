## rl_td_benchmark.jl
##
## Proves Adam's momentum divergence under target shifts in TD learning.
## Demonstrates TADAM's momentum restriction (Condition 22) securing the Bellman projection.

using Pkg
Pkg.activate("lux_test_env_2")

using Lux
using LuxNLPModels
using ComponentArrays
using NLPModels, JSOSolvers
using Optimisers
using MLUtils
using Random
using LinearAlgebra: norm
using Printf
using CairoMakie

# -----------------------------------------------------------------------
# 1. Synthetic Continuous MDP Generation
# -----------------------------------------------------------------------
rng = MersenneTwister(42)

const d_state = 10
const n_samples = 2000
const gamma = 0.99f0
const target_update_freq = 20

S_train = randn(rng, Float32, d_state, n_samples)
# Transition dynamics: s' = 0.8 * s + noise
S_next  = 0.8f0 .* S_train .+ 0.1f0 .* randn(rng, Float32, d_state, n_samples)
# Reward function: r = -||s||_2^2
R_train = -sum(S_train.^2, dims=1)

# Mutable target tensor (y)
Y_train = copy(R_train)

data_loader = DataLoader((S_train, Y_train); batchsize=n_samples, shuffle=false)

# -----------------------------------------------------------------------
# 2. Value Function Approximation (Critic)
# -----------------------------------------------------------------------
model = Chain(
    Dense(d_state => 64, tanh),
    Dense(64 => 64, tanh),
    Dense(64 => 1)
)

ps, st = Lux.setup(rng, model)
ps_cv  = ComponentArray(ps)

loss_fn(ŷ, y) = sum(abs2, ŷ .- y) / size(y, 2)
make_nlp() = LuxNLPModel(model, ps_cv, st, data_loader, loss_fn)

# -----------------------------------------------------------------------
# 3. Adam Baseline (Demonstrates Overshoot Divergence)
# -----------------------------------------------------------------------
function run_adam_td!(nlp, st, S_next, R_train, ps_axes; lr=0.005f0, max_iter=300)
    x = copy(nlp.meta.x0)
    g = similar(x)
    losses = Float32[]
    opt_state = Optimisers.setup(Optimisers.Adam(lr), x)
    
    @info "Starting Adam TD Learning..."
    for i in 1:max_iter
        # Periodic Target Network Update
        if i % target_update_freq == 0
            ps_current = ComponentArray(x, ps_axes)
            V_next, _ = Lux.apply(model, S_next, ps_current, st)
            Y_train .= R_train .+ gamma .* V_next
        end
        
        f, _ = objgrad!(nlp, x, g)
        push!(losses, f)
        
        opt_state, x = Optimisers.update(opt_state, x, g)
    end
    return losses
end

adam_loss = run_adam_td!(make_nlp(), st, S_next, R_train, getaxes(ps_cv))

# -----------------------------------------------------------------------
# 4. TADAM (Demonstrates Robust Bellman Projection)
# -----------------------------------------------------------------------
# Reset targets
Y_train .= R_train
nlp_tadam = make_nlp()
tadam_loss = Float32[]
success_count = Ref(0)

function tadam_td_callback(nlp, solver, stats)
    push!(tadam_loss, stats.objective)
    
    if solver.step_accepted
        success_count[] += 1
        
        # Target Network Update mapped to accepted algorithmic steps
        if success_count[] % target_update_freq == 0
            ps_current = ComponentArray(solver.x, getaxes(ps_cv))
            V_next, _ = Lux.apply(model, S_next, ps_current, st)
            Y_train .= R_train .+ gamma .* V_next
            
            # Re-anchor the base objective to the newly shifted landscape
            stats.objective = obj(nlp, solver.x)
        end
    end
end

@info "Starting TADAM TD Learning..."
stats_tadam = tadam(
    nlp_tadam;
    max_iter = 300,
    atol     = 1f-8,
    rtol     = 1f-8,
    β1       = 0.9f0,
    β2       = 0.99f0,
    ϵ_v      = 1f-6,
    callback = tadam_td_callback,
    verbose  = 0
)
# -----------------------------------------------------------------------
# 5. Plot Results
# -----------------------------------------------------------------------
fig = Figure(size=(800, 500))
ax = CairoMakie.Axis(fig[1, 1];
    xlabel = "Iteration", ylabel = "Bellman Error (TD-Loss)",
    title  = "Target Shift Instability: Adam vs TADAM", yscale = log10)

lines!(ax, adam_loss;  label="Adam (lr=0.005)", color=:tomato,      linewidth=2.0)
lines!(ax, tadam_loss; label="TADAM",           color=:dodgerblue,  linewidth=2.5)

# Add vertical lines for target updates
for i in target_update_freq:target_update_freq:300
    vlines!(ax, [i]; color=:gray80, linestyle=:dash)
end

axislegend(ax; position=:rt)
display(fig)