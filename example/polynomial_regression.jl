## polynomial_regression.jl
##
## Compares three optimisers on a noisy quadratic regression task:
##   • SGD  – manual gradient-descent loop via the NLPModels interface
##   • R2   – JSOSolvers first-order trust-region method
##   • LBFGS – JSOSolvers quasi-Newton method
##
## All three share the same LuxNLPModel infrastructure and the same
## random starting point, so results are directly comparable.
##
## Pkg.add(["Lux", "ComponentArrays", "NLPModels", "JSOSolvers",
##          "MLUtils", "CairoMakie"])

using Pkg
Pkg.activate("lux_test_env")
Pkg.develop(path=".")

using Lux
using LuxNLPModels
using ComponentArrays
using NLPModels, JSOSolvers
using MLUtils
using Random
using Statistics: mean
using Printf
using CairoMakie

# -----------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------
function generate_data(rng)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, Ref((0f0, -2f0, 1f0)))   # 0 − 2x + x²
    y .+= randn(rng, Float32, size(y)) .* 0.1f0
    return x, y
end

rng = MersenneTwister()
Random.seed!(rng, 12345)

x_data, y_data = generate_data(rng)
data_loader    = DataLoader((x_data, y_data); batchsize=128, shuffle=false)

# -----------------------------------------------------------------------
# 2. Model  (fixed for all three runs)
# -----------------------------------------------------------------------
model = Chain(Dense(1 => 16, relu), Dense(16 => 1))
ps, st = Lux.setup(rng, model)
ps_cv  = ComponentArray(ps)          # flat parameter vector + axes

loss_fn(ŷ, y) = mean(abs2, ŷ .- y)  # MSE

# Each solver gets a fresh NLPModel with the same x0 so comparisons are fair.
make_nlp() = LuxNLPModel(model, ps_cv, st, data_loader, loss_fn)

@printf "Parameters : %d\n"   length(ps_cv)
@printf "Loss (x₀)  : %.6f\n" obj(make_nlp(), ps_cv)

# -----------------------------------------------------------------------
# 3a. SGD  – plain gradient descent via objgrad!
# -----------------------------------------------------------------------
function run_sgd!(nlp; lr::Float32=0.02f0, max_iter::Int=800)
    x      = copy(nlp.meta.x0)
    g      = similar(x)
    losses = Float32[]
    for _ in 1:max_iter
        f, _ = objgrad!(nlp, x, g)
        push!(losses, f)
        x .-= lr .* g          # x ← x − α ∇f
    end
    return x, losses
end

nlp_sgd          = make_nlp()
x_sgd, sgd_hist  = run_sgd!(nlp_sgd; lr=0.02f0, max_iter=800)
@printf "\nSGD   final loss : %.6f  (%d iters)\n" sgd_hist[end] length(sgd_hist)

# -----------------------------------------------------------------------
# 3b. R2  – first-order trust-region (one AD pass per iter, like SGD)
# -----------------------------------------------------------------------
r2_hist  = Float32[]
nlp_r2   = make_nlp()
stats_r2 = R2(
    nlp_r2;
    max_iter = 800,
    atol     = 1f-8,
    rtol     = 0f0,
    callback = (nlp, solver, stats) -> push!(r2_hist, stats.objective),
)
@printf "R2    final loss : %.6f  (%d iters)\n" stats_r2.objective stats_r2.iter

# -----------------------------------------------------------------------
# 3c. L-BFGS  – quasi-Newton (line-search; fewer iters, more work/iter)
# -----------------------------------------------------------------------
lbfgs_hist  = Float32[]
nlp_lbfgs   = make_nlp()
stats_lbfgs = lbfgs(
    nlp_lbfgs;
    max_iter = 300,
    atol     = 1f-8,
    callback = (nlp, solver, stats) -> push!(lbfgs_hist, stats.objective),
)
@printf "LBFGS final loss : %.6f  (%d iters)\n" stats_lbfgs.objective stats_lbfgs.iter

# -----------------------------------------------------------------------
# 4. Inference  (reconstruct ComponentVector from flat solution)
# -----------------------------------------------------------------------
function predict(solution)
    opt_ps  = ComponentArray(solution, getaxes(ps_cv))
    ŷ, _    = Lux.apply(model, x_data, opt_ps, Lux.testmode(st))
    return ŷ[1, :]
end

ŷ_sgd   = predict(x_sgd)
ŷ_r2    = predict(stats_r2.solution)
ŷ_lbfgs = predict(stats_lbfgs.solution)

# -----------------------------------------------------------------------
# 5. Plot
# -----------------------------------------------------------------------
fig = Figure(size=(1100, 440))

# ── Left panel: loss curves ─────────────────────────────────────────────
ax_loss = CairoMakie.Axis(fig[1, 1];
    xlabel = "Iteration", ylabel = "MSE loss",
    title  = "Loss curves", yscale = log10)

lines!(ax_loss, sgd_hist;              label="SGD  (lr=0.02)",  color=:tomato,      linewidth=2)
lines!(ax_loss, r2_hist;               label="R2",              color=:dodgerblue,  linewidth=2)
lines!(ax_loss, lbfgs_hist;            label="L-BFGS",          color=:seagreen,    linewidth=2)
axislegend(ax_loss; position=:rt)

# ── Right panel: fitted curves ──────────────────────────────────────────
ax_fit = CairoMakie.Axis(fig[1, 2];
    xlabel = "x", ylabel = "y",
    title  = "Fitted curves")

x_vec  = x_data[1, :]
x_line = range(-2.0f0, 2.0f0; length=300)

lines!(ax_fit,   x_line, t -> evalpoly(t, (0f0, -2f0, 1f0));
       label="True function",  color=:black,      linewidth=2, linestyle=:dash)
scatter!(ax_fit, x_vec,  y_data[1, :];
       label="Noisy data",     color=(:gray, 0.5), markersize=5)
lines!(ax_fit,   x_vec,  ŷ_sgd;
       label="SGD",            color=:tomato,      linewidth=2)
lines!(ax_fit,   x_vec,  ŷ_r2;
       label="R2",             color=:dodgerblue,  linewidth=2)
lines!(ax_fit,   x_vec,  ŷ_lbfgs;
       label="L-BFGS",         color=:seagreen,    linewidth=2)

axislegend(ax_fit; position=:lt)

display(fig)

