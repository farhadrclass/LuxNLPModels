# ================================================================
# sensitivity_study.jl — Experiment 1: Hyperparameter Sensitivity
#
# Key message:
#   Adam test accuracy varies wildly across learning rates.
#   Tadam accuracy is stable across 3 decades of initial Δ₀,
#   demonstrating that its automatic step-size adaptation removes
#   the need for careful learning-rate tuning.
#
# Produces: sensitivity_study.{pdf,png}  (5-panel figure)
# ================================================================

using Pkg
Pkg.activate("lux_test_env_2")

using Lux
using LinearAlgebra: norm
using ComponentArrays
using NLPModels, JSOSolvers, LuxNLPModels
using Optimisers
using MLDatasets
using MLUtils
using OneHotArrays
using NNlib: logsoftmax
using Random
using Statistics: mean, std
using Printf
using CairoMakie

# ────────────────────────────────────────────────────────────────
# 1. FashionMNIST  (fast enough for a multi-configuration sweep)
# ────────────────────────────────────────────────────────────────
train_x_raw, train_y_raw = FashionMNIST.traindata(Float32)
test_x_raw,  test_y_raw  = FashionMNIST.testdata(Float32)

N_TRAIN, N_TEST = 10_000, 2_000

train_x = reshape(train_x_raw, 28, 28, 1, :)[:,:,:,1:N_TRAIN]
train_y = onehotbatch(train_y_raw[1:N_TRAIN], 0:9)
test_x  = reshape(test_x_raw,  28, 28, 1, :)[:,:,:,1:N_TEST]
test_y  = onehotbatch(test_y_raw[1:N_TEST], 0:9)

BATCH = 1000
data_loader = DataLoader((train_x, train_y); batchsize=BATCH, shuffle=true)

# ────────────────────────────────────────────────────────────────
# 2. Model  (same MLP as the FashionMNIST baseline)
# ────────────────────────────────────────────────────────────────
model = Chain(
    FlattenLayer(),
    Dense(28 * 28 => 128, relu),
    Dense(128    => 64,  relu),
    Dense(64     => 10),
)

rng = MersenneTwister(42)
ps_init, st = Lux.setup(rng, model)
const ps_template = ComponentArray(ps_init)

loss_fn(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))

function eval_metrics(ps_vec)
    ps_s = ComponentArray(ps_vec, getaxes(ps_template))
    ŷ_tr, _ = Lux.apply(model, train_x, ps_s, st)
    ŷ_te, _ = Lux.apply(model, test_x,  ps_s, st)
    tr_loss  = Float32(loss_fn(ŷ_tr, train_y))
    acc(ŷ, y) = mean([i.I[1] for i in argmax(ŷ; dims=1)] .==
                     [i.I[1] for i in argmax(y; dims=1)])
    tr_loss, Float32(acc(ŷ_tr, train_y)), Float32(acc(ŷ_te, test_y))
end

# Fresh NLP model for every run so parameters always start from the same x₀
make_nlp() = LuxNLPModel(model, copy(ps_template), st, data_loader, loss_fn)

# ────────────────────────────────────────────────────────────────
# 3. Minimal history struct for this experiment
# ────────────────────────────────────────────────────────────────
mutable struct SensHist
    batches :: Vector{Int}
    tr_loss :: Vector{Float32}
    te_acc  :: Vector{Float32}
end
SensHist() = SensHist(Int[], Float32[], Float32[])

function record!(h::SensHist, batches, ps_vec)
    tr_loss, _, te_acc = eval_metrics(ps_vec)
    push!(h.batches, batches)
    # Guard against diverged runs (NaN / Inf) so log-scale plots don't break
    push!(h.tr_loss, isfinite(tr_loss) ? tr_loss : Inf32)
    push!(h.te_acc,  isfinite(te_acc)  ? te_acc  : 0f0)
end

# ────────────────────────────────────────────────────────────────
# 4a. Adam runner (one lr value)
# ────────────────────────────────────────────────────────────────
function run_adam_lr(lr::Float32; max_iter=800, eval_freq=50)
    nlp = make_nlp()
    x   = copy(nlp.meta.x0)
    g   = similar(x)
    h   = SensHist()
    opt = Optimisers.setup(Optimisers.Adam(lr), x)
    batches = 0
    for i in 0:max_iter
        i % eval_freq == 0 && record!(h, batches, x)
        i == max_iter && break
        objgrad!(nlp, x, g)
        opt, x = Optimisers.update(opt, x, g)
        minibatch_next_train!(nlp)
        batches += 1
    end
    @printf "  Adam lr=%.0e  →  final te_acc=%.3f\n" lr h.te_acc[end]
    return h
end

# ────────────────────────────────────────────────────────────────
# 4b. Tadam runner (one Δ₀ value)
#
# NOTE: the keyword that sets the initial TR radius depends on your
#       JSOSolvers version.  Try `Δ = Δ0` first; if that errors,
#       check fieldnames(TrustRegionSolver) or the solver's help.
# ────────────────────────────────────────────────────────────────
function run_tadam_delta0(Δ0::Float32; max_iter=800, eval_freq=50)
    nlp     = make_nlp()
    h       = SensHist()
    batches = Ref(0)

    cb = (nlp, solver, stats) -> begin
        stats.iter % eval_freq == 0 &&
            record!(h, batches[], copy(solver.x))

        if solver.step_accepted
            minibatch_next_train!(nlp)
            batches[] += 1
            stats.objective = obj(nlp, solver.x)
        end

        # Stall-prevention heuristic (same as baseline script)
        if stats.status == :small_step
            ng = norm(solver.gx)
            solver.Δ = max(ng / (2^round(log2(ng + 1f0))), 1f-5)
            stats.status = :unknown
        end
    end

    tadam(nlp;
        max_iter, atol=1f-8, rtol=1f-5, callback=cb, verbose=0,
        Δ   = Δ0,       # ← initial trust-region radius
        η1  = 0.10f0,   # reject step  if ρ_k < η₁
        η2  = 0.55f0,   # expand TR    if ρ_k ≥ η₂
        γ1  = 0.50f0,   # contraction factor
        γ2  = 2.00f0,   # expansion factor
        β1  = 0.90f0,   # β_mom
        β2  = 0.99f0,   # β_rms
        ϵ_v = 1f-7,     # ε_Adam
        # θ2  = 100.00f0,   # Cauchy-decrease fraction θ
    )

    @printf "  Tadam Δ₀=%.0e  →  final te_acc=%.3f\n" Δ0 h.te_acc[end]
    return h
end

# ────────────────────────────────────────────────────────────────
# 5. Sweep
# ────────────────────────────────────────────────────────────────
# 4 values spanning 3 decades for each method
const ADAM_LRS  = Float32[1e-1, 1e-2, 1e-3, 1e-4]
const TADAM_Δ0S = Float32[1e-2, 1e-1, 1e0,  1e1 ]
const MAX_ITER  = 800
const EVAL_FREQ = 50

println("=== Adam LR sweep ===")
adam_hists = [run_adam_lr(lr;   max_iter=MAX_ITER, eval_freq=EVAL_FREQ) for lr  in ADAM_LRS]

println("\n=== Tadam Δ₀ sweep ===")
tadam_hists = [run_tadam_delta0(Δ0; max_iter=MAX_ITER, eval_freq=EVAL_FREQ) for Δ0 in TADAM_Δ0S]

# ────────────────────────────────────────────────────────────────
# 6. Publication figure  (5 panels)
#
#   (a) Adam    test-accuracy curves (one curve per lr)
#   (b) Tadam  test-accuracy curves (one curve per Δ₀)
#   (c) Adam    train-loss curves    (log scale)
#   (d) Tadam  train-loss curves    (log scale)
#   (e) Summary: final test accuracy vs hyperparameter value
# ────────────────────────────────────────────────────────────────

# Warm palette for Adam (dark = extreme lr)
ADAM_COLORS = [
    RGBf(0.80, 0.10, 0.10),   # lr = 0.1     (likely diverges)
    RGBf(0.88, 0.40, 0.00),   # lr = 0.01
    RGBf(0.90, 0.65, 0.00),   # lr = 0.001   (usually best)
    RGBf(0.95, 0.88, 0.55),   # lr = 0.0001  (too slow)
]

# Cool palette for Tadam (dark = large Δ₀)
TADAM_COLORS = [
    RGBf(0.75, 0.90, 1.00),   # Δ₀ = 0.01
    RGBf(0.35, 0.60, 0.90),   # Δ₀ = 0.1
    RGBf(0.00, 0.45, 0.70),   # Δ₀ = 1.0
    RGBf(0.00, 0.20, 0.50),   # Δ₀ = 10.0
]

ADAM_LABELS  = ["lr = 0.1", "lr = 0.01", "lr = 0.001", "lr = 0.0001"]
TADAM_LABELS = ["Δ₀ = 0.01", "Δ₀ = 0.1", "Δ₀ = 1.0", "Δ₀ = 10.0"]

pub_theme = Theme(
    fontsize = 13,
    Axis = (
        spinewidth = 0.9,
        xgridcolor = (:black, 0.07), ygridcolor = (:black, 0.07),
        titlesize = 12, xlabelsize = 12, ylabelsize = 12,
        xticklabelsize = 11, yticklabelsize = 11,
    ),
    Legend = (framevisible = false, labelsize = 10, patchsize = (18, 2)),
    Lines  = (linewidth = 2.2,),
)

with_theme(pub_theme) do
    fig = Figure(size = (900, 900))

    # ── (a) Adam test-accuracy curves ────────────────────────────
    ax_a = Axis(fig[1,1];
        xlabel = "Minibatches", ylabel = "Test accuracy",
        title  = "(a) Adam: sensitivity to learning rate lr")
    for (h, col, lbl) in zip(adam_hists, ADAM_COLORS, ADAM_LABELS)
        lines!(ax_a, h.batches, h.te_acc; color=col, label=lbl)
    end
    axislegend(ax_a; position=:rb)

    # ── (b) Tadam test-accuracy curves ──────────────────────────
    ax_b = Axis(fig[1,2];
        xlabel = "Minibatches", ylabel = "Test accuracy",
        title  = "(b) Tadam: robustness to initial radius Δ₀")
    for (h, col, lbl) in zip(tadam_hists, TADAM_COLORS, TADAM_LABELS)
        lines!(ax_b, h.batches, h.te_acc; color=col, label=lbl)
    end
    axislegend(ax_b; position=:rb)

    # Link y-axes so panels (a) and (b) are directly comparable
    linkyaxes!(ax_a, ax_b)

    # ── (c) Adam train-loss (log scale) ──────────────────────────
    ax_c = Axis(fig[2,1];
        xlabel = "Minibatches", ylabel = "Train loss", yscale=log10,
        title  = "(c) Adam: train loss by learning rate")
    for (h, col, lbl) in zip(adam_hists, ADAM_COLORS, ADAM_LABELS)
        finite = isfinite.(h.tr_loss)
        any(finite) && lines!(ax_c, h.batches[finite], h.tr_loss[finite]; color=col, label=lbl)
    end
    axislegend(ax_c; position=:rt)

    # ── (d) Tadam train-loss (log scale) ────────────────────────
    ax_d = Axis(fig[2,2];
        xlabel = "Minibatches", ylabel = "Train loss", yscale=log10,
        title  = "(d) Tadam: train loss by initial Δ₀")
    for (h, col, lbl) in zip(tadam_hists, TADAM_COLORS, TADAM_LABELS)
        lines!(ax_d, h.batches, h.tr_loss; color=col, label=lbl)
    end
    axislegend(ax_d; position=:rt)

    linkyaxes!(ax_c, ax_d)

    # ── (e) Robustness summary ────────────────────────────────────
    # One scatter per configuration; horizontal bands show the range.
    # A narrow band → insensitive to hyperparameter choice.
    ax_e = Axis(fig[3, :];
        xlabel = "Hyperparameter value (log scale)",
        ylabel = "Final test accuracy",
        xscale = log10,
        title  = "(e) Robustness summary: final test accuracy across sweep")

    final_adam  = [h.te_acc[end] for h in adam_hists]
    final_tadam = [h.te_acc[end] for h in tadam_hists]

    # Shaded band: width encodes sensitivity (wide = sensitive)
    hspan!(ax_e, minimum(final_adam),  maximum(final_adam);
           color = (RGBf(0.88, 0.40, 0.00), 0.12),
           label = "Adam range")
    hspan!(ax_e, minimum(final_tadam), maximum(final_tadam);
           color = (RGBf(0.00, 0.45, 0.70), 0.12),
           label = "Tadam range")

    # Individual configurations
    scatter!(ax_e, ADAM_LRS,  final_adam;
             color=ADAM_COLORS,  markersize=15, marker=:rect,
             label="Adam (lr sweep)")
    scatter!(ax_e, TADAM_Δ0S, final_tadam;
             color=TADAM_COLORS, markersize=15, marker=:circle,
             label="Tadam (Δ₀ sweep)")

    # Annotate with standard deviation — the headline statistic
    σ_adam  = round(std(final_adam)  * 100; digits=2)
    σ_tadam = round(std(final_tadam) * 100; digits=2)
    x_ann   = minimum([ADAM_LRS; TADAM_Δ0S]) * 1.2

    text!(ax_e, x_ann, maximum(final_adam) + 0.004f0;
          text="σ(Adam) = $(σ_adam)%  ← larger = more sensitive",
          color=RGBf(0.70, 0.10, 0.00), fontsize=10)
    text!(ax_e, x_ann, minimum(final_tadam) - 0.012f0;
          text="σ(Tadam) = $(σ_tadam)%  ← smaller = more robust",
          color=RGBf(0.00, 0.25, 0.55), fontsize=10)

    axislegend(ax_e; position=:lt)

    # ── caption ──────────────────────────────────────────────────
    Label(fig[4, :],
        "FashionMNIST (N=$(N_TRAIN) train, batch=$(BATCH), $(MAX_ITER) iterations). " *
        "Both methods started from the same random seed.\n" *
        "Panels (a–d): training dynamics. " *
        "Panel (e): σ measures sensitivity — lower is better for Tadam.",
        tellwidth=false, fontsize=10, color=:gray40)

    save("sensitivity_study.pdf", fig)
    save("sensitivity_study.png", fig; px_per_unit=2)
    display(fig)
    @info "Saved sensitivity_study.{pdf,png}"
end

# ────────────────────────────────────────────────────────────────
# 7. Print summary table (useful for a paper table)
# ────────────────────────────────────────────────────────────────
println("\n── Sensitivity summary ─────────────────────────────")
println("Adam:")
for (lr, h) in zip(ADAM_LRS, adam_hists)
    @printf "  lr = %.0e   best_te=%.3f   final_te=%.3f\n" lr maximum(h.te_acc) h.te_acc[end]
end
println("Tadam:")
for (Δ0, h) in zip(TADAM_Δ0S, tadam_hists)
    @printf "  Δ₀ = %.0e   best_te=%.3f   final_te=%.3f\n" Δ0 maximum(h.te_acc) h.te_acc[end]
end
σ_adam  = round(std([h.te_acc[end] for h in adam_hists])  * 100; digits=2)
σ_tadam = round(std([h.te_acc[end] for h in tadam_hists]) * 100; digits=2)
println("\nσ(Adam final te_acc)  = $(σ_adam)%")
println("σ(Tadam final te_acc) = $(σ_tadam)%")
println("Sensitivity ratio: $(round(σ_adam / max(σ_tadam, 1e-4); digits=1))×")