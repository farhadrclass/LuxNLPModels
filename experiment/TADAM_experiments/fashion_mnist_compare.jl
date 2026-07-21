# ================================================================
# fashion_mnist_compare.jl
# Fashion-MNIST Benchmark: Adam | AMSGrad | Tadam
#
# Laptop-friendly vs the CIFAR-10 script:
#   - CPU only (no CUDA dependency)
#   - MLP instead of CNN  (784 → 512 → 256 → 10)
#   - 10 000 train / 2 000 test samples
#   - 1 000 iterations, eval every 25
#   - Expected runtime: ~2–4 min on a modern laptop CPU
#
# Bug-fixes over cifar_10_MLP.jl:
#   - `isempty` guard on h.iters[end] in run_tadam! (was a BoundsError risk)
#   - Removed dead `success` Ref (was incremented then immediately reset)
# ================================================================

# using Pkg
# Pkg.activate("lux_test_env_2")   # reuse the same environment

using Lux
using ComponentArrays
using NLPModels, JSOSolvers, LuxNLPModels
using Optimisers
using MLDatasets
using MLUtils
using OneHotArrays
using NNlib: logsoftmax
using Random, Statistics, Printf, LinearAlgebra
using CairoMakie
import CairoMakie: Axis

# ----------------------------------------------------------------
# 1.  Fashion-MNIST  (28 × 28, grayscale, 10 classes)
# ----------------------------------------------------------------
@info "Loading Fashion-MNIST..."
tr_x_raw, tr_y_raw = FashionMNIST.traindata(Float32)   # → (28,28,60_000)
te_x_raw, te_y_raw = FashionMNIST.testdata(Float32)    # → (28,28,10_000)

# MLDatasets returns (H, W, N) for grayscale; add channel dim → (H, W, 1, N)
# so FlattenLayer gets a consistent 4-D tensor and Lux stays happy.
add_ch(x) = reshape(x, size(x, 1), size(x, 2), 1, size(x, 3))

N_TRAIN, N_TEST = 10_000, 2_000

train_x = add_ch(tr_x_raw)[:, :, :, 1:N_TRAIN]
train_y = onehotbatch(tr_y_raw[1:N_TRAIN], 0:9)
test_x  = add_ch(te_x_raw)[:, :, :, 1:N_TEST]
test_y  = onehotbatch(te_y_raw[1:N_TEST], 0:9)

const BATCH = 512
data_loader = DataLoader(
    (train_x, train_y);
    batchsize = BATCH, shuffle = true,
)

# ----------------------------------------------------------------
# 2.  MLP model  (28×28 = 784 inputs, no convolutions)
#     Much faster than CNN on CPU; still competitive on MNIST-scale.
# ----------------------------------------------------------------
model = Chain(
    FlattenLayer(),
    Dense(784 => 512, relu),
    Dense(512 => 256, relu),
    Dense(256 => 10),
)

rng_init = MersenneTwister(42)
ps_cpu, st = Lux.setup(rng_init, model)

const ps_template = ComponentArray(ps_cpu)   # axis template (CPU)
const ps0         = copy(ps_template)        # shared starting point (all methods)

loss_fn(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))

# Evaluate on fixed train/test splits (CPU).
# 10 k samples + MLP → fast enough to call every eval_freq steps.
function eval_metrics(ps_vec)
    ps_s     = ComponentArray(Array(ps_vec), getaxes(ps_template))
    ŷ_tr, _  = Lux.apply(model, train_x, ps_s, st)
    ŷ_te, _  = Lux.apply(model, test_x,  ps_s, st)
    tr_loss  = Float32(loss_fn(ŷ_tr, train_y))
    acc(ŷ, y) = mean(
        [i.I[1] for i in argmax(ŷ; dims=1)] .==
        [i.I[1] for i in argmax(y;  dims=1)],
    )
    Float32(tr_loss), Float32(acc(ŷ_tr, train_y)), Float32(acc(ŷ_te, test_y))
end

make_nlp() = LuxNLPModel(model, copy(ps0), st, data_loader, loss_fn)

# ----------------------------------------------------------------
# 3.  History struct
# ----------------------------------------------------------------
mutable struct Hist
    # Low-frequency evaluation snapshots
    iters   :: Vector{Int}
    batches :: Vector{Int}
    times   :: Vector{Float64}
    tr_loss :: Vector{Float32}
    tr_acc  :: Vector{Float32}
    te_acc  :: Vector{Float32}
    # Tadam high-frequency diagnostics
    hf_iters :: Vector{Int}
    hf_delta :: Vector{Float64}   # trust-region radius Δ_k
    hf_rej   :: Vector{Float64}   # running rejection rate
    n_ok     :: Int
    n_rej    :: Int
    # Internal timer (eval time is excluded from wall-clock)
    _t_accum :: Float64
    _t_start :: UInt64
end

Hist() = Hist(
    Int[], Int[], Float64[], Float32[], Float32[], Float32[],
    Int[], Float64[], Float64[], 0, 0, 0.0, time_ns(),
)

function snap!(h::Hist, iter, batches, ps_vec, tag)
    h._t_accum += (time_ns() - h._t_start) / 1e9   # accumulate before eval
    tr_loss, tr_acc, te_acc = eval_metrics(ps_vec)
    push!(h.iters,   iter);    push!(h.batches, batches)
    push!(h.times,   h._t_accum)
    push!(h.tr_loss, tr_loss); push!(h.tr_acc, tr_acc); push!(h.te_acc, te_acc)
    @printf "[%s] iter=%4d  bat=%4d  t=%5.1fs  loss=%.4f  tr=%.1f%%  te=%.1f%%\n"  tag iter batches h._t_accum tr_loss (100tr_acc) (100te_acc)
    h._t_start = time_ns()     # restart timer after eval
end

# ----------------------------------------------------------------
# 4a.  Runner: first-order optimisers (Adam, AMSGrad)
# ----------------------------------------------------------------
function run_first_order!(rule, name; max_iter = 1000, eval_freq = 25)
    @info "=== $name ==="
    nlp     = make_nlp()
    x       = copy(nlp.meta.x0)
    g       = similar(x)
    h       = Hist()
    batches = 0
    opt     = Optimisers.setup(rule, x)
    h._t_start = time_ns()

    for i in 0:max_iter
        i % eval_freq == 0 && snap!(h, i, batches, x, name)
        i == max_iter && break
        objgrad!(nlp, x, g)
        opt, x = Optimisers.update(opt, x, g)
        minibatch_next_train!(nlp)
        batches += 1
    end
    return x, h
end

# ----------------------------------------------------------------
# 4b.  Runner: Tadam
#
#  Key differences from Adam loop:
#    - `minibatch_next_train!` only advances on *accepted* steps
#      (matches the STORM-like analysis in the Tadam paper).
#    - `stats.objective` is refreshed with the new mini-batch after
#      each advance so the ratio ρ_k is computed consistently.
#    - Stall guard: if solver reports :small_step but ‖g‖ is not
#      negligible, reset Δ and continue instead of stopping early.
# ----------------------------------------------------------------
function run_tadam!(; max_iter = 1000, eval_freq = 25, η1 = 0.10f0, kwargs...)
    @info "=== Tadam ==="
    nlp     = make_nlp()
    h       = Hist()
    batches = Ref(0)
    h._t_start = time_ns()

    cb = (nlp, solver, stats) -> begin
        iter = stats.iter

        # ── high-frequency diagnostics ────────────────────────────
        push!(h.hf_iters, iter)
        push!(h.hf_delta, Float64(solver.Δ))
        total = h.n_ok + h.n_rej
        push!(h.hf_rej, total > 0 ? h.n_rej / total : 0.0)

        # ── periodic evaluation ───────────────────────────────────
        iter % eval_freq == 0 && snap!(h, iter, batches[], solver.x, "Tadam")

        # ── step acceptance / batch advance ───────────────────────
        if solver.step_accepted
            h.n_ok += 1
            # Pause timer while we advance the batch
            h._t_accum += (time_ns() - h._t_start) / 1e9
            minibatch_next_train!(nlp)
            batches[] += 1
            stats.objective = obj(nlp, solver.x)
            h._t_start = time_ns()
        elseif iter > 0
            h.n_rej += 1
        end

        # ── stall guard ───────────────────────────────────────────
        if stats.status == :small_step
            ng        = norm(solver.gx)
            solver.Δ  = max(ng / (2^round(log2(ng + 1f0))), 1f-5)
            stats.status = :unknown
        end
    end

    stats = tadam(nlp; max_iter, atol = 1f-5, rtol = 1f-5,
                  callback = cb, verbose = 0, η1, kwargs...)

    # Final snap 
    (isempty(h.iters) || h.iters[end] != stats.iter) &&
        snap!(h, stats.iter, batches[], stats.solution, "Tadam")

    @printf "  Tadam final: %d accepted | %d rejected | %.1f%% rejection rate\n"  h.n_ok h.n_rej (100 * h.n_rej / max(1, h.n_ok + h.n_rej))
    return stats, h
end

# ----------------------------------------------------------------
# 5.  Run all three methods
#
#   Tadam hyper-parameter guide (names match Section 2 of the paper):
#     η1, η2  — acceptance thresholds for ρ_k  (η1 < η2 ≤ 1)
#     γ1      — TR radius contraction on rejection (0 < γ1 < 1)
#     γ2      — TR radius expansion on very good step (γ2 ≥ 1)
#     β1      — first-moment  decay (β_mom in paper)
#     β2      — second-moment decay (β_rms in paper)
#     ϵ_v     — ε_Adam numerical stabiliser
#     θ2      — Cauchy-decrease fraction (Assumption 4)
# ----------------------------------------------------------------
const MAX_ITER  = 1000
const EVAL_FREQ = 25
const LR        = 3f-4

_, h_adam    = run_first_order!(
    Optimisers.Adam(LR),    "Adam";
    max_iter = MAX_ITER, eval_freq = EVAL_FREQ,
)
_, h_amsgrad = run_first_order!(
    Optimisers.AMSGrad(LR), "AMSGrad";
    max_iter = MAX_ITER, eval_freq = EVAL_FREQ,
)
_, h_tadam   = let
    stats, h = run_tadam!(;
        max_iter  = MAX_ITER,
        eval_freq = EVAL_FREQ,
        # TR acceptance thresholds
        η1  = 1f-4,  
        η2  = 0.90f0,
        # Gentler TR radius updates (prevents 50% ping-ponging)
        γ1  = 0.80f0, 
        γ2  = 1.20f0, 
        # Adam momentum
        β1  = 0.90f0,
        β2  = 0.99f0,
        ϵ_v = 1f-7,
        # Deep Learning specific overrides
        θ1   = 1f-6,   #  
        Δmax = 1f-2,
    )
    stats, h
end

# ----------------------------------------------------------------
# 6.  Figure (6 panels, colorblind-safe, PDF + PNG)
# ----------------------------------------------------------------
const C_ADAM    = RGBf(0.902, 0.624, 0.000)   # orange
const C_AMSGRAD = RGBf(0.835, 0.369, 0.000)   # vermilion
const C_TADAM   = RGBf(0.000, 0.447, 0.698)   # blue

pub_theme = Theme(
    fontsize = 14,
    Axis = (
        spinewidth     = 0.9,
        xgridcolor     = (:black, 0.08),
        ygridcolor     = (:black, 0.08),
        xgridwidth     = 0.6,
        ygridwidth     = 0.6,
        titlesize      = 13,
        xlabelsize     = 12,
        ylabelsize     = 12,
        xticklabelsize = 11,
        yticklabelsize = 11,
    ),
    Legend = (framevisible = false, labelsize = 11, patchsize = (22, 2)),
    Lines  = (linewidth = 2.3,),
)

with_theme(pub_theme) do
    fig = Figure(size = (900, 940))

    function add_curves!(ax, xfield, yfield)
        for (h, label, color) in [
            (h_adam,    "Adam",    C_ADAM),
            (h_amsgrad, "AMSGrad", C_AMSGRAD),
            (h_tadam,   "Tadam",  C_TADAM),
        ]
            lines!(ax, getfield(h, xfield), getfield(h, yfield); color, label)
        end
    end

    # (a-b) Train loss
    ax_a = Axis(fig[1,1];
        xlabel = "Minibatches (gradient evaluations)",
        ylabel = "Train loss", yscale = log10,
        title  = "(a) Train loss vs. gradient evaluations")
    ax_b = Axis(fig[1,2];
        xlabel = "Wall-clock time (s)",
        ylabel = "Train loss", yscale = log10,
        title  = "(b) Train loss vs. time")
    add_curves!(ax_a, :batches, :tr_loss); axislegend(ax_a; position = :rt)
    add_curves!(ax_b, :times,   :tr_loss)

    # (c-d) Test accuracy
    ax_c = Axis(fig[2,1];
        xlabel = "Minibatches (gradient evaluations)",
        ylabel = "Test accuracy",
        title  = "(c) Test accuracy vs. gradient evaluations")
    ax_d = Axis(fig[2,2];
        xlabel = "Wall-clock time (s)",
        ylabel = "Test accuracy",
        title  = "(d) Test accuracy vs. time")
    add_curves!(ax_c, :batches, :te_acc); axislegend(ax_c; position = :rb)
    add_curves!(ax_d, :times,   :te_acc)

    # (e) Trust-region radius
    ax_e = Axis(fig[3,1];
        xlabel = "Iteration k",
        ylabel = "Trust-region radius Δk", yscale = log10,
        title  = "(e) Adaptive step size: Δk over iterations (Tadam)")
    lines!(ax_e, h_tadam.hf_iters, h_tadam.hf_delta;
           color = C_TADAM, linewidth = 1.4)

    # (f) Rejection rate
    ax_f = Axis(fig[3,2];
        xlabel = "Iteration k",
        ylabel = "Cumulative rejection rate",
        limits = (nothing, (0.0, 1.0)),
        title  = "(f) TR step rejection rate over training (Tadam)")
    lines!(ax_f, h_tadam.hf_iters, h_tadam.hf_rej;
           color = C_TADAM, linewidth = 2.0)

    Label(fig[4, :],
        "Fashion-MNIST  (N_train=$(N_TRAIN), N_test=$(N_TEST), batch=$(BATCH)).  " *
        "MLP: 784 → 512 → 256 → 10.  Adam & AMSGrad use lr=$(LR); " *
        "Tadam adapts Δ automatically.\n" *
        "Panels (e–f): Tadam internal diagnostics showing self-tuning behaviour.",
        tellwidth = false, fontsize = 11, color = :gray40,
    )

    save("fashion_mnist_Tadam_results.pdf", fig)
    save("fashion_mnist_Tadam_results.png", fig; px_per_unit = 2)
    display(fig)
    @info "Figures written: fashion_mnist_Tadam_results.{pdf,png}"
end