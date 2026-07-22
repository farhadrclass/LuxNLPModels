# ================================================================
# cifar_10_CNN_v2.jl
# CIFAR-10 CNN Benchmark on GPU: Adam | AMSGrad | Tadam
#
#   - GPU-first (CUDA/LuxCUDA); falls back to CPU if CUDA unavailable.
#   - Small VGG-like CNN: 3 → 32 → 32 → 64 → 64, 2× MaxPool.
#   - Full CIFAR-10: 50 000 train / 10 000 test (toggle subset below).
#   - Batched evaluation on GPU to avoid OOM.
#
# Recommended environment: lux_test_env_4  (has Lux, LuxCUDA, Optimisers,
# JSOSolvers, CairoMakie, MLDatasets, etc.).
# ================================================================

# using Pkg
# Pkg.activate("lux_test_env_4")

using Lux
using LuxCUDA
using CUDA
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
# 0.  Device selection
# ----------------------------------------------------------------
if CUDA.functional()
    @info "CUDA detected; training on GPU."
    const dev = gpu_device()
else
    @info "CUDA not available; falling back to CPU."
    const dev = cpu_device()
end

# ----------------------------------------------------------------
# 1.  CIFAR-10  (32 × 32, RGB, 10 classes)
# ----------------------------------------------------------------
@info "Loading CIFAR-10..."
tr_x_raw, tr_y_raw = CIFAR10.traindata(Float32)   # → (32, 32, 3, 50_000)
te_x_raw, te_y_raw = CIFAR10.testdata(Float32)    # → (32, 32, 3, 10_000)

# Toggle for quick CPU/GPU smoke tests.
const USE_SUBSET = false
const N_TRAIN    = USE_SUBSET ? 10_000 : 50_000
const N_TEST     = USE_SUBSET ? 2_000  : 10_000

# Sanity check: CIFAR-10 labels from MLDatasets are 0..9.
@assert minimum(tr_y_raw) == 0 && maximum(tr_y_raw) == 9 "Unexpected CIFAR-10 label range"

# Move data to the target device *before* batching so the DataLoader yields
# GPU arrays directly.  Labels are cast to Float32 for the cross-entropy.
train_x = tr_x_raw[:, :, :, 1:N_TRAIN] |> dev
train_y = Float32.(onehotbatch(tr_y_raw[1:N_TRAIN], 0:9)) |> dev
test_x  = te_x_raw[:, :, :, 1:N_TEST]  |> dev
test_y  = Float32.(onehotbatch(te_y_raw[1:N_TEST],  0:9)) |> dev

const BATCH = 128
const EVAL_BATCH = 1_000   # used only during full-dataset metric evaluation

# For reproducible shuffling across the three independent runs.
const DL_RNG = MersenneTwister(123)

data_loader = DataLoader(
    (train_x, train_y);
    batchsize = BATCH, shuffle = true, rng = DL_RNG,
)

# ----------------------------------------------------------------
# 2.  CNN model for CIFAR-10
#     32×32 → [Conv3×3, ReLU] ×2 → MaxPool →
#             [Conv3×3, ReLU] ×2 → MaxPool → Flatten → 4096 → 512 → 10
#
#  NOTE: LuxNLPModels discards the state returned by Lux.apply inside
#  obj/grad!/objgrad! (this keeps the solver interface side-effect-free).
#  Stateful layers such as BatchNorm or Dropout would therefore not update
#  their running statistics during training, so this architecture uses only
#  convolutional / dense / pooling layers.  You can still add BatchNorm if
#  you manage the state manually outside the solver loop.
# ----------------------------------------------------------------
model = Chain(
    Conv((3, 3), 3 => 32, relu; pad = (1, 1)),
    Conv((3, 3), 32 => 32, relu; pad = (1, 1)),
    MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, relu; pad = (1, 1)),
    Conv((3, 3), 64 => 64, relu; pad = (1, 1)),
    MaxPool((2, 2)),
    FlattenLayer(),
    Dense(64 * 8 * 8 => 512, relu),
    Dense(512 => 10),
)

rng_init = MersenneTwister(42)
ps_cpu, st = Lux.setup(rng_init, model)
st = st |> dev                               # model state lives on GPU

const ps_template = ComponentArray(ps_cpu)     # axis template (CPU)
const ps0         = copy(ps_template)          # shared starting point

loss_fn(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims = 1); dims = 1))

const st_test = Lux.testmode(st)               # no-op here, but kept for safety

# GPU-friendly accuracy: argmax on the GPU, comparison on the CPU.
function acc(ŷ, y)
    pred = Array(argmax(ŷ; dims = 1))
    true_lbl = Array(argmax(y; dims = 1))
    return mean(pred .== true_lbl)
end

# Evaluate on fixed train/test splits in batched GPU passes.
function eval_metrics(ps_vec)
    ps_s = ComponentArray(Array(ps_vec), getaxes(ps_template)) |> dev

    function eval_split(x, y)
        n = size(x, 4)
        total_loss = 0.0f0
        total      = 0
        correct    = 0
        for i in 1:EVAL_BATCH:n
            idx = i:min(i + EVAL_BATCH - 1, n)
            xi  = x[:, :, :, idx]
            yi  = y[:, idx]
            ŷi, _ = Lux.apply(model, xi, ps_s, st_test)
            m = length(idx)
            total_loss += loss_fn(ŷi, yi) * m
            total      += m
            correct    += sum(Array(argmax(ŷi; dims = 1)) .== Array(argmax(yi; dims = 1)))
        end
        avg_loss = total_loss / total
        avg_acc  = correct / total
        return Float32(avg_loss), Float32(avg_acc)
    end

    tr_loss, tr_acc = eval_split(train_x, train_y)
    te_loss, te_acc = eval_split(test_x,  test_y)
    return tr_loss, tr_acc, te_acc
end

make_nlp() = LuxNLPModel(model, copy(ps0), st, data_loader, loss_fn; dev)

@printf "Total parameters: %d\n" length(ps0)
@printf "Initial train batch loss: %.4f\n\n" obj(make_nlp(), ps0)

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
    h._t_accum += (time_ns() - h._t_start) / 1e9
    tr_loss, tr_acc, te_acc = eval_metrics(ps_vec)
    push!(h.iters,   iter);    push!(h.batches, batches)
    push!(h.times,   h._t_accum)
    push!(h.tr_loss, tr_loss); push!(h.tr_acc, tr_acc); push!(h.te_acc, te_acc)
    @printf "[%s] iter=%4d  bat=%4d  t=%5.1fs  loss=%.4f  tr=%.1f%%  te=%.1f%%\n"  tag iter batches h._t_accum tr_loss (100tr_acc) (100te_acc)
    h._t_start = time_ns()
end

# ----------------------------------------------------------------
# 4a.  Runner: first-order optimisers (Adam, AMSGrad)
# ----------------------------------------------------------------
function run_first_order!(rule, name; max_iter = 3000, eval_freq = 50)
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
        f, _ = objgrad!(nlp, x, g)
        # Stop early if loss becomes non-finite (useful on unstable runs).
        if !isfinite(f)
            @warn "$name encountered non-finite loss at iter $i; stopping early."
            break
        end
        opt, x = Optimisers.update(opt, x, g)
        minibatch_next_train!(nlp)
        batches += 1
    end
    return x, h
end

# ----------------------------------------------------------------
# 4b.  Runner: Tadam
# ----------------------------------------------------------------
function run_tadam!(; max_iter = 3000, eval_freq = 50, η1 = 0.10f0, kwargs...)
    @info "=== Tadam ==="
    nlp     = make_nlp()
    h       = Hist()
    batches = Ref(0)
    h._t_start = time_ns()

    cb = (nlp, solver, stats) -> begin
        iter = stats.iter

        # High-frequency diagnostics.
        push!(h.hf_iters, iter)
        push!(h.hf_delta, Float64(solver.Δ))
        total = h.n_ok + h.n_rej
        push!(h.hf_rej, total > 0 ? h.n_rej / total : 0.0)

        # Periodic evaluation.
        iter % eval_freq == 0 && snap!(h, iter, batches[], solver.x, "Tadam")

        # Step acceptance / batch advance.
        if solver.step_accepted
            h.n_ok += 1
            h._t_accum += (time_ns() - h._t_start) / 1e9
            minibatch_next_train!(nlp)
            batches[] += 1
            stats.objective = obj(nlp, solver.x)
            h._t_start = time_ns()
        elseif iter > 0
            h.n_rej += 1
        end

        # Stall guard.
        if stats.status == :small_step
            ng        = norm(solver.gx)
            solver.Δ  = max(ng / (2^round(log2(ng + 1f0))), 1f-5)
            stats.status = :unknown
        end
    end

    stats = tadam(nlp; max_iter, atol = 1f-5, rtol = 1f-5,
                  callback = cb, verbose = 0, η1, kwargs...)

    if !isfinite(stats.objective)
        @warn "Tadam final objective is non-finite; run may have diverged."
    end

    # Final snap.
    (isempty(h.iters) || h.iters[end] != stats.iter) &&
        snap!(h, stats.iter, batches[], stats.solution, "Tadam")

    @printf "  Tadam final: %d accepted | %d rejected | %.1f%% rejection rate\n"  h.n_ok h.n_rej (100 * h.n_rej / max(1, h.n_ok + h.n_rej))
    return stats, h
end

# ----------------------------------------------------------------
# 5.  Run all three methods
# ----------------------------------------------------------------
const MAX_ITER  = 3_000
const EVAL_FREQ = 50
const LR        = 1f-3

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
        # TR radius updates
        γ1  = 0.80f0,
        γ2  = 1.20f0,
        γ3  = 0.02f0,
        # Adam momentum
        β1  = 0.90f0,
        β2  = 0.99f0,
        ϵ_v = 1f-7,
        # Deep-learning overrides
        θ1   = 1f-6,
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
            (h_tadam,   "Tadam",   C_TADAM),
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
        "CIFAR-10  (N_train=$(N_TRAIN), N_test=$(N_TEST), batch=$(BATCH)).  " *
        "CNN: Conv 3→32→32→64→64, MaxPool×2, 4096→512→10.  Adam & AMSGrad use lr=$(LR); " *
        "Tadam adapts Δ automatically.\n" *
        "Panels (e–f): Tadam internal diagnostics showing self-tuning behaviour.",
        tellwidth = false, fontsize = 11, color = :gray40,
    )

    save("cifar10_cnn_Tadam_results.pdf", fig)
    save("cifar10_cnn_Tadam_results.png", fig; px_per_unit = 2)
    display(fig)
    @info "Figures written: cifar10_cnn_Tadam_results.{pdf,png}"
end
