# ================================================================
# cifar_10_CNN_v2.jl
# CIFAR-10 CNN Benchmark on GPU: Adam | AMSGrad | Tadam
#
#   - GPU-first (CUDA/LuxCUDA); falls back to CPU if CUDA unavailable.
#   - Small VGG-like CNN: 3 → 32 → 32 → 64 → 64, 2× MaxPool.
#   - Full CIFAR-10: 50 000 train / 10 000 test (toggle subset below).
#   - GPU-sized mini-batches and bounded evaluation work.
#   - Elapsed time includes metric evaluation and device synchronization.
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
using Zygote
using CairoMakie
import CairoMakie: Axis

# ----------------------------------------------------------------
# 0.  Device selection
# ----------------------------------------------------------------
if CUDA.functional()
    @info "CUDA detected; training on GPU."
    const dev = gpu_device()
    # Fail loudly on accidental scalar indexing of GPU arrays, which otherwise
    # silently degrades performance to a crawl instead of erroring.
    CUDA.allowscalar(false)
else
    @warn "CUDA NOT available; falling back to CPU. A batch-1024 CNN is extremely slow on CPU — keep SMOKE_TEST=true (below) for a quick sanity run."
    const dev = cpu_device()
end

# Fast end-to-end sanity run: tiny data, small batch, few iterations. Once the
# one-time JIT compilation finishes, the whole script completes in seconds.
# Flip to false for the full 50k-image benchmark.
const SMOKE_TEST = true

# ----------------------------------------------------------------
# 1.  CIFAR-10  (32 × 32, RGB, 10 classes)
# ----------------------------------------------------------------
@info "Loading CIFAR-10..."
tr_x_raw, tr_y_raw = CIFAR10(split = :train)[:]   # → (32, 32, 3, 50_000)
te_x_raw, te_y_raw = CIFAR10(split = :test)[:]    # → (32, 32, 3, 10_000)

# Normalization is explicit because MLDatasets versions can expose stored
# CIFAR pixels either as 0:255 values or as already scaled Float32 values.
const CIFAR_MEAN = reshape(Float32[0.4914, 0.4822, 0.4465], 1, 1, 3, 1)
const CIFAR_STD  = reshape(Float32[0.2470, 0.2435, 0.2616], 1, 1, 3, 1)
function normalize_cifar10(images)
    normalized = Float32.(images)
    maximum(normalized) > 1f0 && (normalized ./= 255f0)
    normalized .-= CIFAR_MEAN
    normalized ./= CIFAR_STD
    return normalized
end
tr_x_raw = normalize_cifar10(tr_x_raw)
te_x_raw = normalize_cifar10(te_x_raw)

# Toggle for quick CPU/GPU smoke tests (driven by SMOKE_TEST).
const USE_SUBSET = SMOKE_TEST
const N_TRAIN    = SMOKE_TEST ? 4_000 : 50_000
const N_TEST     = SMOKE_TEST ? 2_000 : 10_000

# Sanity check: CIFAR-10 labels from MLDatasets are 0..9.
@assert minimum(tr_y_raw) == 0 && maximum(tr_y_raw) == 9 "Unexpected CIFAR-10 label range"

# Move data to the target device *before* batching so the DataLoader yields
# GPU arrays directly.  Labels are cast to Float32 for the cross-entropy.
train_x = tr_x_raw[:, :, :, 1:N_TRAIN] |> dev
train_y = Float32.(onehotbatch(tr_y_raw[1:N_TRAIN], 0:9)) |> dev
test_x  = te_x_raw[:, :, :, 1:N_TEST]  |> dev
test_y  = Float32.(onehotbatch(te_y_raw[1:N_TEST],  0:9)) |> dev

# A large batch keeps big data-center GPUs busy but is punishing on CPU. 1024 is
# a good default for this compact CNN on a high-memory accelerator; SMOKE_TEST
# drops it to 128 so a first run finishes quickly. Reduce further only on OOM.
const BATCH = SMOKE_TEST ? 128 : 1_024
const EVAL_BATCH = SMOKE_TEST ? 512 : 2_048

# Each snapshot evaluates a representative, fixed training prefix and the full
# test set.  Evaluating all 50k training images every few updates dominates the
# wall time while adding little information to a convergence comparison.
const METRIC_TRAIN_SAMPLES = USE_SUBSET ? N_TRAIN : 10_000

# Each method receives the same shuffled mini-batch sequence.  A fresh loader
# is required because DataLoader advances the supplied RNG while iterating.
make_data_loader() = DataLoader(
    (train_x, train_y);
    batchsize = BATCH, shuffle = true, rng = MersenneTwister(123),
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
const model = Chain(
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
ps_cpu, st_raw = Lux.setup(rng_init, model)
const st = st_raw |> dev                     # model state on `dev`; const → type-stable in hot loops

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

# Evaluate a fixed training prefix and the full test split in batched GPU passes.
function eval_metrics(ps_vec)
    ps_s = ComponentArray(Array(getdata(ps_vec)), getaxes(ps_template)) |> dev

    function eval_split(x, y, n)
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

    tr_loss, tr_acc = eval_split(train_x, train_y, METRIC_TRAIN_SAMPLES)
    te_loss, te_acc = eval_split(test_x,  test_y, N_TEST)
    return tr_loss, tr_acc, te_acc
end

make_nlp() = LuxNLPModel(model, copy(ps0), st, make_data_loader(), loss_fn; dev)

@printf "Total parameters: %d\n" length(ps0)
initial_nlp = make_nlp()
CUDA.functional() && @assert initial_nlp.meta.x0 isa CuArray "GPU solver vector was not created on CUDA"
@printf "Initial train batch loss: %.4f\n\n" obj(initial_nlp, initial_nlp.meta.x0)

# One-time JIT compilation of every hot path *before* the timed runs, so the
# per-method timers and the first printed iteration are not dominated by
# compilation. On the very first run this block alone can take a minute or two
# (Zygote + CUDA kernel compilation); afterwards it is fast.
@info "Warming up: compiling forward / backward / update + evaluation paths (one-time)…"
let
    t0 = time_ns()
    warmup_nlp = make_nlp()
    warmup_ps  = ComponentArray(ps0) |> dev
    wx, wy     = warmup_nlp.current_batch
    _, wback = Zygote.pullback(warmup_ps) do p
        ŷ, _ = Lux.apply(model, wx, p, st)
        loss_fn(ŷ, wy)
    end
    wg   = wback(one(Float32))[1]
    wopt = Optimisers.setup(Optimisers.Adam(1f-3), warmup_ps)
    Optimisers.update(wopt, warmup_ps, wg)
    eval_metrics(warmup_ps)                  # also compile the batched inference path
    CUDA.functional() && CUDA.synchronize()
    @info @sprintf("Warm-up done in %.1fs.", (time_ns() - t0) / 1e9)
end

# ----------------------------------------------------------------
# 3.  History struct
# ----------------------------------------------------------------
mutable struct Hist
    # Low-frequency evaluation snapshots
    iters     :: Vector{Int}
    grad_evals :: Vector{Int}   # cumulative gradient (backward-pass) evaluations
    times     :: Vector{Float64}
    tr_loss :: Vector{Float32}
    tr_acc  :: Vector{Float32}
    te_acc  :: Vector{Float32}
    # Tadam high-frequency diagnostics
    hf_iters :: Vector{Int}
    hf_delta :: Vector{Float64}   # trust-region radius Δ_k
    hf_rej   :: Vector{Float64}   # running rejection rate
    n_ok     :: Int
    n_rej    :: Int
    # Elapsed wall-clock timer, including evaluation and GPU synchronization
    _t_accum :: Float64
    _t_start :: UInt64
end

Hist() = Hist(
    Int[], Int[], Float64[], Float32[], Float32[], Float32[],
    Int[], Float64[], Float64[], 0, 0, 0.0, time_ns(),
)

function snap!(h::Hist, iter, grad_evals, ps_vec, tag)
    tr_loss, tr_acc, te_acc = eval_metrics(ps_vec)
    CUDA.functional() && CUDA.synchronize()
    h._t_accum = (time_ns() - h._t_start) / 1e9
    push!(h.iters,     iter);    push!(h.grad_evals, grad_evals)
    push!(h.times,     h._t_accum)
    push!(h.tr_loss, tr_loss); push!(h.tr_acc, tr_acc); push!(h.te_acc, te_acc)
    @printf "[%s] iter=%4d  ge=%5d  t=%5.1fs  loss=%.4f  tr=%.1f%%  te=%.1f%%\n"  tag iter grad_evals h._t_accum tr_loss (100tr_acc) (100te_acc)
end

# ----------------------------------------------------------------
# 4a.  Runner: native GPU first-order optimisers (Adam, AMSGrad)
#
# Tadam uses LuxNLPModel and therefore has CPU-resident solver vectors.
# Adam/AMSGrad do not need that interface: keeping their parameters, gradients,
# and Optimisers state on `dev` avoids a full CPU↔GPU transfer every update.
# ----------------------------------------------------------------
function run_first_order_gpu!(rule, name; max_iter = 3000, eval_freq = 50)
    @info "=== $name ==="
    nlp        = make_nlp()
    ps         = ComponentArray(ps0) |> dev
    h          = Hist()
    grad_evals = 0                       # one backward pass (pullback) per update
    opt        = Optimisers.setup(rule, ps)
    h._t_start = time_ns()

    for i in 0:max_iter
        i % eval_freq == 0 && snap!(h, i, grad_evals, ps, name)
        i == max_iter && break

        x_batch, y_batch = nlp.current_batch
        _, back = Zygote.pullback(ps) do p
            ŷ, _ = Lux.apply(model, x_batch, p, st)
            loss_fn(ŷ, y_batch)
        end
        g = back(one(Float32))[1]
        isnothing(g) && error("$name produced no parameter gradient")
        opt, ps = Optimisers.update(opt, ps, g)
        minibatch_next_train!(nlp)
        grad_evals += 1
    end
    return ps, h
end

# ----------------------------------------------------------------
# 4b.  Runner: Tadam
# ----------------------------------------------------------------
function run_tadam!(; max_iter = 3000, eval_freq = 50, η1 = 0.10f0, kwargs...)
    @info "=== Tadam ==="
    nlp     = make_nlp()
    h       = Hist()
    h._t_start = time_ns()

    # Honest gradient-evaluation count for the x-axis: every objgrad!/grad!
    # (tracked by neval_grad) plus the two finite-difference gradients inside
    # each hprod! (see src/hessian.jl), which are deliberately NOT tallied in
    # neval_grad. This makes the sample-efficiency comparison against Adam /
    # AMSGrad fair, since one of their updates is exactly one gradient eval and
    # rejected Tadam steps still cost gradient work.
    grad_evals() = neval_grad(nlp) + 2 * neval_hprod(nlp)

    cb = (nlp, solver, stats) -> begin
        iter = stats.iter

        # High-frequency diagnostics.
        push!(h.hf_iters, iter)
        push!(h.hf_delta, Float64(solver.Δ))
        total = h.n_ok + h.n_rej
        push!(h.hf_rej, total > 0 ? h.n_rej / total : 0.0)

        # Periodic evaluation.
        iter % eval_freq == 0 && snap!(h, iter, grad_evals(), solver.x, "Tadam")

        # Step acceptance / batch advance.
        if solver.step_accepted
            h.n_ok += 1
            minibatch_next_train!(nlp)
            stats.objective = obj(nlp, solver.x)
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
        snap!(h, stats.iter, grad_evals(), stats.solution, "Tadam")

    @printf "  Tadam final: %d accepted | %d rejected | %.1f%% rejection rate\n"  h.n_ok h.n_rej (100 * h.n_rej / max(1, h.n_ok + h.n_rej))
    return stats, h
end

# ----------------------------------------------------------------
# 5.  Run all three methods
# ----------------------------------------------------------------
const MAX_ITER  = SMOKE_TEST ? 100 : 3_000
const EVAL_FREQ = SMOKE_TEST ?  20 : 250
const LR        = 1f-3

_, h_adam    = run_first_order_gpu!(
    Optimisers.Adam(LR),    "Adam";
    max_iter = MAX_ITER, eval_freq = EVAL_FREQ,
)
_, h_amsgrad = run_first_order_gpu!(
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
        xlabel = "Gradient evaluations",
        ylabel = "Train loss", yscale = log10,
        title  = "(a) Train loss vs. gradient evaluations")
    ax_b = Axis(fig[1,2];
        xlabel = "Wall-clock time (s)",
        ylabel = "Train loss", yscale = log10,
        title  = "(b) Train loss vs. time")
    add_curves!(ax_a, :grad_evals, :tr_loss); axislegend(ax_a; position = :rt)
    add_curves!(ax_b, :times,   :tr_loss)

    # (c-d) Test accuracy
    ax_c = Axis(fig[2,1];
        xlabel = "Gradient evaluations",
        ylabel = "Test accuracy",
        title  = "(c) Test accuracy vs. gradient evaluations")
    ax_d = Axis(fig[2,2];
        xlabel = "Wall-clock time (s)",
        ylabel = "Test accuracy",
        title  = "(d) Test accuracy vs. time")
    add_curves!(ax_c, :grad_evals, :te_acc); axislegend(ax_c; position = :rb)
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
        "CIFAR-10  (N_train=$(N_TRAIN), N_test=$(N_TEST), batch=$(BATCH), metric train=$(METRIC_TRAIN_SAMPLES)).  " *
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
