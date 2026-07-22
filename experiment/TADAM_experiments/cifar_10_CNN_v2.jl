# ================================================================
# cifar10_gpu_compare.jl
# CIFAR-10 Benchmark: Adam | AMSGrad | Tadam  (GPU-accelerated)
#
# Architecture: CNN  (3→32→64→128 conv + FC head)
#   Input : 32 × 32 × 3
#   Body  : Conv(3×3,3→32,relu) → MaxPool(2)
#           Conv(3×3,32→64,relu) → MaxPool(2)
#           Conv(3×3,64→128,relu) → AdaptiveMeanPool(4)
#   Head  : Dense(128*4*4 → 256, relu) → Dense(256 → 10)
#
# GPU notes:
#   - All data + parameters moved to GPU with `gpu_device()`
#   - LuxNLPModel wraps the GPU model transparently
#   - Eval is done on GPU; metrics transferred back with `Array()`
#   - Falls back to CPU automatically if no CUDA device is found
#
# Tested with:
#   Lux 1.x, CUDA 5.x, NNlib 0.9.x, ComponentArrays 0.15.x
#   JSOSolvers (Tadam branch), LuxNLPModels 0.x
# ================================================================

# using Pkg
# Pkg.activate("lux_gpu_env")

using Lux
using LuxCUDA          # exports `gpu_device`, `cpu_device`, CUDA backend
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
const DEV = gpu_device()          # CUDADevice() if GPU is available, else CPUDevice()
const CPU = cpu_device()

@info "Running on: $DEV"

# ----------------------------------------------------------------
# 1.  CIFAR-10  (32 × 32 RGB, 10 classes)
# ----------------------------------------------------------------
@info "Loading CIFAR-10..."
tr_x_raw, tr_y_raw = CIFAR10.traindata(Float32)   # (32,32,3,50_000)
te_x_raw, te_y_raw = CIFAR10.testdata(Float32)    # (32,32,3,10_000)

N_TRAIN, N_TEST = 20_000, 4_000   # increase if your GPU has ≥16 GB VRAM

# Pixel values come in [0,1]; channel-wise normalisation improves convergence.
# CIFAR-10 channel means / stds (precomputed on the full training set).
const CIFAR_MEAN = Float32[0.4914, 0.4822, 0.4465]
const CIFAR_STD  = Float32[0.2470, 0.2435, 0.2616]

function normalise_cifar(x::Array{Float32,4})
    out = similar(x)
    for c in 1:3
        out[:, :, c, :] = (x[:, :, c, :] .- CIFAR_MEAN[c]) ./ CIFAR_STD[c]
    end
    return out
end

# Prepare on CPU, then move to device
train_x_cpu = normalise_cifar(tr_x_raw[:, :, :, 1:N_TRAIN])
train_y_cpu = Float32.(onehotbatch(tr_y_raw[1:N_TRAIN], 0:9))
test_x_cpu  = normalise_cifar(te_x_raw[:, :, :, 1:N_TEST])
test_y_cpu  = Float32.(onehotbatch(te_y_raw[1:N_TEST], 0:9))

# GPU copies (used for evaluation)
const train_x = DEV(train_x_cpu)
const train_y = DEV(train_y_cpu)
const test_x  = DEV(test_x_cpu)
const test_y  = DEV(test_y_cpu)

const BATCH = 256   # 256–512 works well on a 16 GB GPU

# DataLoader stays on CPU; each batch is moved to GPU inside LuxNLPModel.
# If you have enough VRAM you can pass `DEV(train_x_cpu)` here directly.
data_loader = DataLoader(
    (train_x_cpu, train_y_cpu);
    batchsize = BATCH, shuffle = true,
)

# ----------------------------------------------------------------
# 2.  CNN model
# ----------------------------------------------------------------
model = Chain(
    # Block 1
    Conv((3, 3), 3  => 32,  relu; pad = SamePad()),
    BatchNorm(32),
    MaxPool((2, 2)),
    # Block 2
    Conv((3, 3), 32 => 64,  relu; pad = SamePad()),
    BatchNorm(64),
    MaxPool((2, 2)),
    # Block 3
    Conv((3, 3), 64 => 128, relu; pad = SamePad()),
    BatchNorm(128),
    AdaptiveMeanPool((4, 4)),   # → (4,4,128,N)
    # Classifier head
    FlattenLayer(),
    Dense(128 * 4 * 4 => 256, relu),
    Dropout(0.3f0),
    Dense(256 => 10),
)

rng_init = MersenneTwister(42)
ps_cpu, st = Lux.setup(rng_init, model)

# Move parameters & state to GPU
ps_dev = DEV(ps_cpu)
st_dev = DEV(st)

const ps_template = ComponentArray(ps_cpu)     # CPU axis template
const ps0         = copy(ps_template)           # shared starting point

loss_fn(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims = 1); dims = 1))

# ----------------------------------------------------------------
# 3.  Evaluation (runs on GPU, pulls scalars to CPU)
# ----------------------------------------------------------------
function eval_metrics(ps_vec)
    # ps_vec lives on GPU (or CPU in fallback mode)
    ps_gpu = ComponentArray(Array(ps_vec), getaxes(ps_template)) |> DEV

    ŷ_tr, _ = Lux.apply(model, train_x, ps_gpu, st_dev)
    ŷ_te, _ = Lux.apply(model, test_x,  ps_gpu, st_dev)

    tr_loss = Float32(loss_fn(ŷ_tr, train_y))

    # Move predictions to CPU for argmax comparison
    ŷ_tr_c = Array(ŷ_tr);  y_tr_c = Array(train_y)
    ŷ_te_c = Array(ŷ_te);  y_te_c = Array(test_y)

    acc(ŷ, y) = mean(
        [i.I[1] for i in argmax(ŷ; dims = 1)] .==
        [i.I[1] for i in argmax(y;  dims = 1)],
    )
    Float32(tr_loss),
    Float32(acc(ŷ_tr_c, y_tr_c)),
    Float32(acc(ŷ_te_c, y_te_c))
end

# LuxNLPModel moves each mini-batch to the device internally when
# the model's parameters are on the GPU.
make_nlp() = LuxNLPModel(model, DEV(copy(ps0)), st_dev, data_loader, loss_fn)

# ----------------------------------------------------------------
# 4.  History struct  (identical to Fashion-MNIST version)
# ----------------------------------------------------------------
mutable struct Hist
    iters    :: Vector{Int}
    batches  :: Vector{Int}
    times    :: Vector{Float64}
    tr_loss  :: Vector{Float32}
    tr_acc   :: Vector{Float32}
    te_acc   :: Vector{Float32}
    gnorm    :: Vector{Float32}
    hf_iters :: Vector{Int}
    hf_delta :: Vector{Float64}
    hf_rej   :: Vector{Float64}
    n_ok     :: Int
    n_rej    :: Int
    _t_accum :: Float64
    _t_start :: UInt64
end

Hist() = Hist(
    Int[], Int[], Float64[], Float32[], Float32[], Float32[], Float32[],
    Int[], Float64[], Float64[], 0, 0, 0.0, time_ns(),
)

function snap!(h::Hist, iter, batches, ps_vec, tag, gn)
    h._t_accum += (time_ns() - h._t_start) / 1e9
    tr_loss, tr_acc, te_acc = eval_metrics(ps_vec)
    push!(h.iters,   iter);    push!(h.batches, batches)
    push!(h.times,   h._t_accum)
    push!(h.tr_loss, tr_loss); push!(h.tr_acc, tr_acc); push!(h.te_acc, te_acc)
    push!(h.gnorm,   Float32(gn))
    @printf "[%s] iter=%4d  bat=%4d  t=%6.1fs  loss=%.4f  tr=%.1f%%  te=%.1f%%  |g|=%.4f\n" tag iter batches h._t_accum tr_loss (100tr_acc) (100te_acc) gn
    h._t_start = time_ns()
end

# ----------------------------------------------------------------
# 5a.  First-order runner (Adam / AMSGrad)
# ----------------------------------------------------------------
function run_first_order!(rule, name; max_iter = 2000, eval_freq = 50)
    @info "=== $name ==="
    nlp     = make_nlp()
    x       = copy(nlp.meta.x0)   # GPU vector
    g       = similar(x)
    h       = Hist()
    batches = 0
    opt     = Optimisers.setup(rule, x)
    h._t_start = time_ns()

    for i in 0:max_iter
        objgrad!(nlp, x, g)
        gn = norm(g)

        i % eval_freq == 0 && snap!(h, i, batches, x, name, gn)
        i == max_iter && break

        opt, x = Optimisers.update(opt, x, g)
        minibatch_next_train!(nlp)
        batches += 1
    end
    return x, h
end

# ----------------------------------------------------------------
# 5b.  Tadam runner
# ----------------------------------------------------------------
function run_tadam!(; max_iter = 2000, eval_freq = 50, η1 = 1f-4, kwargs...)
    @info "=== Tadam ==="
    nlp     = make_nlp()
    h       = Hist()
    batches = Ref(0)
    h._t_start = time_ns()

    cb = (nlp, solver, stats) -> begin
        iter = stats.iter

        push!(h.hf_iters, iter)
        push!(h.hf_delta, Float64(solver.Δ))
        total = h.n_ok + h.n_rej
        push!(h.hf_rej, total > 0 ? h.n_rej / total : 0.0)

        iter % eval_freq == 0 &&
            snap!(h, iter, batches[], solver.x, "Tadam", norm(solver.gx))

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

        if stats.status == :small_step
            ng       = norm(solver.gx)
            solver.Δ = max(ng / (2^round(log2(ng + 1f0))), 1f-5)
            stats.status = :unknown
        end
    end

    stats = tadam(nlp; max_iter, atol = 1f-5, rtol = 1f-5,
                  callback = cb, verbose = 0, η1, kwargs...)

    if isempty(h.iters) || h.iters[end] != stats.iter
        g_final = similar(stats.solution)
        objgrad!(nlp, stats.solution, g_final)
        snap!(h, stats.iter, batches[], stats.solution, "Tadam", norm(g_final))
    end

    @printf "  Tadam final: %d accepted | %d rejected | %.1f%% rejection rate\n"  h.n_ok h.n_rej (100 * h.n_rej / max(1, h.n_ok + h.n_rej))
    return stats, h
end

# ----------------------------------------------------------------
# 6.  Run all three optimisers
# ----------------------------------------------------------------
const MAX_ITER  = 2000    # more iterations than MNIST; CIFAR is harder
const EVAL_FREQ = 50
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
        η1   = 1f-4,
        η2   = 0.90f0,
        γ1   = 0.80f0,
        γ2   = 1.20f0,
        γ3   = 0.02f0,
        β1   = 0.90f0,
        β2   = 0.99f0,
        ϵ_v  = 1f-7,
        θ1   = 1f-6,
        Δmax = 1f-2,
    )
    stats, h
end

# ----------------------------------------------------------------
# 7.  Figures  (same 7-panel layout as Fashion-MNIST script)
# ----------------------------------------------------------------
const C_ADAM    = RGBf(0.902, 0.624, 0.000)
const C_AMSGRAD = RGBf(0.835, 0.369, 0.000)
const C_TADAM   = RGBf(0.000, 0.447, 0.698)

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
    fig = Figure(size = (900, 1150))

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
    ax_a = Axis(fig[1, 1];
        xlabel = "Minibatches (gradient evaluations)",
        ylabel = "Train loss", yscale = log10,
        title  = "(a) Train loss vs. gradient evaluations")
    ax_b = Axis(fig[1, 2];
        xlabel = "Wall-clock time (s)",
        ylabel = "Train loss", yscale = log10,
        title  = "(b) Train loss vs. time")
    add_curves!(ax_a, :batches, :tr_loss); axislegend(ax_a; position = :rt)
    add_curves!(ax_b, :times,   :tr_loss)

    # (c-d) Test accuracy
    ax_c = Axis(fig[2, 1];
        xlabel = "Minibatches (gradient evaluations)",
        ylabel = "Test accuracy",
        title  = "(c) Test accuracy vs. gradient evaluations")
    ax_d = Axis(fig[2, 2];
        xlabel = "Wall-clock time (s)",
        ylabel = "Test accuracy",
        title  = "(d) Test accuracy vs. time")
    add_curves!(ax_c, :batches, :te_acc); axislegend(ax_c; position = :rb)
    add_curves!(ax_d, :times,   :te_acc)

    # (e) Trust-region radius
    ax_e = Axis(fig[3, 1];
        xlabel = "Iteration k",
        ylabel = "Trust-region radius Δk", yscale = log10,
        title  = "(e) Adaptive step size: Δk over iterations (Tadam)")
    lines!(ax_e, h_tadam.hf_iters, h_tadam.hf_delta;
           color = C_TADAM, linewidth = 1.4)

    # (f) Rejection rate
    ax_f = Axis(fig[3, 2];
        xlabel = "Iteration k",
        ylabel = "Cumulative rejection rate",
        limits = (nothing, (0.0, 1.0)),
        title  = "(f) TR step rejection rate over training (Tadam)")
    lines!(ax_f, h_tadam.hf_iters, h_tadam.hf_rej;
           color = C_TADAM, linewidth = 2.0)

    # (g) Gradient norm
    ax_g = Axis(fig[4, 1:2];
        xlabel = "Minibatches (gradient evaluations)",
        ylabel = "Minibatch gradient norm",
        yscale = log10,
        title  = "(g) Gradient norm ‖g‖ over training")
    add_curves!(ax_g, :batches, :gnorm)

    Label(fig[5, :],
        "CIFAR-10  (N_train=$(N_TRAIN), N_test=$(N_TEST), batch=$(BATCH)).  " *
        "CNN: 3→32→64→128 conv + Dense 256→10.  " *
        "Adam & AMSGrad use lr=$(LR); Tadam adapts Δ automatically.\n" *
        "Panels (e–f): Tadam internal diagnostics.  Device: $(DEV)",
        tellwidth = false, fontsize = 11, color = :gray40,
    )

    save("cifar10_Tadam_results.pdf", fig)
    save("cifar10_Tadam_results.png", fig; px_per_unit = 2)
    display(fig)
    @info "Figures written: cifar10_Tadam_results.{pdf,png}"
end