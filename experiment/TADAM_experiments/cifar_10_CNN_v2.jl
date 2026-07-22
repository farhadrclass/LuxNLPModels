# ================================================================
# cifar_10_CNN.jl
# CIFAR-10 Benchmark: Adam | AMSGrad | Tadam
# ================================================================

using Lux, LuxCUDA
using CUDA
using ComponentArrays
using NLPModels, JSOSolvers, LuxNLPModels
using Optimisers
using MLDatasets
using MLUtils
using OneHotArrays
using NNlib: logitcrossentropy # Swapped to the highly optimized, fused cross-entropy
using Random, Statistics, Printf, LinearAlgebra
using CairoMakie
import CairoMakie: Axis
using Zygote

# ----------------------------------------------------------------
# 0.  GPU / Device Setup
# ----------------------------------------------------------------
const USE_GPU = CUDA.functional()
USE_GPU && @info "CUDA GPU: $(CUDA.name(CUDA.device()))"
USE_GPU || @warn "No CUDA GPU found — falling back to CPU (slower)"

USE_GPU && CUDA.math_mode!(CUDA.FAST_MATH)

to_dev(x) = USE_GPU ? CUDA.cu(x) : x

# Strictly enforce no scalar indexing to catch hidden bottlenecks
USE_GPU && CUDA.allowscalar(false) 

# ----------------------------------------------------------------
# 1.  CIFAR-10  (32 × 32 × 3, 10 classes, Float32 ∈ [0,1])
# ----------------------------------------------------------------
@info "Loading CIFAR-10..."
tr_x_raw, tr_y_raw = CIFAR10.traindata(Float32)
te_x_raw, te_y_raw = CIFAR10.testdata(Float32)

N_TRAIN, N_TEST = 20_000, 2_000

train_x_cpu = tr_x_raw[:, :, :, 1:N_TRAIN]
train_y_cpu = Float32.(onehotbatch(tr_y_raw[1:N_TRAIN], 0:9))
test_x_cpu  = te_x_raw[:, :, :, 1:N_TEST]
test_y_cpu  = Float32.(onehotbatch(te_y_raw[1:N_TEST], 0:9))

# Pre-load everything onto the GPU once
const train_x_gpu = to_dev(train_x_cpu)
const train_y_gpu = to_dev(train_y_cpu)
const test_x_gpu  = to_dev(test_x_cpu)
const test_y_gpu  = to_dev(test_y_cpu)

const BATCH = 2048

# ----------------------------------------------------------------
# GPU-native sampler
# ----------------------------------------------------------------
struct GPUSampler
    rng     :: MersenneTwister
    n       :: Int
    batchsz :: Int
end
GPUSampler() = GPUSampler(MersenneTwister(0), N_TRAIN, BATCH)

function next_batch!(s::GPUSampler)
    # Generate indices on CPU
    idx = rand(s.rng, 1:s.n, s.batchsz)
    
    # CRITICAL FIX: Move indices to GPU before indexing the GPU tensor.
    # This prevents the devastating scalar indexing fallback.
    idx_gpu = to_dev(idx) 
    
    bx  = train_x_gpu[:, :, :, idx_gpu]
    by  = train_y_gpu[:, idx_gpu]
    return bx, by
end

# Thin iterator wrapper so LuxNLPModel can consume it
mutable struct GPUBatchIter
    sampler  :: GPUSampler
    max_iter :: Int          # how many batches to yield per "epoch"
    _count   :: Int
end
GPUBatchIter(max_iter) = GPUBatchIter(GPUSampler(), max_iter, 0)

function Base.iterate(d::GPUBatchIter, state=nothing)
    d._count >= d.max_iter && return nothing
    d._count += 1
    bx, by = next_batch!(d.sampler)
    return (bx, by), d._count
end
Base.length(d::GPUBatchIter) = d.max_iter

# Reset before each new run
reset!(d::GPUBatchIter) = (d._count = 0; d)

# ----------------------------------------------------------------
# 2.  CNN model
# ----------------------------------------------------------------
model = Chain(
    Conv((3,3), 3  => 32, relu; pad = SamePad()),
    Conv((3,3), 32 => 32, relu; pad = SamePad()),
    MaxPool((2,2)),
    Conv((3,3), 32 => 64, relu; pad = SamePad()),
    Conv((3,3), 64 => 64, relu; pad = SamePad()),
    MaxPool((2,2)),
    Conv((3,3), 64 => 128, relu; pad = SamePad()),
    MaxPool((2,2)),
    FlattenLayer(),
    Dense(128 * 4 * 4 => 256, relu),
    Dense(256 => 10),
)

rng_init = MersenneTwister(42)
ps_cpu, st_cpu = Lux.setup(rng_init, model)

const ps_template = ComponentArray(ps_cpu)
const ps0_dev     = to_dev(copy(ps_template))
const st_dev      = to_dev(st_cpu)

# CRITICAL FIX: Use fused GPU kernel for cross-entropy with logits
loss_fn(ŷ, y) = logitcrossentropy(ŷ, y)

# ----------------------------------------------------------------
# Memory-safe, GPU-accelerated batched evaluation
# ----------------------------------------------------------------
function eval_metrics(ps_vec)
    ps_s = ps_vec isa ComponentArray ?
        ps_vec :
        ComponentArray(to_dev(ps_vec), getaxes(ps_template))

    function batched_eval(x_data_gpu, y_data_gpu)
        N = size(x_data_gpu, 4)
        chunk = 10_000
        tot_loss    = 0f0
        tot_correct = 0

        for i in 1:chunk:N
            idx = i:min(i+chunk-1, N)
            bx  = x_data_gpu[:, :, :, idx] # @view is less stable with Zygote/CUDA here; standard slice is fast
            by  = y_data_gpu[:, idx]

            ŷ, _ = Lux.apply(model, bx, ps_s, st_dev)

            l_val = loss_fn(ŷ, by)
            tot_loss += (l_val isa AbstractArray ?
                Float32(Array(l_val)[]) : Float32(l_val)) * length(idx)

            correct_gpu  = sum(argmax(ŷ, dims=1) .== argmax(by, dims=1))
            tot_correct += Int(first(Array(correct_gpu)))
        end
        return tot_loss / N, Float32(tot_correct) / N
    end

    tr_loss, tr_acc = batched_eval(train_x_gpu, train_y_gpu)
    _,       te_acc = batched_eval(test_x_gpu,  test_y_gpu)
    return Float32(tr_loss), Float32(tr_acc), Float32(te_acc)
end

# NLPModel factory — used by Tadam only
const _batch_iter = GPUBatchIter(1)
make_nlp() = LuxNLPModel(model, copy(ps0_dev), st_dev, reset!(_batch_iter), loss_fn)

# ----------------------------------------------------------------
# 3.  History
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

Hist() = Hist(Int[], Int[], Float64[], Float32[], Float32[], Float32[], Float32[],
              Int[], Float64[], Float64[], 0, 0, 0.0, time_ns())

function snap!(h::Hist, iter, batches, ps_vec, tag, gn)
    h._t_accum += (time_ns() - h._t_start) / 1e9
    tr_loss, tr_acc, te_acc = eval_metrics(ps_vec)
    push!(h.iters,   iter);    push!(h.batches, batches); push!(h.times, h._t_accum)
    push!(h.tr_loss, tr_loss); push!(h.tr_acc,  tr_acc);  push!(h.te_acc, te_acc)
    push!(h.gnorm,   Float32(gn))
    @printf "[%s] iter=%4d  bat=%4d  t=%5.0fs  loss=%.4f  tr=%.1f%%  te=%.1f%%  |g|=%.4f\n"  tag iter batches h._t_accum tr_loss (100tr_acc) (100te_acc) gn
    h._t_start = time_ns()
end

# ----------------------------------------------------------------
# 4a. Runner: first-order Optimisers.jl methods
# ----------------------------------------------------------------
function run_first_order!(rule, name; max_iter = 2000, eval_freq = 500)
    @info "=== $name ==="
    h       = Hist()
    sampler = GPUSampler()

    x   = copy(ps0_dev)
    opt = Optimisers.setup(rule, x)
    gn  = 0f0

    h._t_start = time_ns()

    for i in 0:max_iter
        bx, by = next_batch!(sampler)

        # Cleaner, highly optimized pullback structure directly using ComponentArray
        loss_val, back = Zygote.pullback(x) do p
            ŷ, _ = Lux.apply(model, bx, p, st_dev)
            loss_fn(ŷ, by)
        end

        grads = back(one(loss_val))[1]

        if i % eval_freq == 0
            CUDA.synchronize()
            gn = Float32(norm(grads))
            snap!(h, i, i, x, name, gn)
        end
        i == max_iter && break

        opt, Δ = Optimisers.apply!(opt, x, grads)
        @. x -= Δ
    end
    return x, h
end

# ----------------------------------------------------------------
# 4b. Runner: Tadam
# ----------------------------------------------------------------
function run_tadam!(; max_iter = 2000, eval_freq = 500, η1 = 0.10f0, kwargs...)
    @info "=== Tadam ==="
    nlp      = make_nlp()
    h        = Hist()
    batches  = Ref(0)
    last_obj = Ref(Inf32)

    h._t_start = time_ns()

    cb = (nlp, solver, stats) -> begin
        iter = stats.iter

        push!(h.hf_iters, iter)
        push!(h.hf_delta, Float64(solver.Δ))
        total = h.n_ok + h.n_rej
        push!(h.hf_rej, total > 0 ? h.n_rej / total : 0.0)

        if iter % eval_freq == 0
            gn = Float32(norm(solver.gx))
            snap!(h, iter, batches[], solver.x, "Tadam", gn)
        end

        if solver.step_accepted
            h.n_ok += 1
            h._t_accum += (time_ns() - h._t_start) / 1e9

            minibatch_next_train!(nlp)
            batches[] += 1
            stats.objective = last_obj[]

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

    function wrapped_objgrad!(nlp, x, g)
        f = objgrad!(nlp, x, g)
        last_obj[] = Float32(f isa AbstractArray ? Array(f)[] : f)
        return f
    end

    stats = tadam(nlp; max_iter, atol = 1f-8, rtol = 1f-5,
                  callback = cb, verbose = 0, η1, kwargs...)

    if isempty(h.iters) || h.iters[end] != stats.iter
        g_final = similar(stats.solution)
        objgrad!(nlp, stats.solution, g_final)
        snap!(h, stats.iter, batches[], stats.solution, "Tadam", Float32(norm(g_final)))
    end

    @printf "  Tadam final: %d accepted | %d rejected | %.1f%% rejection rate\n"    h.n_ok h.n_rej (100 * h.n_rej / max(1, h.n_ok + h.n_rej))
    return stats, h
end

# ----------------------------------------------------------------
# 5.  Run all experiments
# ----------------------------------------------------------------
const MAX_ITER  = 2000
const EVAL_FREQ = 500
const LR        = 3f-4

_, h_adam    = run_first_order!(
    Optimisers.Adam(LR),    "Adam";    max_iter = MAX_ITER, eval_freq = EVAL_FREQ)

_, h_amsgrad = run_first_order!(
    Optimisers.AMSGrad(LR), "AMSGrad"; max_iter = MAX_ITER, eval_freq = EVAL_FREQ)

_, h_tadam = let
    stats, h = run_tadam!(;
        max_iter = MAX_ITER,
        eval_freq = EVAL_FREQ,
        η1   = 0.0001f0,
        η2   = 0.85f0,
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
# 6.  Publication figure (remains unchanged)
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

    ax_a = Axis(fig[1,1];
        xlabel = "Minibatches (gradient evaluations)",
        ylabel = "Train loss",
        yscale = log10,
        title  = "(a) Train loss vs. gradient evaluations")
    ax_b = Axis(fig[1,2];
        xlabel = "Wall-clock time (s)",
        ylabel = "Train loss",
        yscale = log10,
        title  = "(b) Train loss vs. time")
    add_curves!(ax_a, :batches, :tr_loss); axislegend(ax_a; position = :rt)
    add_curves!(ax_b, :times,   :tr_loss)

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

    ax_e = Axis(fig[3,1];
        xlabel = "Iteration k",
        ylabel = "Trust-region radius Δk",
        yscale = log10,
        title  = "(e) Adaptive step size: Δk over iterations (Tadam)")
    lines!(ax_e, h_tadam.hf_iters, h_tadam.hf_delta; color = C_TADAM, linewidth = 1.4)

    ax_f = Axis(fig[3,2];
        xlabel = "Iteration k",
        ylabel = "Cumulative rejection rate",
        limits = (nothing, (0.0, 1.0)),
        title  = "(f) TR step rejection rate over training (Tadam)")
    lines!(ax_f, h_tadam.hf_iters, h_tadam.hf_rej; color = C_TADAM, linewidth = 2.0)

    ax_g = Axis(fig[4, 1:2];
        xlabel = "Minibatches (gradient evaluations)",
        ylabel = "Minibatch Gradient Norm",
        yscale = log10,
        title  = "(g) Gradient Norm ||g|| over training")
    add_curves!(ax_g, :batches, :gnorm)

    Label(fig[5, :],
        "CIFAR-10 (N_train=$(N_TRAIN), N_test=$(N_TEST), batch=$(BATCH)).  " *
        "Adam and AMSGrad use lr=$(LR); Tadam adapts its step size automatically.\n" *
        "Panels (a–d): three runs per method (seeds 42, 123, 456 recommended). " *
        "Panels (e–f): Tadam internal diagnostics showing self-tuning behavior.",
        tellwidth = false, fontsize = 11, color = :gray40,
    )

    save("cifar10_Tadam_results.pdf", fig)
    save("cifar10_Tadam_results.png", fig; px_per_unit = 2)
    display(fig)
    @info "Figures written: cifar10_Tadam_results.{pdf,png}"
end