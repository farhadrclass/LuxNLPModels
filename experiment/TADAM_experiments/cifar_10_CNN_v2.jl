# ================================================================
# cifar_10_CNN.jl
# CIFAR-10 Benchmark: Adam | AMSGrad | Tadam 
# Methodologically Corrected & GPU-Optimized
# ================================================================

using Lux, LuxCUDA
using CUDA
using ComponentArrays
using NLPModels, JSOSolvers, LuxNLPModels
using Optimisers
using MLDatasets
using OneHotArrays
using NNlib: logitcrossentropy
using Random, Statistics, Printf, LinearAlgebra
using CairoMakie
import CairoMakie: Axis
using Zygote

# ----------------------------------------------------------------
# 0. GPU / Device Setup (Reproducibility focused)
# ----------------------------------------------------------------
const USE_GPU = CUDA.functional()
USE_GPU && @info "CUDA GPU: $(CUDA.name(CUDA.device()))"
USE_GPU || @warn "No CUDA GPU found — falling back to CPU"

# FAST_MATH is disabled to guarantee numerical reproducibility across runs
# CUDA.math_mode!(CUDA.FAST_MATH) 

to_dev(x) = USE_GPU ? CUDA.cu(x) : x
USE_GPU && CUDA.allowscalar(false)

# ----------------------------------------------------------------
# 1. Unbiased CIFAR-10 Preparation
# ----------------------------------------------------------------
@info "Loading and Shuffling CIFAR-10..."
tr_x_raw, tr_y_raw = CIFAR10.traindata(Float32)
te_x_raw, te_y_raw = CIFAR10.testdata(Float32)

N_TRAIN, N_TEST = 20_000, 2_000

# Shuffle the entire dataset BEFORE selecting the subset to avoid bias
rng_data = MersenneTwister(42)
full_train_idx = randperm(rng_data, length(tr_y_raw))
train_idx = full_train_idx[1:N_TRAIN]

train_x_cpu = tr_x_raw[:, :, :, train_idx]
train_y_cpu = Float32.(onehotbatch(tr_y_raw[train_idx], 0:9))
test_x_cpu  = te_x_raw[:, :, :, 1:N_TEST]
test_y_cpu  = Float32.(onehotbatch(te_y_raw[1:N_TEST], 0:9))

const train_x_gpu = to_dev(train_x_cpu)
const train_y_gpu = to_dev(train_y_cpu)
const test_x_gpu  = to_dev(test_x_cpu)
const test_y_gpu  = to_dev(test_y_cpu)

const BATCH = 2048

# ----------------------------------------------------------------
# GPU Resident Minibatch Loader (Stateless Iterator)
# ----------------------------------------------------------------
struct GPULoader
    x_gpu::typeof(train_x_gpu)
    y_gpu::typeof(train_y_gpu)
    batchsize::Int
    n::Int
end

GPULoader(x, y, batchsize) = GPULoader(x, y, batchsize, size(x, 4))

function Base.iterate(d::GPULoader, state = (1, cu(randperm(d.n))))
    curr, idx_gpu = state
    curr > d.n && return nothing 
    
    end_idx = min(curr + d.batchsize - 1, d.n)
    batch_idx = idx_gpu[curr:end_idx]
    
    bx = d.x_gpu[:, :, :, batch_idx]
    by = d.y_gpu[:, batch_idx]
    
    return ((bx, by), (end_idx + 1, idx_gpu))
end

Base.length(d::GPULoader) = cld(d.n, d.batchsize)

# ----------------------------------------------------------------
# 2. CNN Model & Parameters
# ----------------------------------------------------------------
# Note: No BatchNorm/Dropout used, meaning `st_dev` mutation is not 
# strictly required during the optimization loop for this architecture.
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

const st_dev = to_dev(st_cpu)
const ps_dev_namedtuple = to_dev(ps_cpu)

const ps_template = ComponentArray(ps_cpu)
const axes_template = getaxes(ps_template)
const ps0_dev_ca = to_dev(copy(ps_template))

loss_fn(ŷ, y) = logitcrossentropy(ŷ, y)

# Zero-allocation recursive norm for NamedTuples
recursive_sq_norm(x::AbstractArray) = sum(abs2, x)
recursive_sq_norm(x::NamedTuple) = sum(recursive_sq_norm, values(x))

# ----------------------------------------------------------------
# Evaluation Metrics (Timer stops during this block)
# ----------------------------------------------------------------
function eval_metrics(ps_vec)
    ps_nt = ps_vec isa ComponentArray ? ComponentArray(ps_vec, axes_template) : ps_vec

    function batched_eval(x_data, y_data)
        N = size(x_data, 4)
        chunk = 10_000
        tot_loss, tot_correct = 0f0, 0

        for i in 1:chunk:N
            idx = i:min(i+chunk-1, N)
            bx, by = x_data[:, :, :, idx], y_data[:, idx]
            ŷ, _ = Lux.apply(model, bx, ps_nt, st_dev)
            
            l_val = loss_fn(ŷ, by)
            tot_loss += (l_val isa AbstractArray ? Float32(Array(l_val)[]) : Float32(l_val)) * length(idx)
            tot_correct += Int(Array(sum(argmax(ŷ, dims=1) .== argmax(by, dims=1)))[])
        end
        return tot_loss / N, Float32(tot_correct) / N
    end

    tr_loss, tr_acc = batched_eval(train_x_gpu, train_y_gpu)
    _,       te_acc = batched_eval(test_x_gpu,  test_y_gpu)
    return Float32(tr_loss), Float32(tr_acc), Float32(te_acc)
end

# ----------------------------------------------------------------
# 3. Methodologically Sound History Tracker
# ----------------------------------------------------------------
mutable struct Hist
    f_calls  :: Vector{Int}
    g_calls  :: Vector{Int}
    times    :: Vector{Float64}
    tr_loss  :: Vector{Float32}
    tr_acc   :: Vector{Float32}
    te_acc   :: Vector{Float32}
    gnorm    :: Vector{Float32}
    
    # Tadam specific tracking
    hf_iters :: Vector{Int}
    hf_delta :: Vector{Float64}
    hf_rej   :: Vector{Float64}
    
    elapsed  :: Float64
end

Hist() = Hist(Int[], Int[], Float64[], Float32[], Float32[], Float32[], Float32[], Int[], Float64[], Float64[], 0.0)

function snap!(h::Hist, f_calls, g_calls, ps_vec, tag, gn)
    tr_loss, tr_acc, te_acc = eval_metrics(ps_vec)
    push!(h.f_calls, f_calls); push!(h.g_calls, g_calls); push!(h.times, h.elapsed)
    push!(h.tr_loss, tr_loss); push!(h.tr_acc, tr_acc);   push!(h.te_acc, te_acc)
    push!(h.gnorm, Float32(gn))
    @printf "[%s] f_eval=%4d g_eval=%4d t=%5.1fs loss=%.4f tr=%.1f%% te=%.1f%% |g|=%.4f\n" tag f_calls g_calls h.elapsed tr_loss (100tr_acc) (100te_acc) gn
end

# ----------------------------------------------------------------
# 4a. Runner: First Order (Adam/AMSGrad)
# ----------------------------------------------------------------
function run_first_order!(rule, name; max_g_evals = 2000, eval_freq = 500)
    @info "=== $name ==="
    h = Hist()
    x = deepcopy(ps_dev_namedtuple)
    opt_state = Optimisers.setup(rule, x)
    loader = GPULoader(train_x_gpu, train_y_gpu, BATCH)
    infinite_loader = Iterators.Stateful(Iterators.cycle(loader))
    
    gn = 0f0
    g_calls = 0
    f_calls = 0

    while g_calls < max_g_evals
        (bx, by) = popfirst!(infinite_loader)

        CUDA.synchronize()
        t0 = time_ns()

        loss_val, back = Zygote.pullback(x) do p
            ŷ, _ = Lux.apply(model, bx, p, st_dev)
            loss_fn(ŷ, by)
        end
        grads = back(1f0)[1]
        opt_state, x = Optimisers.update!(opt_state, x, grads)

        CUDA.synchronize()
        h.elapsed += (time_ns() - t0) / 1e9
        
        # 1 step = 1 f_eval + 1 g_eval
        f_calls += 1
        g_calls += 1

        if g_calls % eval_freq == 0 || g_calls == max_g_evals
            gn = Float32(sqrt(recursive_sq_norm(grads)))
            snap!(h, f_calls, g_calls, x, name, gn)
        end
    end
    
    return x, h
end

# ----------------------------------------------------------------
# 4b. Runner: Tadam
# ----------------------------------------------------------------
function run_tadam!(; max_g_evals = 2000, eval_freq = 500, η1 = 0.10f0, kwargs...)
    @info "=== Tadam ==="
    loader = GPULoader(train_x_gpu, train_y_gpu, BATCH)
    infinite_loader = Iterators.Stateful(Iterators.cycle(loader))
    
    nlp = LuxNLPModel(model, copy(ps0_dev_ca), st_dev, infinite_loader, loss_fn)
    h = Hist()

    cb = (nlp, solver, stats) -> begin
        # Fetch objective and gradient evaluation counts natively from NLPModel
        f_calls = neval_obj(nlp)
        g_calls = neval_grad(nlp)
        
        push!(h.hf_iters, stats.iter)
        push!(h.hf_delta, Float64(solver.Δ))
        
        # Track rejection rate logically
        total_steps = stats.iter
        acc_steps = sum(h.hf_delta[1:end-1] .!= h.hf_delta[2:end]) # Simplistic proxy for accepted
        push!(h.hf_rej, total_steps > 0 ? (total_steps - acc_steps) / total_steps : 0.0)

        # Trigger evaluation strictly based on gradient evaluations budget
        if g_calls > 0 && (g_calls % eval_freq == 0 || g_calls >= max_g_evals)
            gn = Float32(norm(solver.gx))
            # Pause timer for eval
            snap!(h, f_calls, g_calls, solver.x, "Tadam", gn)
        end

        if solver.step_accepted
            minibatch_next_train!(nlp)
        end

        # Small step rescue
        if stats.status == :small_step
            ng = norm(solver.gx)
            solver.Δ = max(ng / (2^round(log2(ng + 1f0))), 1f-5)
            stats.status = :unknown
        end
        
        # Terminate if gradient budget is met
        g_calls >= max_g_evals && (stats.status = :user)
    end

    CUDA.synchronize()
    t0 = time_ns()
    
    # Custom timing wrapper to pause timer inside the callback
    function timing_cb(nlp, solver, stats)
        CUDA.synchronize()
        h.elapsed += (time_ns() - t0) / 1e9
        cb(nlp, solver, stats)
        CUDA.synchronize()
        t0 = time_ns()
    end

    stats = tadam(nlp; max_iter = max_g_evals * 10, atol = 1f-8, rtol = 1f-5, callback = timing_cb, verbose = 0, η1, kwargs...)

    return stats, h
end







# ----------------------------------------------------------------
# Warmup Compiler (Prevents JIT overhead from ruining benchmarks)
# ----------------------------------------------------------------
@info "Warming up Zygote/GPU compiler (this will take 1-3 minutes)..."
let
    # Create tiny dummy arrays just to trigger compilation
    dummy_x = CUDA.zeros(Float32, 32, 32, 3, 2)
    dummy_y = CUDA.zeros(Float32, 10, 2)
    
    # Run one forward and backward pass
    _, dummy_back = Zygote.pullback(ps_dev_namedtuple) do p
        ŷ, _ = Lux.apply(model, dummy_x, p, st_dev)
        loss_fn(ŷ, dummy_y)
    end
    dummy_back(1f0)
end
@info "Warmup complete! Starting real benchmarks..."

# ----------------------------------------------------------------
# 5. Execute Experiments
# ----------------------------------------------------------------
const EVAL_FREQ = 500
const MAX_G_EVALS = 2000
const LR = 3f-4

_, h_adam    = run_first_order!(Optimisers.Adam(LR), "Adam"; max_g_evals = MAX_G_EVALS, eval_freq = EVAL_FREQ)
_, h_amsgrad = run_first_order!(Optimisers.AMSGrad(LR), "AMSGrad"; max_g_evals = MAX_G_EVALS, eval_freq = EVAL_FREQ)

_, h_tadam = run_tadam!(;
    max_g_evals = MAX_G_EVALS,
    eval_freq = EVAL_FREQ,
    η1 = 0.0001f0,
    η2 = 0.85f0,
    γ1 = 0.80f0,
    γ2 = 1.20f0,
    γ3 = 0.02f0,
    β1 = 0.90f0,
    β2 = 0.99f0,
    ϵ_v = 1f-7,
    θ1 = 1f-6,
    # Δmax removed as per review recommendations to allow proper scaling
)

# ----------------------------------------------------------------
# 6. Publication Figure (X-axis corrected to Gradient Evaluations)
# ----------------------------------------------------------------
const C_ADAM    = RGBf(0.902, 0.624, 0.000)
const C_AMSGRAD = RGBf(0.835, 0.369, 0.000)
const C_TADAM   = RGBf(0.000, 0.447, 0.698)

pub_theme = Theme(
    fontsize = 14,
    Axis = (spinewidth = 0.9, xgridcolor = (:black, 0.08), ygridcolor = (:black, 0.08), xgridwidth = 0.6, ygridwidth = 0.6),
    Legend = (framevisible = false, labelsize = 11, patchsize = (22, 2)), Lines = (linewidth = 2.3,)
)

with_theme(pub_theme) do
    fig = Figure(size = (900, 1150))

    function add_curves!(ax, xfield, yfield)
        for (h, label, color) in [(h_adam, "Adam", C_ADAM), (h_amsgrad, "AMSGrad", C_AMSGRAD), (h_tadam, "Tadam", C_TADAM)]
            lines!(ax, getfield(h, xfield), getfield(h, yfield); color, label)
        end
    end

    # Note the change from :batches to :g_calls
    ax_a = Axis(fig[1,1]; xlabel = "Gradient Evaluations", ylabel = "Train loss", yscale = log10, title = "(a) Train loss vs. Eval Budget")
    ax_b = Axis(fig[1,2]; xlabel = "Wall-clock time (s)", ylabel = "Train loss", yscale = log10, title = "(b) Train loss vs. Time")
    add_curves!(ax_a, :g_calls, :tr_loss); axislegend(ax_a; position = :rt)
    add_curves!(ax_b, :times,   :tr_loss)

    ax_c = Axis(fig[2,1]; xlabel = "Gradient Evaluations", ylabel = "Test accuracy", title = "(c) Test accuracy vs. Eval Budget")
    ax_d = Axis(fig[2,2]; xlabel = "Wall-clock time (s)", ylabel = "Test accuracy", title = "(d) Test accuracy vs. Time")
    add_curves!(ax_c, :g_calls, :te_acc); axislegend(ax_c; position = :rb)
    add_curves!(ax_d, :times,   :te_acc)

    ax_e = Axis(fig[3,1]; xlabel = "Iteration k", ylabel = "Trust-region radius Δk", yscale = log10, title = "(e) Adaptive step size (Tadam)")
    lines!(ax_e, h_tadam.hf_iters, h_tadam.hf_delta; color = C_TADAM, linewidth = 1.4)

    ax_f = Axis(fig[3,2]; xlabel = "Iteration k", ylabel = "Cumulative rejection rate", limits = (nothing, (0.0, 1.0)), title = "(f) TR rejection rate (Tadam)")
    lines!(ax_f, h_tadam.hf_iters, h_tadam.hf_rej; color = C_TADAM, linewidth = 2.0)

    ax_g = Axis(fig[4, 1:2]; xlabel = "Gradient Evaluations", ylabel = "Minibatch Gradient Norm", yscale = log10, title = "(g) Gradient Norm ||g||")
    add_curves!(ax_g, :g_calls, :gnorm)

    # Note text updated to reflect single seed execution as requested
    Label(fig[5, :],
        "CIFAR-10 (N_train=$(N_TRAIN), N_test=$(N_TEST), batch=$(BATCH)).\n" *
        "X-axis tracks true gradient evaluations to ensure fair budget comparison against Trust-Region rejection steps.\n" *
        "Experiment represents a single seed execution (seed 42). Wall-clock time strictly isolates forward/backward passes.",
        tellwidth = false, fontsize = 11, color = :gray40,
    )

    save("cifar10_Tadam_results_Corrected.pdf", fig)
    display(fig)
end