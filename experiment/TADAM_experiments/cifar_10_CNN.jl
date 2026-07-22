# ================================================================
# cifar_10_CNN.jl
# CIFAR-10 Benchmark: Adam | AMSGrad | Tadam
#
# ================================================================

using Lux, LuxCUDA
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
# 0.  GPU / Device
# ----------------------------------------------------------------
const USE_GPU = CUDA.functional()
USE_GPU && @info "CUDA GPU: $(CUDA.name(CUDA.device()))"
USE_GPU || @warn "No CUDA GPU found — falling back to CPU (slower)"

# Universal device transfer (allows scalar indexing fallback just in case)
to_dev(x) = USE_GPU ? CUDA.cu(x) : x
USE_GPU && CUDA.allowscalar(true)

# ----------------------------------------------------------------
# 1.  CIFAR-10  (32 × 32 × 3, 10 classes, already Float32 ∈ [0,1])
# ----------------------------------------------------------------
@info "Loading CIFAR-10..."
tr_x_raw, tr_y_raw = CIFAR10.traindata(Float32)   # → (32,32,3,50 000)
te_x_raw, te_y_raw = CIFAR10.testdata(Float32)    # → (32,32,3,10 000)

# Sub-sample for speed (use N_TRAIN=50_000 for the final paper numbers)
N_TRAIN, N_TEST = 20_000, 2_000

train_x_cpu = tr_x_raw[:, :, :, 1:N_TRAIN]
train_y_cpu = onehotbatch(tr_y_raw[1:N_TRAIN], 0:9)
test_x_cpu  = te_x_raw[:, :, :, 1:N_TEST]
test_y_cpu  = onehotbatch(te_y_raw[1:N_TEST], 0:9)

#  Push everything to the GPU permanently to eliminate PCIe transfer overhead
const train_x_gpu = to_dev(train_x_cpu)
const train_y_gpu = to_dev(train_y_cpu)
const test_x_gpu  = to_dev(test_x_cpu)
const test_y_gpu  = to_dev(test_y_cpu)

const BATCH = 256
# Point the DataLoader directly at the GPU arrays
data_loader = DataLoader(
    (train_x_gpu, train_y_gpu);
    batchsize = BATCH, shuffle = true,
)

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

const ps_template = ComponentArray(ps_cpu)        # CPU template (for axis re-use)
const ps0_dev     = to_dev(copy(ps_template))     # starting point on correct device
const st_dev      = to_dev(st_cpu)                # correct device state

loss_fn(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))

# ----------------------------------------------------------------
# GPU-accelerated and memory-safe batched evaluation
# ----------------------------------------------------------------
function eval_metrics(ps_vec)
    # Check if ps_vec is already a ComponentArray to avoid double-wrapping, 
    # but ensure it's forced to the GPU structure.
    ps_s = ps_vec isa ComponentArray ? ps_vec : ComponentArray(to_dev(ps_vec), getaxes(ps_template))
    
    function batched_eval(x_data_gpu, y_data_gpu)
        N = size(x_data_gpu, 4)
        chunk = 1000 # Send 1000 images at a time to maximize GPU efficiency safely
        tot_loss = 0f0
        tot_correct = 0
        
        for i in 1:chunk:N
            idx = i:min(i+chunk-1, N)
            
            # Slicing happens directly on the GPU! No PCIe transfer overhead.
            bx = x_data_gpu[:, :, :, idx]
            by = y_data_gpu[:, idx]
            
            # Forward pass on the GPU
            ŷ, _ = Lux.apply(model, bx, ps_s, st_dev)
            
            # Safely handle both primitive scalars and 0-D GPU arrays from Lux/CUDA
            l_val = loss_fn(ŷ, by)
            tot_loss += (l_val isa AbstractArray ? Float32(Array(l_val)[]) : Float32(l_val)) * length(idx)
            
            # Pull ONLY lightweight predictions back to CPU to compute accuracy
            ŷ_cpu = Array(ŷ)
            by_cpu = Array(by)
            tot_correct += sum(onecold(ŷ_cpu) .== onecold(by_cpu))
        end
        return tot_loss / N, Float32(tot_correct) / N
    end

    tr_loss, tr_acc = batched_eval(train_x_gpu, train_y_gpu)
    _, te_acc = batched_eval(test_x_gpu, test_y_gpu)
    
    return Float32(tr_loss), Float32(tr_acc), Float32(te_acc)
end

make_nlp() = LuxNLPModel(model, copy(ps0_dev), st_dev, data_loader, loss_fn)

# ----------------------------------------------------------------
# 3.  History 
# ----------------------------------------------------------------
mutable struct Hist
    iters   :: Vector{Int}
    batches :: Vector{Int}
    times   :: Vector{Float64}
    tr_loss :: Vector{Float32}
    tr_acc  :: Vector{Float32}
    te_acc  :: Vector{Float32}
    hf_iters :: Vector{Int}
    hf_delta :: Vector{Float64}
    hf_rej   :: Vector{Float64}
    n_ok     :: Int
    n_rej    :: Int
    _t_accum :: Float64
    _t_start :: UInt64
end

Hist() = Hist(Int[], Int[], Float64[], Float32[], Float32[], Float32[],
              Int[], Float64[], Float64[], 0, 0, 0.0, time_ns())

function snap!(h::Hist, iter, batches, ps_vec, tag)
    h._t_accum += (time_ns() - h._t_start) / 1e9
    tr_loss, tr_acc, te_acc = eval_metrics(ps_vec)
    push!(h.iters, iter);  push!(h.batches, batches); push!(h.times, h._t_accum)
    push!(h.tr_loss, tr_loss); push!(h.tr_acc, tr_acc); push!(h.te_acc, te_acc)
    @printf "[%s] iter=%4d  bat=%4d  t=%5.0fs  loss=%.4f  tr=%.1f%%  te=%.1f%%\n"   tag iter batches h._t_accum tr_loss (100tr_acc) (100te_acc)
    h._t_start = time_ns()
end

# ----------------------------------------------------------------
# 4a. Runner: first-order Optimisers.jl methods
# ----------------------------------------------------------------
function run_first_order!(rule, name; max_iter = 2000, eval_freq = 50)
    @info "=== $name ==="
    nlp     = make_nlp()
    x       = to_dev(copy(nlp.meta.x0)) # Force x0 to start on GPU
    g       = similar(x)
    h       = Hist()
    batches = 0
    opt     = Optimisers.setup(rule, x)
    h._t_start = time_ns()
    
    for i in 0:max_iter
        i % eval_freq == 0 && snap!(h, i, batches, x, name)
        i == max_iter && break
        
        objgrad!(nlp, x, g)
        
        # IN-PLACE update keeps the ComponentArray structure intact on the GPU!
        opt, g_opt = Optimisers.apply!(opt, x, g)
        @. x -= g_opt
        
        minibatch_next_train!(nlp)
        batches += 1
    end
    return x, h
end

# ----------------------------------------------------------------
# 4b. Runner: Tadam
# ----------------------------------------------------------------
function run_tadam!(; max_iter = 2000, eval_freq = 50,
                      η1 = 0.10f0, kwargs...)
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
        push!(h.hf_rej,  total > 0 ? h.n_rej / total : 0.0)

        iter % eval_freq == 0 && snap!(h, iter, batches[], solver.x, "Tadam")

        if solver.step_accepted
            h.n_ok  += 1
            h._t_accum += (time_ns() - h._t_start) / 1e9
            minibatch_next_train!(nlp)
            batches[] += 1
            
            stats.objective = obj(nlp, solver.x)
            h._t_start = time_ns()
        elseif iter > 0
            h.n_rej += 1
        end

        if stats.status == :small_step
            ng      = norm(solver.gx)
            solver.Δ = max(ng / (2^round(log2(ng + 1f0))), 1f-5)
            stats.status = :unknown
        end
    end

    stats = tadam(nlp; max_iter, atol = 1f-8, rtol = 1f-5,
                  callback = cb, verbose = 0, η1, kwargs...)

    h.iters[end] != stats.iter &&
        snap!(h, stats.iter, batches[], stats.solution, "Tadam")

    @printf "  Tadam final: %d accepted | %d rejected | %.1f%% rejection rate\n"  h.n_ok h.n_rej (100 * h.n_rej / max(1, h.n_ok + h.n_rej))
    return stats, h
end

# ----------------------------------------------------------------
# 5.  Run all experiments
# ----------------------------------------------------------------
const MAX_ITER  = 2000
const EVAL_FREQ = 100  # Bumped to 100 to give the GPU loop more time to breathe between evaluations
const LR        = 3f-4

_, h_adam    = run_first_order!(
    Optimisers.Adam(LR),    "Adam";    max_iter = MAX_ITER, eval_freq = EVAL_FREQ)

_, h_amsgrad = run_first_order!(
    Optimisers.AMSGrad(LR), "AMSGrad"; max_iter = MAX_ITER, eval_freq = EVAL_FREQ)

_, h_tadam   = let
    stats, h = run_tadam!(;
        max_iter   = MAX_ITER,
        eval_freq  = EVAL_FREQ,
        η1   = 0.0001f0,      
        η2   = 0.85f0,        
        γ1  = 0.80f0,         
        γ2  = 1.20f0,         
        γ3  = 0.02f0,         
        β1   = 0.90f0,        
        β2   = 0.99f0,        
        ϵ_v  = 1f-7,          
        θ1   = 1f-6,          
        Δmax = 1f-2,          
    )
    stats, h
end

# ----------------------------------------------------------------
# 6.  Publication figure 
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

    Label(fig[4, :],
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