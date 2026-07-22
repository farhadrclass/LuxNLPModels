# ================================================================
# cifar_10_CNN.jl
# Methodologically Corrected, VRAM-Optimized, Fast Compilation
# ================================================================

using Lux, LuxCUDA, CUDA
using ComponentArrays
using NLPModels, JSOSolvers, LuxNLPModels
using Optimisers
using MLDatasets, OneHotArrays
using NNlib: logitcrossentropy
using Random, Statistics, Printf, LinearAlgebra
using CairoMakie
import CairoMakie: Axis
using Zygote

# ----------------------------------------------------------------
# 0. Hardware & Seed
# ----------------------------------------------------------------
const USE_GPU = CUDA.functional()
USE_GPU && @info "CUDA GPU: $(CUDA.name(CUDA.device()))"
to_dev(x) = USE_GPU ? CUDA.cu(x) : x
USE_GPU && CUDA.allowscalar(false)

# ----------------------------------------------------------------
# 1. Unbiased Data Setup
# ----------------------------------------------------------------
@info "Loading CIFAR-10..."
tr_x_raw, tr_y_raw = CIFAR10.traindata(Float32)
te_x_raw, te_y_raw = CIFAR10.testdata(Float32)

N_TRAIN, N_TEST = 20_000, 2_000

rng_data = MersenneTwister(42)
full_train_idx = randperm(rng_data, length(tr_y_raw))
train_idx = full_train_idx[1:N_TRAIN]

const train_x_gpu = to_dev(tr_x_raw[:, :, :, train_idx])
const train_y_gpu = to_dev(Float32.(onehotbatch(tr_y_raw[train_idx], 0:9)))
const test_x_gpu  = to_dev(te_x_raw[:, :, :, 1:N_TEST])
const test_y_gpu  = to_dev(Float32.(onehotbatch(te_y_raw[1:N_TEST], 0:9)))

const BATCH = 2048

# ----------------------------------------------------------------
# GPU Iterator
# ----------------------------------------------------------------
struct GPULoader
    x_gpu
    y_gpu
    batchsize::Int
    n::Int
end
GPULoader(x, y, batchsize) = GPULoader(x, y, batchsize, size(x, 4))

function Base.iterate(d::GPULoader, state = (1, cu(randperm(d.n))))
    curr, idx_gpu = state
    curr > d.n && return nothing 
    end_idx = min(curr + d.batchsize - 1, d.n)
    batch_idx = idx_gpu[curr:end_idx]
    return ((d.x_gpu[:, :, :, batch_idx], d.y_gpu[:, batch_idx]), (end_idx + 1, idx_gpu))
end
Base.length(d::GPULoader) = cld(d.n, d.batchsize)

# ----------------------------------------------------------------
# 2. Architecture 
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
    Dense(256 => 10)
)

ps_cpu, st_cpu = Lux.setup(MersenneTwister(42), model)
const st_dev = to_dev(st_cpu)
const ps_dev_namedtuple = to_dev(ps_cpu)

const ps_template = ComponentArray(ps_cpu)
const axes_template = getaxes(ps_template)
const ps0_dev_ca = to_dev(copy(ps_template))

loss_fn(ŷ, y) = logitcrossentropy(ŷ, y)
recursive_sq_norm(x::AbstractArray) = sum(abs2, x)
recursive_sq_norm(x::NamedTuple) = sum(recursive_sq_norm, values(x))

# ----------------------------------------------------------------
# Compilation Warmup (Solves the 10-minute hang)
# ----------------------------------------------------------------
@info "Warming up Zygote compiler (~30 seconds)..."
let
    bx, by = train_x_gpu[:,:,:,1:2], train_y_gpu[:,1:2]
    # Warmup standard path
    Zygote.withgradient(p -> loss_fn(Lux.apply(model, bx, p, st_dev)[1], by), ps_dev_namedtuple)
    # Warmup ComponentArray path
    Zygote.withgradient(p -> loss_fn(Lux.apply(model, bx, p, st_dev)[1], by), ps0_dev_ca)
end
@info "Warmup complete."

# ----------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------
function eval_metrics(ps_vec)
    ps_nt = ps_vec isa ComponentArray ? ComponentArray(ps_vec, axes_template) : ps_vec
    function batched_eval(x_data, y_data)
        N = size(x_data, 4)
        tot_loss, tot_correct = 0f0, 0
        for i in 1:10_000:N
            idx = i:min(i+9999, N)
            bx, by = x_data[:, :, :, idx], y_data[:, idx]
            ŷ, _ = Lux.apply(model, bx, ps_nt, st_dev)
            tot_loss += Float32(sum(loss_fn(ŷ, by))) * length(idx)
            tot_correct += Int(Array(sum(argmax(ŷ, dims=1) .== argmax(by, dims=1)))[])
        end
        return tot_loss / N, Float32(tot_correct) / N
    end
    tr_loss, tr_acc = batched_eval(train_x_gpu, train_y_gpu)
    _,       te_acc = batched_eval(test_x_gpu,  test_y_gpu)
    return Float32(tr_loss), Float32(tr_acc), Float32(te_acc)
end

# ----------------------------------------------------------------
# 3. History
# ----------------------------------------------------------------
mutable struct Hist
    f_calls::Vector{Int}; g_calls::Vector{Int}; times::Vector{Float64}
    tr_loss::Vector{Float32}; tr_acc::Vector{Float32}; te_acc::Vector{Float32}; gnorm::Vector{Float32}
    hf_iters::Vector{Int}; hf_delta::Vector{Float64}; hf_rej::Vector{Float64}; elapsed::Float64
end
Hist() = Hist(Int[], Int[], Float64[], Float32[], Float32[], Float32[], Float32[], Int[], Float64[], Float64[], 0.0)

function snap!(h::Hist, f_calls, g_calls, ps_vec, tag, gn)
    tr_loss, tr_acc, te_acc = eval_metrics(ps_vec)
    push!(h.f_calls, f_calls); push!(h.g_calls, g_calls); push!(h.times, h.elapsed)
    push!(h.tr_loss, tr_loss); push!(h.tr_acc, tr_acc);   push!(h.te_acc, te_acc); push!(h.gnorm, Float32(gn))
    @printf "[%s] f_eval=%4d g_eval=%4d t=%5.1fs loss=%.4f tr=%.1f%% te=%.1f%% |g|=%.4f\n" tag f_calls g_calls h.elapsed tr_loss (100tr_acc) (100te_acc) gn
end

# ----------------------------------------------------------------
# 4a. Adam / AMSGrad
# ----------------------------------------------------------------
function run_first_order!(rule, name; max_g_evals = 2000, eval_freq = 500)
    @info "=== $name ==="
    h, x = Hist(), deepcopy(ps_dev_namedtuple)
    opt_state = Optimisers.setup(rule, x)
    infinite_loader = Iterators.Stateful(Iterators.cycle(GPULoader(train_x_gpu, train_y_gpu, BATCH)))
    
    f_calls, g_calls = 0, 0
    while g_calls < max_g_evals
        (bx, by) = popfirst!(infinite_loader)

        CUDA.synchronize()
        t0 = time_ns()

        # Safely typed gradient computation
        loss_val, grads_tuple = Zygote.withgradient(x) do p
            ŷ, _ = Lux.apply(model, bx, p, st_dev)
            loss_fn(ŷ, by)
        end
        grads = grads_tuple[1]
        opt_state, x = Optimisers.update!(opt_state, x, grads)

        CUDA.synchronize()
        h.elapsed += (time_ns() - t0) / 1e9
        
        f_calls += 1; g_calls += 1

        if g_calls % eval_freq == 0 || g_calls == max_g_evals
            snap!(h, f_calls, g_calls, x, name, sqrt(recursive_sq_norm(grads)))
        end
    end
    return x, h
end

# ----------------------------------------------------------------
# 4b. Tadam
# ----------------------------------------------------------------
function run_tadam!(; max_g_evals = 2000, eval_freq = 500, η1 = 0.10f0, kwargs...)
    @info "=== Tadam ==="
    infinite_loader = Iterators.Stateful(Iterators.cycle(GPULoader(train_x_gpu, train_y_gpu, BATCH)))
    nlp = LuxNLPModel(model, copy(ps0_dev_ca), st_dev, infinite_loader, loss_fn)
    h = Hist()

    cb = (nlp, solver, stats) -> begin
        f_calls, g_calls = neval_obj(nlp), neval_grad(nlp)
        
        push!(h.hf_iters, stats.iter)
        push!(h.hf_delta, Float64(solver.Δ))
        
        acc_steps = sum(h.hf_delta[1:end-1] .!= h.hf_delta[2:end])
        push!(h.hf_rej, stats.iter > 0 ? (stats.iter - acc_steps) / stats.iter : 0.0)

        if g_calls > 0 && (g_calls % eval_freq == 0 || g_calls >= max_g_evals)
            snap!(h, f_calls, g_calls, solver.x, "Tadam", norm(solver.gx))
        end

        solver.step_accepted && minibatch_next_train!(nlp)

        if stats.status == :small_step
            ng = norm(solver.gx)
            solver.Δ = max(ng / (2^round(log2(ng + 1f0))), 1f-5)
            stats.status = :unknown
        end
        
        g_calls >= max_g_evals && (stats.status = :user)
    end

    CUDA.synchronize(); t0 = time_ns()
    
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
# 5. Run & Plot
# ----------------------------------------------------------------
const EVAL_FREQ, MAX_G_EVALS, LR = 500, 2000, 3f-4

_, h_adam    = run_first_order!(Optimisers.Adam(LR), "Adam"; max_g_evals = MAX_G_EVALS, eval_freq = EVAL_FREQ)
_, h_amsgrad = run_first_order!(Optimisers.AMSGrad(LR), "AMSGrad"; max_g_evals = MAX_G_EVALS, eval_freq = EVAL_FREQ)
_, h_tadam   = run_tadam!(; max_g_evals = MAX_G_EVALS, eval_freq = EVAL_FREQ, η1 = 0.0001f0, η2 = 0.85f0, γ1 = 0.80f0, γ2 = 1.20f0, γ3 = 0.02f0, β1 = 0.90f0, β2 = 0.99f0, ϵ_v = 1f-7, θ1 = 1f-6)

pub_theme = Theme(fontsize = 14, Axis = (spinewidth = 0.9, xgridcolor = (:black, 0.08), ygridcolor = (:black, 0.08), xgridwidth = 0.6, ygridwidth = 0.6), Legend = (framevisible = false, labelsize = 11, patchsize = (22, 2)), Lines = (linewidth = 2.3,))
with_theme(pub_theme) do
    fig = Figure(size = (900, 1150))
    add_curves!(ax, x, y) = [lines!(ax, getfield(h, x), getfield(h, y); color=c, label=l) for (h, l, c) in [(h_adam, "Adam", RGBf(0.9, 0.6, 0.0)), (h_amsgrad, "AMSGrad", RGBf(0.8, 0.4, 0.0)), (h_tadam, "Tadam", RGBf(0.0, 0.4, 0.7))]]

    ax_a = Axis(fig[1,1]; xlabel = "Gradient Evaluations", ylabel = "Train loss", yscale = log10, title = "(a) Train loss vs. Budget"); add_curves!(ax_a, :g_calls, :tr_loss); axislegend(ax_a; position = :rt)
    ax_b = Axis(fig[1,2]; xlabel = "Wall-clock time (s)", ylabel = "Train loss", yscale = log10, title = "(b) Train loss vs. Time"); add_curves!(ax_b, :times, :tr_loss)
    ax_c = Axis(fig[2,1]; xlabel = "Gradient Evaluations", ylabel = "Test accuracy", title = "(c) Test accuracy vs. Budget"); add_curves!(ax_c, :g_calls, :te_acc); axislegend(ax_c; position = :rb)
    ax_d = Axis(fig[2,2]; xlabel = "Wall-clock time (s)", ylabel = "Test accuracy", title = "(d) Test accuracy vs. Time"); add_curves!(ax_d, :times, :te_acc)
    
    ax_e = Axis(fig[3,1]; xlabel = "Iteration k", ylabel = "Trust-region radius Δk", yscale = log10, title = "(e) Adaptive step size (Tadam)"); lines!(ax_e, h_tadam.hf_iters, h_tadam.hf_delta; color = RGBf(0.0, 0.4, 0.7), linewidth = 1.4)
    ax_f = Axis(fig[3,2]; xlabel = "Iteration k", ylabel = "Cumulative rejection rate", limits = (nothing, (0.0, 1.0)), title = "(f) TR rejection rate (Tadam)"); lines!(ax_f, h_tadam.hf_iters, h_tadam.hf_rej; color = RGBf(0.0, 0.4, 0.7), linewidth = 2.0)
    ax_g = Axis(fig[4, 1:2]; xlabel = "Gradient Evaluations", ylabel = "Minibatch Gradient Norm", yscale = log10, title = "(g) Gradient Norm ||g||"); add_curves!(ax_g, :g_calls, :gnorm)

    save("cifar10_Tadam_results_Corrected.pdf", fig)
end