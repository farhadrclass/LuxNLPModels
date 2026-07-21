# ================================================================
# cifar_10_Resnet18.jl
# CIFAR-10 Benchmark: AdamW | AMSGrad | TRAdam
#
# Key upgrades:
#   - Multi-seed execution (Mean ± 1 Std Dev plotting)
#   - Encapsulated model states to prevent cross-seed contamination
#   - AdamW and AMSGrad updated to 1e-3 with Weight Decay
#   - VGG-style CNN explicitly documented
# ================================================================

# using Pkg
# Pkg.activate("lux_test_env_2")

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

# ----------------------------------------------------------------
# 0.  GPU / Device
# ----------------------------------------------------------------
const USE_GPU = CUDA.functional()
USE_GPU && @info "CUDA GPU: $(CUDA.name(CUDA.device()))"
USE_GPU || @warn "No CUDA GPU found — falling back to CPU (slower)"

to_dev(x) = USE_GPU ? CUDA.cu(x) : x

# ----------------------------------------------------------------
# 1.  CIFAR-10  (32 × 32 × 3, 10 classes, Float32 ∈ [0,1])
# ----------------------------------------------------------------
@info "Loading CIFAR-10..."
tr_x_raw, tr_y_raw = CIFAR10.traindata(Float32)
te_x_raw, te_y_raw = CIFAR10.testdata(Float32)

N_TRAIN, N_TEST = 20_000, 2_000 # Increase to 50_000 for final paper

train_x_cpu = tr_x_raw[:, :, :, 1:N_TRAIN]
train_y_cpu = onehotbatch(tr_y_raw[1:N_TRAIN], 0:9)
test_x_cpu  = te_x_raw[:, :, :, 1:N_TEST]
test_y_cpu  = onehotbatch(te_y_raw[1:N_TEST], 0:9)

const BATCH = 256
data_loader = DataLoader(
    (to_dev(train_x_cpu), to_dev(train_y_cpu));
    batchsize = BATCH, shuffle = true,
)

# ----------------------------------------------------------------
# 2.  VGG-Style CNN Generator & Evaluator
# ----------------------------------------------------------------
function create_model_and_state(seed::Int)
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
    
    rng = MersenneTwister(seed)
    ps_cpu, st = Lux.setup(rng, model)
    ps_template = ComponentArray(ps_cpu) 
    ps0_dev = to_dev(copy(ps_template))
    st_dev = to_dev(st)
    
    return model, ps_template, ps0_dev, st_dev
end

loss_fn(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))

function eval_metrics(ps_vec, model, st, ps_template)
    ps_s = ComponentArray(Array(ps_vec), getaxes(ps_template))
    ŷ_tr, _ = Lux.apply(model, train_x_cpu, ps_s, st) # Run on CPU to save VRAM
    ŷ_te, _ = Lux.apply(model, test_x_cpu,  ps_s, st)
    
    tr_loss  = Float32(loss_fn(ŷ_tr, train_y_cpu))
    acc(ŷ, y) = mean([i.I[1] for i in argmax(ŷ; dims=1)] .==
                     [i.I[1] for i in argmax(y; dims=1)])
                     
    return Float32(tr_loss), Float32(acc(ŷ_tr, train_y_cpu)), Float32(acc(ŷ_te, test_y_cpu))
end

function make_nlp(model, ps0_dev, st_dev)
    LuxNLPModel(model, copy(ps0_dev), st_dev, data_loader, loss_fn)
end

# ----------------------------------------------------------------
# 3.  History Struct
# ----------------------------------------------------------------
mutable struct Hist
    iters    :: Vector{Int}
    batches  :: Vector{Int}
    times    :: Vector{Float64}
    tr_loss  :: Vector{Float32}
    tr_acc   :: Vector{Float32}
    te_acc   :: Vector{Float32}
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

function snap!(h::Hist, iter, batches, ps_vec, model, st, ps_template, tag)
    h._t_accum += (time_ns() - h._t_start) / 1e9
    tr_loss, tr_acc, te_acc = eval_metrics(ps_vec, model, st, ps_template)
    push!(h.iters, iter); push!(h.batches, batches); push!(h.times, h._t_accum)
    push!(h.tr_loss, tr_loss); push!(h.tr_acc, tr_acc); push!(h.te_acc, te_acc)
    @printf "[%s] iter=%4d  bat=%4d  t=%5.0fs  loss=%.4f  tr=%.1f%%  te=%.1f%%\n" tag iter batches h._t_accum tr_loss (100tr_acc) (100te_acc)
    h._t_start = time_ns()
end

# ----------------------------------------------------------------
# 4a. Runner: First-order (AdamW, AMSGrad)
# ----------------------------------------------------------------
function run_first_order!(rule, name, model, ps0_dev, st_dev, ps_template; max_iter=2000, eval_freq=50)
    @info "=== $name ==="
    nlp     = make_nlp(model, ps0_dev, st_dev)
    x       = copy(nlp.meta.x0)
    g       = similar(x)
    h       = Hist()
    batches = 0
    opt     = Optimisers.setup(rule, x)
    
    h._t_start = time_ns()
    for i in 0:max_iter
        i % eval_freq == 0 && snap!(h, i, batches, x, model, st_dev, ps_template, name)
        i == max_iter && break
        objgrad!(nlp, x, g)
        opt, x = Optimisers.update(opt, x, g)
        minibatch_next_train!(nlp)
        batches += 1
    end
    return h
end

# ----------------------------------------------------------------
# 4b. Runner: TRAdam
# ----------------------------------------------------------------
function run_tadam!(model, ps0_dev, st_dev, ps_template; max_iter=2000, eval_freq=50, η1=0.10f0, kwargs...)
    @info "=== TRAdam ==="
    nlp     = make_nlp(model, ps0_dev, st_dev)
    h       = Hist()
    batches = Ref(0)
    success = Ref(0)
    
    h._t_start = time_ns()
    cb = (nlp, solver, stats) -> begin
        iter = stats.iter
        
        # High-frequency logging
        push!(h.hf_iters, iter)
        push!(h.hf_delta, Float64(solver.Δ))
        total = h.n_ok + h.n_rej
        push!(h.hf_rej, total > 0 ? h.n_rej / total : 0.0)

        # Snapshot
        iter % eval_freq == 0 && snap!(h, iter, batches[], solver.x, model, st_dev, ps_template, "TRAdam")

        # Step acceptance
        if solver.step_accepted
            h.n_ok += 1; success[] += 1
            h._t_accum += (time_ns() - h._t_start) / 1e9
            minibatch_next_train!(nlp)
            batches[] += 1; success[] = 0
            stats.objective = obj(nlp, solver.x)
            h._t_start = time_ns()
        elseif iter > 0
            h.n_rej += 1
        end

        # Stall-prevention heuristic
        if stats.status == :small_step
            ng = norm(solver.gx)
            solver.Δ = max(ng / (2^round(log2(ng + 1f0))), 1f-5)
            stats.status = :unknown
        end
    end

    stats = tadam(nlp; max_iter, atol=1f-8, rtol=1f-5, callback=cb, verbose=0, η1, kwargs...)
    h.iters[end] != stats.iter && snap!(h, stats.iter, batches[], stats.solution, model, st_dev, ps_template, "TRAdam")

    @printf "  TRAdam final: %d accepted | %d rejected | %.1f%% rej rate\n" h.n_ok h.n_rej (100 * h.n_rej / max(1, h.n_ok + h.n_rej))
    return h
end

# ----------------------------------------------------------------
# 5.  Execution Loop over Random Seeds
# ----------------------------------------------------------------
const MAX_ITER  = 2000
const EVAL_FREQ = 50
const LR        = 1f-3   # Standard CIFAR-10 learning rate
const WD        = 1f-4   # Weight decay

SEEDS = [42, 123, 456]

hists_adamw   = Hist[]
hists_amsgrad = Hist[]
hists_tadam   = Hist[]

for seed in SEEDS
    @info "--- Starting Evaluation for Seed: $seed ---"
    
    model, ps_template, ps0_dev, st_dev = create_model_and_state(seed)
    rule_adamw = OptimiserChain(Optimisers.WeightDecay(WD), Optimisers.Adam(LR))
    push!(hists_adamw, run_first_order!(rule_adamw, "AdamW", model, ps0_dev, st_dev, ps_template; max_iter=MAX_ITER, eval_freq=EVAL_FREQ))
    
    model, ps_template, ps0_dev, st_dev = create_model_and_state(seed)
    rule_amsgrad = OptimiserChain(Optimisers.WeightDecay(WD), Optimisers.AMSGrad(LR))
    push!(hists_amsgrad, run_first_order!(rule_amsgrad, "AMSGrad", model, ps0_dev, st_dev, ps_template; max_iter=MAX_ITER, eval_freq=EVAL_FREQ))
    
    model, ps_template, ps0_dev, st_dev = create_model_and_state(seed)
    push!(hists_tadam, run_tadam!(model, ps0_dev, st_dev, ps_template; 
        max_iter=MAX_ITER, eval_freq=EVAL_FREQ,
        η1=0.10f0, η2=0.55f0, γ1=0.50f0, γ2=2.00f0, γ3=0.02f0,
        β1=0.90f0, β2=0.99f0, ϵ_v=1f-7, θ2=1.00f0
    ))
end

# ----------------------------------------------------------------
# 6.  Publication Figure (Mean ± 1 Std Dev)
# ----------------------------------------------------------------
const C_ADAMW   = RGBf(0.902, 0.624, 0.000) # Orange
const C_AMSGRAD = RGBf(0.835, 0.369, 0.000) # Vermilion
const C_TADAM   = RGBf(0.000, 0.447, 0.698) # Blue

pub_theme = Theme(
    fontsize = 14,
    Axis = (spinewidth=0.9, xgridcolor=(:black, 0.08), ygridcolor=(:black, 0.08),
            xgridwidth=0.6, ygridwidth=0.6, titlesize=13),
    Legend = (framevisible=false, labelsize=11, patchsize=(22, 2)),
)

with_theme(pub_theme) do
    fig = Figure(size = (900, 940))

    function plot_with_std!(ax, xfield, yfield, hists_list, color, label)
        # Assuming all runs have the exact same x-axis checkpoints
        x_vals = getfield(hists_list[1], xfield)
        y_mat = hcat([getfield(h, yfield) for h in hists_list]...) 
        
        y_mean = vec(mean(y_mat, dims=2))
        y_std  = vec(std(y_mat, dims=2))
        
        band!(ax, x_vals, y_mean .- y_std, y_mean .+ y_std; color=(color, 0.2))
        lines!(ax, x_vals, y_mean; color=color, label=label, linewidth=2.3)
    end

    # (a) Train loss vs Eval
    ax_a = Axis(fig[1,1]; xlabel="Minibatches", ylabel="Train loss", yscale=log10, title="(a) Train loss vs. iterations")
    plot_with_std!(ax_a, :batches, :tr_loss, hists_adamw,   C_ADAMW,   "AdamW")
    plot_with_std!(ax_a, :batches, :tr_loss, hists_amsgrad, C_AMSGRAD, "AMSGrad")
    plot_with_std!(ax_a, :batches, :tr_loss, hists_tadam,   C_TADAM,   "TRAdam")
    axislegend(ax_a; position=:rt)

    # (b) Train loss vs Time
    ax_b = Axis(fig[1,2]; xlabel="Wall-clock time (s)", ylabel="Train loss", yscale=log10, title="(b) Train loss vs. time")
    plot_with_std!(ax_b, :times, :tr_loss, hists_adamw,   C_ADAMW,   "AdamW")
    plot_with_std!(ax_b, :times, :tr_loss, hists_amsgrad, C_AMSGRAD, "AMSGrad")
    plot_with_std!(ax_b, :times, :tr_loss, hists_tadam,   C_TADAM,   "TRAdam")

    # (c) Test Acc vs Eval
    ax_c = Axis(fig[2,1]; xlabel="Minibatches", ylabel="Test accuracy", title="(c) Test accuracy vs. iterations")
    plot_with_std!(ax_c, :batches, :te_acc, hists_adamw,   C_ADAMW,   "AdamW")
    plot_with_std!(ax_c, :batches, :te_acc, hists_amsgrad, C_AMSGRAD, "AMSGrad")
    plot_with_std!(ax_c, :batches, :te_acc, hists_tadam,   C_TADAM,   "TRAdam")
    axislegend(ax_c; position=:rb)

    # (d) Test Acc vs Time
    ax_d = Axis(fig[2,2]; xlabel="Wall-clock time (s)", ylabel="Test accuracy", title="(d) Test accuracy vs. time")
    plot_with_std!(ax_d, :times, :te_acc, hists_adamw,   C_ADAMW,   "AdamW")
    plot_with_std!(ax_d, :times, :te_acc, hists_amsgrad, C_AMSGRAD, "AMSGrad")
    plot_with_std!(ax_d, :times, :te_acc, hists_tadam,   C_TADAM,   "TRAdam")

    # (e & f) High-Frequency Diagnostics (Only plot the first seed for clarity)
    ax_e = Axis(fig[3,1]; xlabel="Iteration k", ylabel="Radius Δk", yscale=log10, title="(e) Adaptive step size (Seed 1)")
    lines!(ax_e, hists_tadam[1].hf_iters, hists_tadam[1].hf_delta; color=C_TADAM, linewidth=1.4)

    ax_f = Axis(fig[3,2]; xlabel="Iteration k", ylabel="Cumulative rejection rate", limits=(nothing, (0.0, 1.0)), title="(f) Rejection rate (Seed 1)")
    lines!(ax_f, hists_tadam[1].hf_iters, hists_tadam[1].hf_rej; color=C_TADAM, linewidth=2.0)

    Label(fig[4, :], "CIFAR-10 (N_train=$(N_TRAIN), N_test=$(N_TEST), batch=$(BATCH)). AdamW & AMSGrad use lr=$(LR), wd=$(WD).\nCurves denote mean ± 1 std dev across 3 random seeds.", tellwidth=false, fontsize=11, color=:gray40)

    save("cifar10_tradam_results.pdf", fig)
    display(fig)
    @info "Figures written: cifar10_tradam_results.pdf"
end