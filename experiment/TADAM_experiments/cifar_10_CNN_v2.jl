# ================================================================
# cifar_10_CNN_v2.jl
# CIFAR-10 CNN Benchmark on GPU:  Adam  vs  AMSGrad
#
# Uses Lux + LuxCUDA. The entire dataset is pushed to VRAM once and
# every mini-batch, parameter vector, and gradient stays on the GPU
# for the whole run (no host <-> device copies in the hot loop).
#
# Optimisation is driven through the repo's `LuxNLPModel` wrapper so
# that Adam / AMSGrad (from Optimisers.jl) consume the exact same
# `objgrad!` gradients as the trust-region solvers do.
# ================================================================

using Pkg
Pkg.activate("lux_test_env")
# Run once to install dependencies into the test environment:
# Pkg.add(["Lux", "ComponentArrays", "NLPModels", "JSOSolvers", "MLDatasets",
#          "MLUtils", "OneHotArrays", "NNlib", "Optimisers", "LuxCUDA", "CUDA"])
Pkg.develop(path=".")

using CUDA
using Lux
using LuxCUDA
using LuxNLPModels
using ComponentArrays
using NLPModels, JSOSolvers
using Optimisers
using MLDatasets, OneHotArrays, MLUtils
using NNlib: logsoftmax
using Random, Statistics, Printf, LinearAlgebra
using CairoMakie
import CairoMakie: Axis

# ----------------------------------------------------------------
# 0.  Device selection
# ----------------------------------------------------------------
if CUDA.functional()
    @info "CUDA detected — training on GPU."
    const dev = gpu_device()
else
    @warn "CUDA not functional — falling back to CPU."
    const dev = cpu_device()
end

# ----------------------------------------------------------------
# 1.  CIFAR-10 data → VRAM
#     MLDatasets returns features as (32, 32, 3, N) Float32 in [0,1].
# ----------------------------------------------------------------
@info "Loading CIFAR-10..."
const N_TRAIN = 10_000
const N_TEST  = 2_000
const BATCH   = 512

train_set = MLDatasets.CIFAR10(Float32, split=:train)
test_set  = MLDatasets.CIFAR10(Float32, split=:test)

# CIFAR-10 channel-wise normalisation (standard mean/std).
const CIFAR_MEAN = reshape(Float32[0.4914, 0.4822, 0.4465], 1, 1, 3, 1)
const CIFAR_STD  = reshape(Float32[0.2470, 0.2435, 0.2616], 1, 1, 3, 1)
normalise(x) = (x .- CIFAR_MEAN) ./ CIFAR_STD

train_x_cpu = normalise(train_set.features[:, :, :, 1:N_TRAIN])
train_y_cpu = Float32.(onehotbatch(train_set.targets[1:N_TRAIN], 0:9))
test_x_cpu  = normalise(test_set.features[:, :, :, 1:N_TEST])
test_y_cpu  = Float32.(onehotbatch(test_set.targets[1:N_TEST], 0:9))

# Push everything to the device *before* batching so the DataLoader
# yields CuArrays directly to the solver.
train_x = train_x_cpu |> dev
train_y = train_y_cpu |> dev
test_x  = test_x_cpu  |> dev
test_y  = test_y_cpu  |> dev

data_loader = DataLoader((train_x, train_y); batchsize=BATCH, shuffle=true)

# ----------------------------------------------------------------
# 2.  CNN model  (32×32×3 → 10)
# ----------------------------------------------------------------
model = Chain(
    Conv((3, 3), 3  => 32, relu; pad=SamePad()),
    Conv((3, 3), 32 => 32, relu; pad=SamePad()),
    MaxPool((2, 2)),                              # 16×16
    Conv((3, 3), 32 => 64, relu; pad=SamePad()),
    Conv((3, 3), 64 => 64, relu; pad=SamePad()),
    MaxPool((2, 2)),                              # 8×8
    FlattenLayer(),
    Dense(8 * 8 * 64 => 256, relu),
    Dense(256 => 10),
)

rng = Xoshiro(0)
ps, st = Lux.setup(rng, model)

# States live on the device; parameters start as a CPU ComponentArray.
st_gpu = st |> dev
const ps_template = ComponentArray(ps)   # CPU axis template
const ps0         = copy(ps_template)     # shared starting point

loss_fn(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))

# `dev` is forwarded so meta.x0, gradients, and solver state live in VRAM.
make_nlp() = LuxNLPModel(model, copy(ps0), st_gpu, data_loader, loss_fn; dev)

@printf "Parameters : %d\n"   length(ps_template)
@printf "Device     : %s\n\n" string(dev)

# ----------------------------------------------------------------
# 3.  Evaluation (test-mode; predictions pulled to CPU for accuracy)
# ----------------------------------------------------------------
function eval_metrics(x_vec)
    ps_s   = ComponentArray(x_vec, getaxes(ps_template))   # stays on device
    st_ev  = Lux.testmode(st_gpu)
    ŷ_tr, _ = Lux.apply(model, train_x, ps_s, st_ev)
    ŷ_te, _ = Lux.apply(model, test_x,  ps_s, st_ev)
    tr_loss = Float32(loss_fn(ŷ_tr, train_y))
    acc(ŷ, y) = mean(Array(onecold(ŷ)) .== Array(onecold(y)))
    return tr_loss, Float32(acc(ŷ_tr, train_y)), Float32(acc(ŷ_te, test_y))
end

# ----------------------------------------------------------------
# 4.  History + first-order runner (Adam / AMSGrad)
# ----------------------------------------------------------------
mutable struct Hist
    iters   :: Vector{Int}
    times   :: Vector{Float64}
    tr_loss :: Vector{Float32}
    tr_acc  :: Vector{Float32}
    te_acc  :: Vector{Float32}
end
Hist() = Hist(Int[], Float64[], Float32[], Float32[], Float32[])

function run_first_order!(rule, name; max_iter=1000, eval_freq=50)
    @info "=== $name ==="
    nlp = make_nlp()
    x   = copy(nlp.meta.x0)          # CuArray when dev == gpu_device()
    g   = similar(x)
    h   = Hist()
    opt = Optimisers.setup(rule, x)

    t_accum = 0.0
    t_start = time_ns()
    for i in 0:max_iter
        if i % eval_freq == 0
            t_accum += (time_ns() - t_start) / 1e9
            trl, tra, tea = eval_metrics(x)
            push!(h.iters, i);   push!(h.times, t_accum)
            push!(h.tr_loss, trl); push!(h.tr_acc, tra); push!(h.te_acc, tea)
            @printf "[%-7s] iter=%4d  t=%5.1fs  loss=%.4f  tr=%.1f%%  te=%.1f%%\n"  name i t_accum trl 100tra 100tea
            t_start = time_ns()
        end
        i == max_iter && break
        objgrad!(nlp, x, g)
        opt, x = Optimisers.update(opt, x, g)
        minibatch_next_train!(nlp)   # advance to the next mini-batch
    end
    return x, h
end

# ----------------------------------------------------------------
# 4b.  Runner: TADAM  (JSOSolvers trust-region Adam)
#
#  Differences from the first-order loop:
#    - The solver owns the iteration; we hook into it via `callback`.
#    - `minibatch_next_train!` only advances on *accepted* steps, and
#      `stats.objective` is refreshed on the new batch so the ratio ρ_k
#      stays consistent (STORM-like mini-batch analysis).
#    - Stall guard: if the solver reports :small_step while ‖g‖ is not
#      negligible, reset Δ and continue instead of terminating early.
#    - Runs entirely on the GPU: solver.x / solver.gx are CuArrays and
#      hprod! is finite-difference based (no scalar indexing).
# ----------------------------------------------------------------
function run_tadam!(; max_iter=1000, eval_freq=50, kwargs...)
    @info "=== Tadam ==="
    nlp     = make_nlp()
    h       = Hist()
    n_ok    = Ref(0)
    n_rej   = Ref(0)
    t_accum = Ref(0.0)
    t_start = Ref(time_ns())

    cb = (nlp, solver, stats) -> begin
        iter = stats.iter

        # ── periodic evaluation (timer paused during eval) ──────────
        if iter % eval_freq == 0
            t_accum[] += (time_ns() - t_start[]) / 1e9
            trl, tra, tea = eval_metrics(solver.x)
            push!(h.iters, iter); push!(h.times, t_accum[])
            push!(h.tr_loss, trl); push!(h.tr_acc, tra); push!(h.te_acc, tea)
            @printf "[%-7s] iter=%4d  t=%5.1fs  loss=%.4f  tr=%.1f%%  te=%.1f%%\n"  "Tadam" iter t_accum[] trl 100tra 100tea
            t_start[] = time_ns()
        end

        # ── step acceptance / batch advance ─────────────────────────
        if solver.step_accepted
            n_ok[] += 1
            t_accum[] += (time_ns() - t_start[]) / 1e9   # pause during batch swap
            minibatch_next_train!(nlp)
            stats.objective = obj(nlp, solver.x)
            t_start[] = time_ns()
        elseif iter > 0
            n_rej[] += 1
        end

        # ── stall guard ─────────────────────────────────────────────
        if stats.status == :small_step
            ng = norm(solver.gx)
            solver.Δ = max(ng / (2^round(log2(ng + 1f0))), 1f-5)
            stats.status = :unknown
        end
    end

    stats = tadam(nlp; max_iter, atol=1f-5, rtol=1f-5, callback=cb, verbose=0, kwargs...)

    # Final snapshot if the last iteration was not on an eval boundary.
    if isempty(h.iters) || h.iters[end] != stats.iter
        trl, tra, tea = eval_metrics(stats.solution)
        push!(h.iters, stats.iter); push!(h.times, t_accum[])
        push!(h.tr_loss, trl); push!(h.tr_acc, tra); push!(h.te_acc, tea)
    end

    @printf "  Tadam: %d accepted | %d rejected | %.1f%% rejection rate\n"  n_ok[] n_rej[] (100 * n_rej[] / max(1, n_ok[] + n_rej[]))
    return stats.solution, h
end

# ----------------------------------------------------------------
# 5.  Train & compare
# ----------------------------------------------------------------
const MAX_ITER  = 1000
const EVAL_FREQ = 50
const LR        = 3f-4

_, h_adam    = run_first_order!(Optimisers.Adam(LR),    "Adam";    max_iter=MAX_ITER, eval_freq=EVAL_FREQ)
_, h_amsgrad = run_first_order!(Optimisers.AMSGrad(LR), "AMSGrad"; max_iter=MAX_ITER, eval_freq=EVAL_FREQ)
_, h_tadam   = run_tadam!(;
    max_iter  = MAX_ITER,
    eval_freq = EVAL_FREQ,
    # Trust-region acceptance thresholds (η1 < η2 ≤ 1)
    η1  = 1f-4,
    η2  = 0.90f0,
    # TR radius updates (γ1 contract on reject, γ2 expand on good step)
    γ1  = 0.80f0,
    γ2  = 1.20f0,
    γ3  = 0.02f0,
    # Adam momentum / RMS decay
    β1  = 0.90f0,
    β2  = 0.99f0,
    ϵ_v = 1f-7,
    # Deep-learning specific overrides
    θ1   = 1f-6,
    Δmax = 1f-2,
)

println("\n================  Final comparison  ================")
@printf "Adam    | train loss %.4f | train acc %.1f%% | test acc %.1f%%\n"  h_adam.tr_loss[end]    100h_adam.tr_acc[end]    100h_adam.te_acc[end]
@printf "AMSGrad | train loss %.4f | train acc %.1f%% | test acc %.1f%%\n"  h_amsgrad.tr_loss[end] 100h_amsgrad.tr_acc[end] 100h_amsgrad.te_acc[end]
@printf "Tadam   | train loss %.4f | train acc %.1f%% | test acc %.1f%%\n"  h_tadam.tr_loss[end]   100h_tadam.tr_acc[end]   100h_tadam.te_acc[end]

# ----------------------------------------------------------------
# 6.  Figure (4 panels, colorblind-safe, PDF + PNG)
# ----------------------------------------------------------------
const C_ADAM    = RGBf(0.902, 0.624, 0.000)   # orange
const C_AMSGRAD = RGBf(0.835, 0.369, 0.000)   # vermilion
const C_TADAM   = RGBf(0.000, 0.447, 0.698)   # blue

pub_theme = Theme(
    fontsize = 14,
    Axis = (
        spinewidth = 0.9,
        xgridcolor = (:black, 0.08), ygridcolor = (:black, 0.08),
        xgridwidth = 0.6,           ygridwidth = 0.6,
        titlesize = 13, xlabelsize = 12, ylabelsize = 12,
        xticklabelsize = 11, yticklabelsize = 11,
    ),
    Legend = (framevisible = false, labelsize = 11, patchsize = (22, 2)),
    Lines  = (linewidth = 2.3,),
)

with_theme(pub_theme) do
    fig = Figure(size = (900, 640))

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
        xlabel = "Iteration", ylabel = "Train loss", yscale = log10,
        title  = "(a) Train loss vs. iterations")
    ax_b = Axis(fig[1,2];
        xlabel = "Wall-clock time (s)", ylabel = "Train loss", yscale = log10,
        title  = "(b) Train loss vs. time")
    add_curves!(ax_a, :iters, :tr_loss); axislegend(ax_a; position = :rt)
    add_curves!(ax_b, :times, :tr_loss)

    # (c-d) Test accuracy
    ax_c = Axis(fig[2,1];
        xlabel = "Iteration", ylabel = "Test accuracy",
        title  = "(c) Test accuracy vs. iterations")
    ax_d = Axis(fig[2,2];
        xlabel = "Wall-clock time (s)", ylabel = "Test accuracy",
        title  = "(d) Test accuracy vs. time")
    add_curves!(ax_c, :iters, :te_acc); axislegend(ax_c; position = :rb)
    add_curves!(ax_d, :times, :te_acc)

    Label(fig[3, :],
        "CIFAR-10  (N_train=$(N_TRAIN), N_test=$(N_TEST), batch=$(BATCH)).  " *
        "CNN: 2×(3×3 conv) → pool → 2×(3×3 conv) → pool → 256 → 10.  " *
        "Adam & AMSGrad use lr=$(LR); Tadam adapts Δ automatically.",
        tellwidth = false, fontsize = 11, color = :gray40,
    )

    save("cifar_10_Tadam_results.pdf", fig)
    save("cifar_10_Tadam_results.png", fig; px_per_unit = 2)
    display(fig)
    @info "Figures written: cifar_10_Tadam_results.{pdf,png}"
end
