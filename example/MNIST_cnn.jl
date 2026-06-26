using Pkg
Pkg.activate("lux_test_env")
# Uncomment and run the following line once to install dependencies to the test environment:
# Pkg.add(["Lux", "ComponentArrays", "NLPModels", "JSOSolvers", "MLDatasets", "MLUtils", "OneHotArrays", "NNlib"])
# Pkg.add("LuxCUDA")

# Load modules sequentially

# Map the local package into this environment. 
# Adjust the path to wherever you saved the LuxNLPModels folder.
Pkg.develop(path=".")

# # Force CUDA.jl to download and link strictly against the 13.0 runtime
# CUDA.set_runtime_version!(v"13.0")

# Pkg.build("CUDA")
using CUDA
using Lux
using LuxCUDA
# using CUDA
using LuxNLPModels
using ComponentArrays, Random, NNlib
using NLPModels, JSOSolvers
using MLDatasets, OneHotArrays, MLUtils
using Statistics: mean
using LinearAlgebra: norm
using Printf

# Fetch the active GPU device manager #TODO check on atlas
if CUDA.functional()
    println("CUDA detected.")
    const dev = gpu_device()
else
    println("CUDA not functional, defaulting to CPU.")
    const dev = cpu_device()
end

# ====================================================================
# 1. Data Pipeline (Pre-loaded to VRAM)
# ====================================================================
function create_gpu_dataloaders(batchsize=1024)
    train_data = MLDatasets.MNIST(Float32, split=:train)
    test_data  = MLDatasets.MNIST(Float32, split=:test)
    
    # Extract and reshape for MLP (784, Batch)
    xtrain_cpu = reshape(train_data.features[:, :, 1:5000], 784, 5000)
    ytrain_cpu = Float32.(onehotbatch(train_data.targets[1:5000], 0:9))
    
    xtest_cpu = reshape(test_data.features[:, :, 1:1000], 784, 1000)
    ytest_cpu = Float32.(onehotbatch(test_data.targets[1:1000], 0:9))
    
    # Move the entire dataset to the GPU *before* batching.
    # This guarantees the DataLoader yields CuArrays directly to the solver.
    xtrain_gpu = xtrain_cpu |> dev
    ytrain_gpu = ytrain_cpu |> dev
    xtest_gpu  = xtest_cpu  |> dev
    ytest_gpu  = ytest_cpu  |> dev
    
    train_loader = DataLoader((xtrain_gpu, ytrain_gpu); batchsize, shuffle=true)
    test_loader  = DataLoader((xtest_gpu, ytest_gpu); batchsize, shuffle=false)
    
    return train_loader, test_loader
end

println("Loading Data to VRAM...")
train_loader, test_loader = create_gpu_dataloaders(512)

# ====================================================================
# 2. Model & Objective Setup (GPU)
# ====================================================================

model = Chain(
    Dense(784 => 256, relu),
    Dense(256 => 128, relu),
    Dense(128 => 84, relu),
    Dense(84 => 10)
)

rng = Xoshiro(0)
ps, st = Lux.setup(rng, model)

# Migrate states to VRAM
st_gpu = st |> dev

# ComponentArray on CPU first; LuxNLPModel will handle the GPU transfer internally.
# Do NOT pre-migrate ps_cv to GPU: the constructor copies x0 to CPU regardless,
# and passing a GPU ComponentVector is fine (Array(getdata(ps)) handles both cases).
ps_cv = ComponentArray(ps)

function lossfn(y_pred, y_true)
    log_probs = NNlib.logsoftmax(y_pred; dims=1)
    return mean(-sum(y_true .* log_probs; dims=1))
end

# Pass dev= so that obj/grad!/objgrad! move parameters to the correct device.
# meta.x0 is always a plain CPU Vector{Float32}.
nlp = LuxNLPModel(model, ps_cv, st_gpu, train_loader, lossfn; dev)

# ====================================================================
# 3. Hardware Verification & Derivative Checks
# ====================================================================
println("\n--- Hardware & Model Summary ---")
x0 = nlp.meta.x0   # always a plain CPU Vector{Float32}

# x0 is intentionally on CPU; GPU execution is handled inside obj/grad!.
println("x0 type (solver-side):  ", typeof(x0))
println("Model device:           ", nlp.dev)

println("\nTesting Derivative Kernels...")
println("f(x0):    ", obj(nlp, x0))

g = similar(x0)
grad!(nlp, x0, g)
println("|g(x0)|:  ", norm(g))

# Generate a normalized random projection vector on CPU (solver-side)
v = randn(rng, Float32, length(x0))
v ./= norm(v)
hv = similar(x0)
hprod!(nlp, x0, v, hv)
println("|Hv|:     ", norm(hv))
println("--------------------------------\n")

# # ====================================================================
# # 4. Second-Order Optimization (JSOSolvers.R2)
# # ====================================================================
# iteration_callback = (nlp, solver, stats) -> begin
#     LuxNLPModels.minibatch_next_train!(nlp)
#     return false 
# end

# println("Starting JSOSolvers.R2 on GPU...")
# solver_stats = JSOSolvers.R2(nlp, callback=iteration_callback, max_time=120.0)

# # ====================================================================
# # 5. Evaluation
# # ====================================================================
# # The solution returned by the solver is already on the GPU
# optimized_ps = ComponentArray(solver_stats.solution, getaxes(ps_cv))

# function evaluate_accuracy(model, ps, st, dataloader)
#     correct, total = 0, 0
#     st_test = Lux.testmode(st) 
#     for (x, y) in dataloader
#         # Both x and ps are on the GPU, execution remains in VRAM
#         y_pred, _ = model(x, ps, st_test)
        
#         # Pull predictions back to CPU for accuracy math to prevent synchronization stalls
#         pred_cpu = Array(onecold(y_pred))
#         true_cpu = Array(onecold(y))
        
#         correct += sum(pred_cpu .== true_cpu)
#         total += length(true_cpu)
#     end
#     return (correct / total) * 100
# end

# @printf("\nFinal Train Accuracy: %.2f%%\n", evaluate_accuracy(model, optimized_ps, st_gpu, train_loader))
# @printf("Final Test Accuracy:  %.2f%%\n", evaluate_accuracy(model, optimized_ps, st_gpu, test_loader))