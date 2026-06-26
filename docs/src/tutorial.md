# LuxNLPModels.jl Tutorial

This tutorial assumes basic familiarity with
[Julia](https://julialang.org/),
[Lux.jl](https://lux.csail.mit.edu/stable/), and
[NLPModels.jl](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/).

We fit a small MLP to noisy quadratic data, then compare three
JSOSolvers optimisers: plain SGD (via the NLPModels interface), R2, and L-BFGS.

## Packages

```@example tutorial
using Lux, ComponentArrays, MLUtils
using NLPModels, JSOSolvers
using LuxNLPModels
using Random, Statistics
```

## Data

We generate 128 samples from $y = x^2 - 2x$ with additive Gaussian noise.

```@example tutorial
rng = Random.MersenneTwister(42)

x_data = reshape(collect(range(-2f0, 2f0, 128)), 1, 128)
y_data = evalpoly.(x_data, Ref((0f0, -2f0, 1f0)))
y_data .+= randn(rng, Float32, size(y_data)) .* 0.1f0

# Full-batch DataLoader (single batch = full-batch gradient descent)
loader = DataLoader((x_data, y_data); batchsize=128, shuffle=false)
```

## Model

```@example tutorial
model = Chain(Dense(1 => 16, relu), Dense(16 => 1))
ps, st = Lux.setup(rng, model)
ps_cv  = ComponentArray(ps)   # flat parameter vector with ComponentArray axes
```

## Loss and NLPModel wrapper

The loss function must have the signature `(ŷ, y) -> scalar`.

```@example tutorial
loss_fn(ŷ, y) = mean(abs2, ŷ .- y)   # MSE

# Each solver gets a fresh model from the same starting point
make_nlp() = LuxNLPModel(model, ps_cv, st, loader, loss_fn)

nlp = make_nlp()
println("Parameters : ", nlp.meta.nvar)
println("Loss (x₀)  : ", obj(nlp, nlp.meta.x0))
```

`nlp.meta.x0` is always a plain `Vector{Float32}` on CPU so that JSOSolvers
can allocate its internal buffers correctly.

## Evaluating gradients

```@example tutorial
g = similar(nlp.meta.x0)
grad!(nlp, nlp.meta.x0, g)
println("‖∇f(x₀)‖ = ", sqrt(sum(abs2, g)))
```

Use `objgrad!` to compute both in a single AD pass (recommended inside solvers):

```@example tutorial
f, g = objgrad!(nlp, nlp.meta.x0, g)
```

## Solving with L-BFGS

```@example tutorial
lbfgs_hist = Float32[]
nlp_lbfgs  = make_nlp()

stats = lbfgs(
    nlp_lbfgs;
    max_iter = 300,
    atol     = 1f-6,
    callback = (nlp, solver, stats) -> push!(lbfgs_hist, stats.objective),
)

println("Status     : ", stats.status)
println("Final loss : ", stats.objective)
println("Iterations : ", stats.iter)
```

## Solving with R2

```@example tutorial
r2_hist = Float32[]
nlp_r2  = make_nlp()

stats_r2 = R2(
    nlp_r2;
    max_iter = 800,
    atol     = 1f-6,
    rtol     = 0f0,
    callback = (nlp, solver, stats) -> push!(r2_hist, stats.objective),
)
println("R2 final loss : ", stats_r2.objective)
```

## Mini-batch training

For mini-batch training call `minibatch_next_train!` in the callback to
advance to the next batch after each solver iteration:

```julia
loader_mb = DataLoader((x_data, y_data); batchsize=32, shuffle=true)
nlp_mb    = LuxNLPModel(model, ps_cv, st, loader_mb, loss_fn)

stats = R2(
    nlp_mb;
    max_iter = 1000,
    callback = (nlp, solver, stats) -> minibatch_next_train!(nlp),
)
```

## Inference

Reconstruct the optimised `ComponentVector` from the flat solution and run
the model in test mode:

```@example tutorial
opt_ps  = ComponentArray(stats.solution, getaxes(ps_cv))
st_test = Lux.testmode(st)
y_pred, _ = Lux.apply(model, x_data, opt_ps, st_test)
println("Prediction range: ", extrema(y_pred))
```

## GPU usage

Move data and state to the GPU before construction.
The solver's parameter vector stays on CPU automatically.

```julia
using LuxCUDA
dev = gpu_device()

x_gpu  = x_data |> dev
y_gpu  = y_data |> dev
st_gpu = st     |> dev

loader_gpu = DataLoader((x_gpu, y_gpu); batchsize=128, shuffle=false)

nlp_gpu = LuxNLPModel(model, ps_cv, st_gpu, loader_gpu, loss_fn; dev)

stats_gpu = lbfgs(nlp_gpu; max_iter=300, atol=1f-6)
```



We have aligned this tutorial to [MLP_MNIST](https://github.com/FluxML/model-zoo/blob/master/vision/mlp_mnist/mlp_mnist.jl) example and reused some of their functions.

### What we cover in this tutorial

We will cover the following:

- Define a Neural Network (NN) Model in Flux, 
  - Fully connected model
- Define or set the loss function
- Data loading
  - MNIST 
  - Divide the data into train and test
- Define a method for calculating accuracy and loss
- Transfer the NN model to FluxNLPModel 
- Using FluxNLPModels and access 
  - Gradient of current weight
  - Objective (or loss) evaluated at current weights 


### Packages needed
```@example FluxNLPModel
using FluxNLPModels
using Flux, NLPModels
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using MLDatasets
using JSOSolvers
```

### Setting Neural Network (NN) Model

First, a NN model needs to be define in Flux.jl.
Our model is very simple: It consists of one "hidden layer" with 32 "neurons", each connected to every input pixel. Each neuron has a sigmoid nonlinearity and is connected to every "neuron" in the output layer. Finally, softmax produces probabilities, i.e., positive numbers that add up to 1.

One can create a method that returns the model. This method can encapsulate the specific architecture and parameters of the model, making it easier to reuse and manage. It provides a convenient way to define and initialize the model when needed.

```@example FluxNLPModel
function build_model(; imgsize = (28, 28, 1), nclasses = 10)
  return Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses)) 
end
```

### Loss function

We can define any loss function that we need, here we use Flux build-in logitcrossentropy function. 
```@example FluxNLPModel
## Loss function
const loss = Flux.logitcrossentropy
```

### Load datasets and define minibatch 
In this section, we will cover the process of loading datasets and defining minibatches for training your model using Flux. Loading and preprocessing data is an essential step in machine learning, as it allows you to train your model on real-world examples.

We will specifically focus on loading the MNIST dataset. We will divide the data into training and testing sets, ensuring that we have separate data for model training and evaluation.

Additionally, we will define minibatches, which are subsets of the dataset that are used during the training process. Minibatches enable efficient training by processing a small batch of examples at a time, instead of the entire dataset. This technique helps in managing memory resources and improving convergence speed.

```@example FluxNLPModel
function getdata(bs)
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

  # Loading Dataset	
  xtrain, ytrain = MLDatasets.MNIST(Tx = Float32, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = Float32, split = :test)[:]

  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)

  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

  # Create DataLoaders (mini-batch iterators)
  train_loader = DataLoader((xtrain, ytrain), batchsize = bs, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = bs)

  return train_loader, test_loader
end
```

### Transfering to FluxNLPModels

```@example FluxNLPModel
  device = cpu
  train_loader, test_loader = getdata(128)

  ## Construct model
  model = build_model() |> device

  # now we set the model to FluxNLPModel
  nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)
```

## Tools associated with a FluxNLPModel
The problem dimension `n`, where `w` ∈ ℝⁿ:
```@example FluxNLPModel
n = nlp.meta.nvar
```

### Get the current network weights:
```@example FluxNLPModel
w = nlp.w
```

### Evaluate the loss function (i.e. the objective function) at `w`:
```@example FluxNLPModel
using NLPModels
NLPModels.obj(nlp, w)
```
The length of `w` must be `nlp.meta.nvar`.

### Evaluate the gradient at `w`:
```@example FluxNLPModel
g = similar(w)
NLPModels.grad!(nlp, w, g)
```

## Train a neural network with JSOSolvers.R2

```@example FluxNLPModel
max_time = 60. # run at most 1min
callback = (nlp, 
            solver, 
            stats) -> FluxNLPModels.minibatch_next_train!(nlp)

solver_stats = R2(nlp; callback, max_time)
test_accuracy = FluxNLPModels.accuracy(nlp) #check the accuracy
```