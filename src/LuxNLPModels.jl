module LuxNLPModels

using NLPModels
using Lux
using ComponentArrays
using Zygote
import ForwardDiff

export LuxNLPModel, minibatch_next_train!

include("types.jl")
include("data.jl")
include("objective.jl")
include("hessian.jl")

end