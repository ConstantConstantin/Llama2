module Llama2

using StatsBase: wsample
using LinearAlgebra: dot

export talktollm

include("structs.jl")
include("tokenizer.jl")
include("sampler.jl")
include("decode_transformer.jl")
include("forward.jl")
include("talk.jl")

end
