using Random

function sample_mult(probabilities::Vector{Float32}, coin::Float64)
    cdf = 0.0f0
    for i in eachindex(probabilities)
        cdf += probabilities[i]
        coin < cdf && return i
    end
    return lastindex(probabilities)
end

function isless_probindex(first_probindex::ProbIndex, second_probindex::ProbIndex)
    return first_probindex.prob < second_probindex.prob
end

function sample_topp(probabilities::Vector{Float32},
    topp::Float32,
    probindex::Vector{ProbIndex},
    coin::Float64
    )

    cutoff = (1.0f0 - topp) / length(probabilities)
    probindex.index = findall(x -> x >= cutoff, probabilities)
    probindex.prob = probabilities[probindex.index]

    sort!(probindex, lt=isless_probindex)

    cumulative_prob = 0.0f0
    last_idx = lastindex(probindex)
    @inbounds for i in eachindex(probindex)
        cumulative_prob += probindex[i].prob
        if cumulative_prob > topp
            last_idx = i
            break
        end
    end

    r = coin * cumulative_prob
    cdf = 0.0f0
    @inbounds for i in eachindex(probindex)
        cdf += probindex[i].prob
        if r < cdf
            return probindex[i].index
        end
    end
    return probindex[last_idx].index
end

struct ProbIndex
    prob::Float32
    index::Int32

    function ProbIndex(prob::Float32, index::Int32)
        index < 0 && throw(DomainError("Prob index must be > 0."))
        new(convert(Float32, prob), convert(Int32, index))
    end
end

struct Sampler
    vocab_size::Int32
    probindex::Vector{ProbIndex} # karpathy: buffer used in top-p sampling
    temperature::Float32
    topp::Float32
    rng_state::Int128

    function Sampler(vocab_size::Int32,
        probindex::Vector{ProbIndex},
        temperature::Float32,
        topp::Float32,
        rng_state::Int128
    )
        vocab_size <= 0 && throw(DomainError("vocab_size must be > 0."))
        new(convert(Int32, vocab_size),
            convert(Vector{ProbIndex}, probindex),
            convert(Float32, temperature),
            convert(Float32, topp),
            convert(Int128, rng_state)
        )
    end
end
function Sampler(vocab_size::Int32, temperature::Float32, topp::Float32, rng_seed::Int128) 
    # karpathy: buffer only used with nucleus sampling; may not need but it's ~small:
    probindex = Vector{ProbIndex}(undef, vocab_size)
    Sampler(vocab_size, probindex, temperature, topp, rng_seed)
end

function sample(sampler::Sampler)(logits::Vector{Float32})
    if sampler.temperature == 0.0
        next = argmax(logits)
    else
        logits = logits / sampler.temperature
        softmax!(logits)

        rng = MersenneTwister(sampler.rng_state)
        coin = rand(rng)

        if sampler.topp <= 0 || sampler.topp >= 1
            next = sample_mult(logits, coin)
        else
            next = sample_topp(logits, sampler.topp, sampler.probindex, coin)
        end
    end

    return next
end
