"""
    rmsnorm(x, w)

Calculate the rmsnorm of `x` and `w`, the scaled product 'λw * x'.

# Examples
```jldoctest
julia>  x = [1.0f0,2,3];

julia>  w = [1.0f0,1,1];

julia> o = Llama2.rmsnorm(x, w) 
3-element Vector{Float32}:
 0.46290955
 0.9258191
 1.3887286
```
"""
function rmsnorm(x::AbstractVector{Float32}, w::AbstractVector{Float32})

    (length(w) != length(x)) && throw(DimensionMismatch("x and w must have the same dimensions"))
    isempty(x) && throw(ArgumentError("x must not be empty"))

    #calculate sum of squares
    ss = dot(x, x)

    ss = ss / length(x) + 1f-5
    scale = inv(sqrt(ss))

    return scale * w .* x
end


"""
softmax!(x)
Updates the Output of an Layer 'x' with the softmax of the input.

# Examples
```jldoctest
julia> x = [-1.0f0,0,1];

julia> Llama2.softmax!(x);

julia> x
3-element Vector{Float32}:
 0.09003057
 0.24472848
 0.66524094
```
"""
function softmax!(x::AbstractVector{Float32})

    isempty(x) && throw(ArgumentError("x must not be empty"))

    max_x = maximum(x)

    for i in eachindex(x)
        x[i] = exp(x[i] - max_x)
    end

    x ./= sum(x)

    return x
end

# function forward!(transformer::Transformer, token::Int32, pos::Int32)

#     config = transformer.config
#     weights = transformer.weights
#     state = transformer.state

#     dim = config.dim
#     kv_dim = div(dim * config.n_kv_heads, config.n_heads)
#     kv_mul = div(config.n_heads, config.n_kv_heads)
#     hidden_dim = config.hidden_dim
#     head_size = div(dim, config.n_heads)

#     # assigning input token embedding to x
#     x = @view weights.token_embedding_table[token, :]

#     for l in 1:config.n_layers

#         xb = rmsnorm(x, weights.rms_att_weight[l, :])

#         k = @view state.key_cache[l, pos, :]
#         v = @view state.value_cache[l, pos, :]
#         # matmul to get q, k, v

#         q = weights.wq[l, :, :] * xb
#         k .= weights.wk[l, :, :] * xb
#         v .= weights.wv[l, :, :] * xb

#         for i in 1:2:dim

#             head_dim = i % head_size
#             freq = 1.0f0 / (10000.0f0^(head_dim / head_size))
#             val = (pos - 1) * freq
#             fcr = cos(val)
#             fci = sin(val)
            
#             for v in 1:(1 + (i < kv_dim))
#                 vec = v == 0 ? q : k
#                 v0 = vec[i]
#                 v1 = vec[i + 1]
#                 vec[i] = v0 * fcr - v1 * fci
#                 vec[i + 1] = v0 * fci + v1 * fcr
#             end

#         end

#         for h in 1:config.n_heads # multi-head attention

#             q_head = @view q[((h - 1)* head_size + 1):(h  * head_size)]
#             att = Vector{Float32}(undef, pos)

#             for t in 1:pos

#                 k = state.key_cache[l, t, ((div(h, kv_mul) - 1) * head_size + 1):(div(h, kv_mul) * head_size)]

#                 score = dot(q_head, k)/sqrt(head_size)
#                 att[t] = score

#             end

#             softmax!(att)

#             xb_head = @view xb[((h - 1) * head_size + 1):(h * head_size)]

#             for t in 1:pos

#                 v = state.value_cache[l, t, ((div(h, kv_mul) - 1) * head_size + 1):(div(h, kv_mul) * head_size)]
#                 a = att[t]
                
#                 xb_head += a * v

#             end

#         end

#         x .+= weights.wo[l, :, :] * xb

#         xb = rmsnorm(x, weights.rms_ffn_weight[l, :])

#         hb = weights.w1[l, :, :] * xb
#         hb2 = weights.w3[l, :, :] * xb

#         for i in 1:hidden_dim
#             val = hb[i]
#             val *= (1.0f0 / (1.0f0 + exp(-val)))
#             val *= hb2[i]
#             hb[i] = val
#         end

#         xb = weights.w2[l, :, :] * hb

#         x += xb
#     end

#     x .= rmsnorm(x, weights.rms_final_weight) # final rmsnorm
    
#     # classifier into logits
#     state.logits .= weights.wcls * x

#     return state.logits

# end

function forward!(token::Int32, pos::Int32, conf::Config, state::RunState, weights::TransformerWeights)
    x = state.x
    dim = conf.dim
    hidden_dim = conf.hidden_dim
    head_size = div(dim, conf.n_heads)

    x = weights.token_embedding_table[token, :]

    freq_cis_real_row = weights.freq_cis_real[div(pos * head_size, 2)+1:div((pos+1)*head_size, 2)]
    freq_cis_imag_row = weights.freq_cis_imag[div(pos * head_size, 2)+1:div((pos+1)*head_size, 2)]
    
    # iterate over layers
    for l in 1:conf.n_layers
        # Attention rmsnorm
        state.xb = rmsnorm(x, weights.rms_att_weight[(l-1) * dim+1:l * dim])

        state.q .= weights.wq[l, :, :] * state.xb
        state.k .= weights.wk[l, :, :] * state.xb
        state.v .= weights.wv[l, :, :] * state.xb

        for h in 1:conf.n_heads
            q = state.q[(h-1) * head_size+1:h * head_size]
            k = state.k[(h-1) * head_size+1:h * head_size]

            # NOTE: +1
            for i in 0:2:head_size-1
                q0, q1 = q[i+1], q[i+2]
                k0, k1 = k[i+1], k[i+2]
                fcr = freq_cis_real_row[div(i, 2)+1]
                fci = freq_cis_imag_row[div(i, 2)+1]
                q[i+1] = q0*fcr - q1*fci
                q[i+2] = q0*fci + q1*fcr
                k[i+1] = k0*fcr - k1*fci
                k[i+2] = k0*fci + k1*fcr
            end

            state.q[(h-1) * head_size+1:h * head_size] = q
            state.k[(h-1) * head_size+1:h * head_size] = k
        end

        state.key_cache[l, pos, :] = state.k
        state.value_cache[l, pos, :] = state.v

        for h in 1:conf.n_heads
            q = state.q[(h-1) * head_size+1:h * head_size]

            att = state.att[h,:]

            for t in 1:pos
                k = state.key_cache[l, t, (h-1)*head_size + 1:h*head_size]

                score = dot(q, k)
                score /= sqrt(head_size)

                att[t] = score
            end

            att = softmax!(att)

            state.xb[(h-1) * head_size+1:h * head_size] = zeros(Float32, head_size)

            for t in 1:pos
                v = state.value_cache[l, t, (h-1) * head_size+1:h * head_size]

                a = att[t]

                state.xb[(h-1) * head_size+1:(h-1) * head_size+1 + head_size-1] .+= a .* v
            end
        end
        state.xb2 .= weights.wo[l, :, :] * state.xb

        x = x + state.xb2

        state.xb = rmsnorm(x, weights.rms_ffn_weight[l, :])

        state.hb .= weights.w1[l, :, :] * state.xb

        state.hb2 .= weights.w3[l, :, :] * state.xb

        state.hb .= state.hb .* (1.0f0 ./ (1.0f0 .+ exp.(-state.hb)))

        state.hb .= state.hb .* state.hb2

        state.xb .= weights.w2[l, :, :] * state.hb
        
        x = x + state.xb
    end

    x = rmsnorm(x, weights.rms_final_weight)

    state.logits .= weights.wcls[:, :] * x
    return state.logits
end