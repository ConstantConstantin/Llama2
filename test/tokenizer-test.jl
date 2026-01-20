using Llama2
using Test

@testset "tokenizer" begin
    @testset "TokenIndex" begin
            # TokenIndex stores a token string + its integer id.
            ti = Llama2.TokenIndex("Julia", 1)
            @test ti.str == "Julia"
            @test ti.id == Int16(1)

            # ids are normalized to Int16 internally.
            ti2 = Llama2.TokenIndex("X", Int64(2))
            @test ti2.id == Int16(2)

            # Negative ids are invalid.
            @test_throws DomainError Llama2.TokenIndex("Bad", -1)
            # Zero is allowed 
            ti0 = Llama2.TokenIndex("Zero", 0)
            @test ti0.id == Int16(0)
        end

    @testset "compare_tokens" begin
            # Comparison helper for sorting by string (and/or tie-breaking rules).
            @test Llama2.compare_tokens(Llama2.TokenIndex("A", 1), Llama2.TokenIndex("B", 2)) == true
            @test Llama2.compare_tokens(Llama2.TokenIndex("B", 1), Llama2.TokenIndex("A", 2)) == false

            # Same string should not be considered "less".
            @test Llama2.compare_tokens(Llama2.TokenIndex("aa", 1), Llama2.TokenIndex("aa", 2)) == false
        end

    @testset "Tokenizer" begin
            # Build a tiny tokenizer (vocab/scores + byte pieces table).
            vocab = ["a", "b", "ab"]
            scores = Float32[0.0, 0.0, 10.0]

            # 'sorted' can start empty; encode() typically fills/sorts it.
            sorted = Llama2.TokenIndex[]
            byte_pieces = collect(UInt8.(0:255))

            tok = Tokenizer(vocab, scores, sorted, 3, 10, byte_pieces)

            # Basic field sanity.
            @test tok.vocab_size == Int16(3)
            @test tok.max_token_length == UInt16(10)
            @test length(tok.byte_pieces) == 256

            # Wrong byte_pieces length should error.
            @test_throws ArgumentError Tokenizer(vocab, scores, sorted, 3, 10, UInt8[1,2,3])

            # Invalid max token length should error.
            @test_throws DomainError Tokenizer(vocab, scores, sorted, 3, -1, byte_pieces)
        end

    @testset "str_lookup" begin
            # Binary search / lookup in a sorted vocab list.
            sorted_vocab = [Llama2.TokenIndex("aa", 1), Llama2.TokenIndex("bb", 2), Llama2.TokenIndex("cc", 3)]
            @test Llama2.str_lookup("aa", sorted_vocab) == Int16(1)
            @test Llama2.str_lookup("bb", sorted_vocab) == Int16(2)

            # Not found returns -1 (sentinel).
            @test Llama2.str_lookup("ba", sorted_vocab) == Int16(-1)
        end

    @testset "encode" begin
            # Encode a string into token ids using the tiny vocab.
            vocab = ["a", "b", "ab"]
            scores = Float32[0.0, 0.0, 10.0]
            byte_pieces = collect(UInt8.(0:255))

            # Preallocate sorted vocab storage.
            sorted = Vector{Llama2.TokenIndex}(undef, 3)
            tok = Tokenizer(vocab, scores, sorted, 3, 10, byte_pieces)

            ids = Llama2.encode(tok, "ab")
            @test ids == [Int16(3)]

            # If the tokenizer cannot represent a string, encode should throw.
            sorted2 = Vector{Llama2.TokenIndex}(undef, 1)
            tok2 = Tokenizer(["a"], Float32[0.0], sorted2, 1, 10, byte_pieces)
            @test_throws ArgumentError Llama2.encode(tok2, "b")
    end
end 