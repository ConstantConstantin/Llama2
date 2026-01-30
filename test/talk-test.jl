@testset "talk" begin

    p = normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin"))

    @testset "talktollm" begin
        
        for result in (talktollm(p), talktollm(p, "Some ducks on the pond "), talktollm(p; max_tokens = 127))

            @test result isa String

        end

    end

    @testset "chatwithllm" begin

        c = ChatBot(p)

        d = chatwithllm(c)
        e = chatwithllm(c, " and then he")

        @test d isa String

    end

end
