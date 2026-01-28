@testset "talk" begin

    @testset "talktollm" begin

        p = normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin"))
        
        for result in (talktollm(p), talktollm(p, "Some ducks on the pond "), talktollm(p; max_tokens = 127), talktollm(p; temperature=1.f0), talktollm(p; temperature=1.2f0, topp=0.5f0))

            @test result isa String

        end

    end

end
