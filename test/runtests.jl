using Test, Documenter, CoDeBetHe
@testset "CoDeBetHe" begin
    ... # other tests & testsets
    doctest(CoDeBetHe; manual = false)
    ... # other tests & testsets
end