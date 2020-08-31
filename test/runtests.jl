using Test
@testset "Multigrid" begin
include("multigrid_helmholtz/shifted_laplacian_helmholtz_test.jl");
include("multigrid_laplacian/laplacian_test.jl");
end
