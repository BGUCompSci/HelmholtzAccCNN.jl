using Test
using Flux
using Flux: @functor
using Zygote

include("../src/unet/model.jl")

@testset "Gradient Tests" begin
  model = create_model!(e_vcycle_input=false,kappa_input=false,gamma_input=false;kernel=(3,3),type=SUnet)
  ip = rand(Float32, 64, 64, 2, 1)
  gs = gradient(Flux.params(model)) do
    sum(model(ip))
  end

  @test gs isa Zygote.Grads
end
