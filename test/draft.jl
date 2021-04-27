using Statistics
using LinearAlgebra
using Flux
using Flux: @functor
using LaTeXStrings
using KrylovMethods
using Distributions: Normal
using Plots
using CSV, DataFrames
using Dates
using Random
using BSON: @load
pyplot()
cgpu = cpu
r_type = Float64
c_type = ComplexF64
u_type = Float16

include("../src/multigrid/helmholtz_methods.jl")

# Grid

n = 128
n_cells = [n,n];
h = 1.0./n;
src = [64,64]

b = zeros(c_type,n-1,n-1)
b[src[1],src[2]] = 1.0 ./mean(h.^2);

# Parameters

f = 10.0
kappa = ones(Float64,tuple(n_cells .-1...))
omega = 2*pi*f;

# Gamma

gamma_val = 0.00001
gamma = gamma_val*2*pi * ones(ComplexF64,size(kappa));
pad_cells = [20;20]
gamma = absorbing_layer!(gamma, pad_cells, omega);

# V cycle

x = randn(c_type,n-1,n-1,1,1)
xh = x .* h^2

_, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)

b = helmholtz_chain!(x, helmholtz_matrix; h=h)
bh = helmholtz_chain!(xh, helmholtz_matrix; h=h)
bh2 = b .* h^2

heatmap(real(x[:,:,1,1]), color=:grays)
heatmap(real(xh[:,:,1,1]), color=:grays)

heatmap(real(b[:,:,1,1]), color=:grays)
heatmap(real(bh[:,:,1,1]), color=:grays)
heatmap(real(bh2[:,:,1,1]), color=:grays)

x_vc = fgmres_v_cycle_helmholtz!(n, h, b, kappa, omega, gamma; restrt=30, maxIter=10)
println("$(norm(x-x_vc)/norm(x))")
println("$(norm(xh - x_vc .* h^2) / norm(xh))")
xh_vc = fgmres_v_cycle_helmholtz!(n, h, bh, kappa, omega, gamma; restrt=30, maxIter=10)
println("$(norm(xh-xh_vc)/norm(xh))")
println("$(norm(x - xh_vc ./ h^2) / norm(x))")
println("$(norm(x .* h^2 - xh_vc) / norm(xh_vc))")


heatmap(real(x[:,:,1,1]), color=:grays)
heatmap(real((xh_vc ./ h^2)[:,:,1,1]), color=:grays)
