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
heatmap(real(x[:,:,1,1]), color=:grays)
_, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
b = (h^2) .* helmholtz_chain!(x, helmholtz_matrix; h=h)
heatmap(real(b[:,:,1,1]), color=:grays)
