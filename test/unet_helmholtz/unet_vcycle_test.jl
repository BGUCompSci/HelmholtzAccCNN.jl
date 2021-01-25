using Statistics
using LinearAlgebra
using Flux
using Flux: @functor
using LaTeXStrings
using KrylovMethods
using BSON: @load
using Plots
using Dates
using Distributions: Normal

use_gpu = false
if use_gpu == true
    using CUDA
    #CUDA.allowscalar(false)
    cgpu = gpu
else
    cgpu = cpu
    pyplot()
end

include("../../src/multigrid/helmholtz_methods.jl")
include("../../src/unet/model.jl")
include("../../src/unet/data.jl")
include("../../src/unet/train.jl")
include("../../src/kappa_models.jl")
include("utils.jl")

# Parameters

n = 64
f = 5
kappa = ones(Float64,tuple(n_cells .-1...))
omega = 2*pi*f;

# Gamma

gamma_val = 0.00001
gamma = gamma_val*2*pi * ones(ComplexF64,size(kappa));
pad_cells = [10;10]
gamma = absorbing_layer!(gamma, pad_cells, omega);

# GMRES Parameters

level = 3
v2_iter = 10
restrt = 20
max_iter = 4
smooth=true
m = 5

test_name = "small resnet=false kernel=3 random_batch=false gamma_input=false da=false models_count=50000 cifar=true e_vcycle=false n=64 m=100000 bs=20 opt=ADAM lr=0_0005 iter=30"
e_vcycle_input = false
kappa_input = true
gamma_input = false
kernel = (3,3)
check_model!(test_name, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth)

test_name = "small resnet=false kernel=3 random_batch=false gamma_input=true da=false models_count=50000 cifar=true e_vcycle=false n=64 m=100000 bs=20 opt=ADAM lr=0_0005 iter=30"
e_vcycle_input = false
gamma_input = true
check_model!(test_name, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth)

test_name = "small resnet=false kernel=3 random_batch=false gamma_input=false da=false models_count=50000 cifar=true e_vcycle=true n=64 m=100000 bs=20 opt=ADAM lr=0_0005 iter=30"
e_vcycle_input = true
gamma_input = false
check_model!(test_name, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth)

test_name = "small resnet=false kernel=3 random_batch=false gamma_input=true da=false models_count=50000 cifar=true e_vcycle=true n=64 m=100000 bs=20 opt=ADAM lr=0_0005 iter=30"
e_vcycle_input = true
gamma_input = true
check_model!(test_name, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth)
