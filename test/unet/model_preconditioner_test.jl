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
using CSV, DataFrames

use_gpu = false
if use_gpu == true
    using CUDA
    #CUDA.allowscalar(false)
    cgpu = gpu
else
    cgpu = cpu
    pyplot()
end

r_type = Float64
c_type = ComplexF64
u_type = Float16

include("../../src/multigrid/helmholtz_methods.jl")
include("../../src/unet/model.jl")
include("../../src/unet/data.jl")
include("../../src/unet/train.jl")
include("../../src/kappa_models.jl")
include("unet_test_utils.jl")

# Test Parameters

level = 3
v2_iter = 10
gamma_val = 0.00001
pad_cells = [10;10]
point_sorce_results = false
check_unet_as_preconditioner = true
m = 20

test_name = "22_56_28 DNUnet axb=f norm=t t=Float64 k=3 25 g=t e=f da=f k=2 n=128 f=10_0 m=20000 bs=10 opt=ADAM lr=0_005 each=50 i=200"

# Model Parameters

model_type = DNUnet
kernel = (3,3)
e_vcycle_input = false
kappa_type = 2 # 0 - uniform, 1 - CIFAR10, 2 - STL10
kappa_threshold = 25 # kappa ∈ [0.01*threshold, 1]
kappa_input = true
gamma_input = true
axb = false
norm_input = true
smooth = false

model = load_model!(test_name, e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type)

n = 64
f = 5.0
restrt = 10
max_iter = 3
kappa = ones(r_type,n-1,n-1)
omega = 2*pi*f;
gamma = gamma_val*2*pi * ones(c_type,size(kappa));
gamma = absorbing_layer!(gamma, pad_cells, omega);
if check_unet_as_preconditioner == true
    check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, threshold=kappa_threshold, axb=axb, norm_input=norm_input)
end
if point_sorce_results == true
    check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end

n = 128
f = 10.0
restrt = 20
max_iter = 3
kappa = ones(r_type,n-1,n-1)
omega = 2*pi*f;
gamma = gamma_val*2*pi * ones(c_type,size(kappa));
gamma = absorbing_layer!(gamma, pad_cells, omega);
if check_unet_as_preconditioner == true
    check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, threshold=kappa_threshold, axb=axb, norm_input=norm_input)
end
if point_sorce_results == true
    check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end

n = 256
f = 20.0
restrt = 25
max_iter = 4
kappa = ones(r_type,n-1,n-1)
omega = 2*pi*f;
gamma = gamma_val*2*pi * ones(c_type,size(kappa));
gamma = absorbing_layer!(gamma, pad_cells, omega);
if point_sorce_results == true
    check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end
if check_unet_as_preconditioner == true
    check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, threshold=kappa_threshold, axb=axb, norm_input=norm_input)
end
