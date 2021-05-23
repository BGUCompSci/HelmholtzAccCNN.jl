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
u_type = Float64

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


test_name = "16_48_39 SDNUnet1 DTUnet 10 elu 3 10 g=-1 t=Float32 g=t e=f k=1 25 n=128 f=10_0 m=20000 bs=20 lr=0_0001 each=64 i=128"

# Model Parameters

model_type = SDNUnet1
k_type = DTUnet
k_chs = 10
indexes = 3
σ = elu
kernel = (3,3)
e_vcycle_input = false
kappa_type = 1 # 0 - uniform, 1 - CIFAR10, 2 - STL10
kappa_threshold = 25 # kappa ∈ [0.01*threshold, 1]
kappa_input = true
gamma_input = true
axb = false
norm_input = false
smooth = true
k_kernel = 10
before_jacobi = false
unet_in_vcycle = false

model = load_model!(test_name, e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, k_chs=k_chs, indexes=indexes, σ=σ)

# n = 64
# f = 5.0
# restrt = 5
# max_iter = 10
# kappa = ones(r_type,n-1,n-1)
# omega = 2*pi*f;
# gamma = gamma_val*2*pi * ones(r_type,size(kappa));
# gamma = absorbing_layer!(gamma, pad_cells, omega);
# if check_unet_as_preconditioner == true
#     check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=before_jacobi)
# end
# if point_sorce_results == true
#     check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
# end

n = 128
f = 10.0
restrt = 10 # 1
max_iter = 4 # 20
kappa = ones(r_type,n-1,n-1)
omega = 2*pi*f;
gamma = gamma_val*2*pi * ones(r_type,size(kappa));
gamma = absorbing_layer!(gamma, pad_cells, omega);
if check_unet_as_preconditioner == true #gmres_alternatively_
    check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes)
    #    check_gmres_alternatively_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=true)
#    check_full_solution_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, threshold=kappa_threshold, axb=axb, norm_input=norm_input)
end
if point_sorce_results == true
    check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end

# n = 256
# f = 20.0
# restrt = 10
# max_iter = 10
# kappa = ones(r_type,n-1,n-1)
# omega = 2*pi*f;
# gamma = gamma_val*2*pi * ones(r_type,size(kappa));
# gamma = absorbing_layer!(gamma, pad_cells, omega);
# if point_sorce_results == true
#     check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
# end
# if check_unet_as_preconditioner == true
#     check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=before_jacobi)
# end
