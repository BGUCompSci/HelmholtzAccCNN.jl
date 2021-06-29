using Statistics
using LinearAlgebra
using Flux
using Flux: @functor
using LaTeXStrings
using KrylovMethods
using Distributions: Normal
using BSON: @load
using Plots
using Dates
using CSV, DataFrames
using Random

use_gpu = true
if use_gpu == true
    using CUDA
    #CUDA.allowscalar(false)
    cgpu = gpu
else
    cgpu = cpu
    pyplot()
end

r_type = Float32
c_type = ComplexF32
u_type = Float32

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
point_sorce_results = true
check_unet_as_preconditioner = true
m = 1
blocks = 10

test_name = "13_48_25 SDNUnet1 NaN -1 g=-1 t=Float32 g=t e=f k=0 25 n=128 f=10_0 m=25000 bs=20 lr=0_0001 each=60 i=125"
sm_test_name = "13_48_25 SDNUnet1 4 25"

# Model Parameters

model_type = SDNUnet1
k_type = NaN
resnet_type = SResidualBlock
arch = 0
k_chs = 10
indexes = 3
σ = elu
kernel = (3,3)
e_vcycle_input = false
kappa_type = 4 # 0 - uniform, 1 - CIFAR10, 2 - STL10
kappa_threshold = 25 # kappa ∈ [0.01*threshold, 1]
kappa_input = true
gamma_input = true
axb = false
norm_input = false
smooth = true
k_kernel = 3
before_jacobi = false
unet_in_vcycle = false

model = load_model!(test_name, e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)

n = 128
f = 10.0
restrt = 10
max_iter = 10
kappa = generate_kappa!(n; type=kappa_type, smooth=smooth, threshold=kappa_threshold, kernel=k_kernel) #ones(r_type,n-1,n-1)
omega = 2*pi*f;
gamma = gamma_val*2*pi * ones(r_type,size(kappa));
gamma = absorbing_layer!(gamma, pad_cells, omega);

# Re-training:
# bs = 10
# iter = 30
#model, sm_test_name = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f, 100, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1;residual_loss=false)

r_type = Float64
c_type = ComplexF64
u_type = Float64
if point_sorce_results == true
    check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end
if check_unet_as_preconditioner == true #gmres_alternatively_
    check_model!(sm_test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
end

n = 256
f = 20.0
restrt = 15
max_iter = 20
k_kernel = 3
kappa = generate_kappa!(n; type=kappa_type, smooth=true, threshold=kappa_threshold, kernel=k_kernel)
omega = 2*pi*f;
gamma = gamma_val*2*pi * ones(r_type,size(kappa));
gamma = absorbing_layer!(gamma, pad_cells, omega);

# Re-training:
# bs = 10
# iter = 30
#model, sm_test_name = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f, 200, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1;residual_loss=false)

r_type = Float64
c_type = ComplexF64
u_type = Float64
if point_sorce_results == true
    check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end
if check_unet_as_preconditioner == true
    check_model!(sm_test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
end

n = 512
f = 40.0
restrt = 20
max_iter = 20
k_kernel = 3
kappa = generate_kappa!(n; type=kappa_type, smooth=false, threshold=kappa_threshold, kernel=k_kernel)
omega = 2*pi*f;
gamma = gamma_val*2*pi * ones(r_type,size(kappa));
gamma = absorbing_layer!(gamma, pad_cells, omega);

# Re-training:
# bs = 20
# iter = 30
#model, sm_test_name = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f, 300, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1;residual_loss=false)

r_type = Float64
c_type = ComplexF64
u_type = Float64
if point_sorce_results == true
    check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end
if check_unet_as_preconditioner == true
    check_model!(sm_test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
end
