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

pu = cpu # gpu
r_type = Float64
c_type = ComplexF64
u_type = Float32
gmres_type = ComplexF64
a_type = CuArray{gmres_type}
a_type = Array{gmres_type}
run_title = "64bit_cpu"

include("../../src/multigrid/helmholtz_methods.jl")
include("../../src/unet/model.jl")
include("../../src/unet/data.jl")
include("../../src/unet/train.jl")
include("../../src/kappa_models.jl")
include("../../src/gpu_krylov.jl")
include("unet_test_utils.jl")

fgmres_func = KrylovMethods.fgmres # gpu_flexible_gmres #

# Test Parameters

level = 3
v2_iter = 10
gamma_val = 0.00001
pad_cells = [10;10]
point_sorce_results = true
check_unet_as_preconditioner = true
dataset_size = 1
blocks = 10

# Model Parameters

model_type = FFSDNUnet
k_type = FFKappa
resnet_type = SResidualBlock
arch = 2
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
k_kernel = 5
before_jacobi = false
unet_in_vcycle = false

n = m = 128
h = r_type(2.0 / (n+m))
f = 10.0
restrt = 10
max_iter = 30
kappa = r_type.(generate_kappa!(n, m; type=kappa_type, smooth=smooth, threshold=kappa_threshold, kernel=k_kernel)|>pu) #ones(r_type,n-1,n-1)
omega = r_type(2*pi*f);
gamma = gamma_val*2*pi * ones(r_type,size(kappa))
gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))|>pu
bs = 10

retrain_size = 300
iter = 30

test_name = "09_40_06 SDNUnet1 g=-1 t=Float32 g=t e=f k=0 50 n=128 f=10_0 m=20000 bs=20 lr=0_0001 each=70 i=155"
test_name = "23_48_23 RADAM FFSDNUnet FFKappa SResidualBlock 10 elu 3 5 g=-1 t=Float32 g=t e=f r=f k=1 25 n=128 f=10_0 m=20000 bs=20 lr=0_0001 each=48 i=100"

sm_test_name = "23_48_23_$(run_title)_b$(blocks)_m$(kappa_type)_f$(Int32(f))_$(retrain_size)_$(iter)"
sm_test_name_r = "$(sm_test_name)_retrain"

model = load_model!("../../models/$(test_name).bson", e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)

model_r128 = load_model!("../../models/23_48_23 128 100 10 30 5 f f -1 r=f.bson", e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)
model_r256 = load_model!("../../models/23_48_23 10 blocks 256 300 10 20 3 f f -1 r=f.bson", e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)
model_r512 = load_model!("../../models/23_48_23 10 blocks 512 500 20 30 3 f f -1 r=f.bson", e_vcycle_input, kappa_input, gamma_input;kernel = kernel,model_type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)

# model_r128, _ = model_tuning!(model1, sm_test_name, kappa, omega, gamma, n, m, f, retrain_size, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1,kappa_type;residual_loss=false)

if point_sorce_results == false
    check_point_source_problem!("$(Dates.format(now(), "HH_MM_SS")) $(sm_test_name)", model, n, m, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end
if check_unet_as_preconditioner == true #gmres_alternatively_
    # check_model_times!(sm_test_name, model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
    check_model!(sm_test_name, model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
    check_model!(sm_test_name_r, model_r128, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
end

n = m = 256
f = 20.0
restrt = 15
max_iter = 40
k_kernel = 3
kappa = r_type.((generate_kappa!(n, m; type=kappa_type, smooth=true, threshold=kappa_threshold, kernel=k_kernel))|>pu) #ones(r_type,n-1,n-1)
omega = r_type(2*pi*f)
gamma = gamma_val*2*pi * ones(r_type,size(kappa))
gamma = r_type.((absorbing_layer!(gamma, pad_cells, omega))|>pu)
bs = 10
iter = 40
# model, _ = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f, 50, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1)
# model, sm_test_name = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f,500, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1;residual_loss=false)

if point_sorce_results == false
    check_point_source_problem!("$(Dates.format(now(), "HH_MM_SS")) $(sm_test_name)", model, n, m, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end
if check_unet_as_preconditioner == true
    # check_model_times!("$(sm_test_name)", model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
    check_model!(sm_test_name, model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
    check_model!(sm_test_name_r, model_r256, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
end

n = m = 512
f = 40.0
restrt = 20
max_iter = 50
k_kernel = 3
kappa = r_type.(generate_kappa!(n, m; type=kappa_type, smooth=false, threshold=kappa_threshold, kernel=k_kernel))|>pu
omega = r_type(2*pi*f)
gamma = gamma_val*2*pi * ones(r_type,size(kappa))
gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))|>pu
bs = 20
iter = 40
# model, _ = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f, 50, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1)
# model, sm_test_name = model_tuning!(model, sm_test_name, kappa, omega, gamma, n, f, 500, bs, iter, 0.001, kappa_threshold, false, false, k_kernel, -1;residual_loss=false)

if point_sorce_results == false
    check_point_source_problem!(test_name, model, n, m, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=v2_iter, level=level)
end
if check_unet_as_preconditioner == true
    # check_model_times!("$(sm_test_name)", model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
    check_model!(sm_test_name, model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
    check_model!(sm_test_name_r, model_r512, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=smooth, k_kernel=k_kernel, threshold=kappa_threshold, axb=axb, norm_input=norm_input, before_jacobi=false,log_error=false,unet_in_vcycle=unet_in_vcycle,indexes=indexes, arch=arch)
end
