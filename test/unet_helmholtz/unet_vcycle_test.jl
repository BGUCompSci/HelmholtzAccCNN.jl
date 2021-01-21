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

# Grid

n = 64
n_cells = [n,n];
h = 1.0./n;

# Parameters

f = 5
kappa = cifar_model!(n;smooth=true) # ones(Float64,tuple(n_cells .-1...)) # cifar_model!(n) #
omega = 2*pi*f;

# Gamma

gamma_val = 0.00001
gamma = gamma_val*2*pi * ones(ComplexF64,size(kappa));
pad_cells = [10;10]
gamma = absorbing_layer!(gamma, pad_cells, omega);

_, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)

# Test Network Parameters"/home/yaelaz/HelmholtzAccCNN.jl/unet conv data augmentetion m=10000 bs=100 opt=ADAM lr=0_001 iter=20.bson"
# The Best! test_name = "unet resnet data augmentetion n=64 m=50000 bs=2000 r=0_5 opt=ADAM lr=0_001 iter=10"
test_name = "resnet=false kernel=5 random_batch=false gamma_input=false models_count=50000 cifar=true e_vcycle=true n=64 m=100 bs=5 opt=ADAM lr=0_001 iter=3"
e_vcycle_input = true
after_vcycle = false
kappa_input = true
gamma_input = false

model = create_model!(e_vcycle_input,kappa_input,gamma_input)

model = model|>cpu
@load "$(test_name).bson" model
@info "$(Dates.format(now(), "HH:MM:SS")) - Load Model"

model = model|>cgpu

# GMRES Parameters

level = 3
v2_iter = 10
restrt = 15
max_iter = after_vcycle == true ? 5 : 8

x_true = randn(ComplexF64,n-1,n-1, 1, 1)
r_vcycle, e_true = generate_r_vcycle!(n, kappa, omega, gamma, x_true)

A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))

function M_Unet(r)
    r = reshape(r, n-1, n-1)
    if e_vcycle_input == true
        e_vcycle = zeros(ComplexF64,n-1,n-1)
        e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)

        input = cat(complex_grid_to_channels!(reshape(e_vcycle, n-1, n-1, 1, 1), n),
                            complex_grid_to_channels!(reshape(r, n-1, n-1, 1, 1), n), dims=3)
    else
        input = complex_grid_to_channels!(reshape(r, n-1, n-1, 1, 1), n)
    end

    input = kappa_input == true ? cat(input, reshape(kappa, n-1, n-1, 1, 1), dims=3) : input
    input = gamma_input == true ? cat(input, complex_grid_to_channels!(gamma, n), dims=3) : input

    e_unet = model(input|>cgpu)|>cpu

    e_vcycle = e_unet[:,:,1,1] +im*e_unet[:,:,2,1]
    if after_vcycle == true
        e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
    end
    return vec(e_vcycle)
end

function M(r)
    r = reshape(r, n-1, n-1)
    e_vcycle = zeros(ComplexF64,n-1,n-1)
    e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
    if after_vcycle == true
        e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
    end
    return vec(e_vcycle)
end

x = zeros(ComplexF64,n-1,n-1)
x2,flag2,err2,iter2,resvec2 = KrylovMethods.fgmres(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                M=M, x=vec(x), out=2, flexible=true)

@info "$(Dates.format(now(), "HH:MM:SS")) - Error without UNet: $(norm_diff!(reshape(x,n-1,n-1),e_true[:,:,1,1]))"

x = zeros(ComplexF64,n-1,n-1)
x1,flag1,err1,iter1,resvec1 = KrylovMethods.fgmres(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                M=M_Unet, x=vec(x), out=2, flexible=true)
@info "$(Dates.format(now(), "HH:MM:SS")) - Error with UNet: $(norm_diff!(reshape(x,n-1,n-1),e_true[:,:,1,1]))"
# heatmap(real(x-e_true), color=:grays)
# savefig("$(test_name) error with UNet")

iter = range(1, length=length(resvec1))
p = plot(iter,resvec1,label="With UNet")
plot!(iter,resvec2,label="Without UNet")
yaxis!(L"\Vert b - Hx \Vert_2", :log10)
xlabel!("Iterations")
sub_title = "residual graph"
if after_vcycle == true
    sub_title = "with after vcycle $(sub_title)"
    if e_vcycle_input == true
        title!("M=Vcycle(r,Unet(Vcycle(r,0),r)) vs M=Vcycle(r,Vcycle(r,0)).")
    else
        title!("M=Vcycle(r,Unet(r)) vs M=Vcycle(r,Vcycle(r,0)).")
    end
else
    if e_vcycle_input == true
        title!("M=Unet(Vcycle(r,0),r) vs M=Vcycle(r,0).")
    else
        title!("M=Unet(r) vs M=Vcycle(r,0).")
    end
end

savefig("$(test_name) $(max_iter) $(restrt) $(level) $(v2_iter) $(n) $(f) $(sub_title)")

# heatmap(real(reshape(x1,n-1,n-1)), color=:grays)
# savefig("$(test_name) $(max_iter) $(restrt) $(level) $(v2_iter) $(n) $(f) unet x")
#
# heatmap(real(reshape(x2,n-1,n-1)), color=:grays)
# savefig("$(test_name) $(max_iter) $(restrt) $(level) $(v2_iter) $(n) $(f) vcycle x")
#heatmap(kappa, color=:grays)
#savefig("$(test_name) $(max_iter) $(restrt) $(level) $(v2_iter) $(n) $(f) $(sub_title) kappa")
