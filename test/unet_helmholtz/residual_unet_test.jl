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

# Grid

n = 64
n_cells = [n,n];
h = 1.0./n;

# Parameters

f = 5
kappa = ones(Float64,tuple(n_cells .-1...))
omega = 2*pi*f;

# Gamma

gamma_val = 0.00001
gamma = gamma_val*2*pi * ones(ComplexF64,size(kappa));
pad_cells = [10;10]
gamma = absorbing_layer!(gamma, pad_cells, omega);

# Network Parameters

e_vcycle_input = true
data_augmentetion = false
cifar_kappa = true
kappa_input = true
kappa_smooth = true
models_count = 50000
gamma_input = false
random_batch = false
kernel = (3,3)
resnet = false

lr = 0.0005
opt = ADAM(lr)#), (0.9, 0.8))
iterations = 30
v2_iter = 10
level = 3

# Generate Data

train_size = 100000
test_size = 1000
batch_size = 20

test_name = replace("small resnet=$(resnet) kernel=$(kernel[1]) random_batch=$(random_batch) gamma_input=$(gamma_input) da=$(data_augmentetion) models_count=$(models_count) cifar=$(cifar_kappa) e_vcycle=$(e_vcycle_input) n=$(n) m=$(train_size) bs=$(batch_size) opt=$(SubString("$(opt)",1,findfirst("(", "$(opt)")[1]-1)) lr=$(lr) iter=$(iterations)","."=>"_")

model, train_loss, test_loss = train_residual_unet!(test_name, n, kappa, omega, gamma,
                                                    train_size, test_size, batch_size, iterations, opt;
                                                    e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion=data_augmentetion, cifar_kappa=cifar_kappa,
                                                    kappa_input=kappa_input, gamma_input=gamma_input, random_batch=random_batch, kernel=kernel)

iter = range(1, length =iterations)
p = plot(iter, train_loss, label="Train residual")
plot!(iter, test_loss, label="Test residual")
yaxis!("Loss", :log10)
xlabel!("Iterations")
savefig("test/unet_helmholtz/results/$(test_name) residual graph")

# New Example Check

(input,e_true) = generate_random_data!(1, n, kappa, omega, gamma; e_vcycle_input=e_vcycle_input)[1]
e_vcycle = input[:,:,1:2,:]
r = input[:,:,end-1:end,:]
if gamma_input==true
    input = cat(input, complex_grid_to_channels!(gamma, n), dims=3)
end
input =input|> cgpu
model_result = model(input)
model_result = model_result|>cpu

heatmap(e_true[:,:,1,1], color=:grays)
title!(L"e^{true} = x - \tilde x")
savefig("test/unet_helmholtz/results/$(test_name) e_true")

heatmap(r[:,:,1,1], color=:grays)
title!(L"r = b^{true} - A \tilde x")
savefig("test/unet_helmholtz/results/$(test_name) r")

if e_vcycle_input == true
    heatmap(real(e_vcycle[:,:,1,1]), color=:grays)
    title!(L"e^{vcycle} = Vcycle(A,r,e^{(0)}=0)")
    savefig("test/unet_helmholtz/results/$(test_name) e_vcycle")
    @info "$(Dates.format(now(), "HH:MM:SS")) - V-Cycle train error norm = $(norm_diff!(e_vcycle, e_true)), UNet train error norm = $(norm_diff!(model_result, e_true))"
else
    @info "$(Dates.format(now(), "HH:MM:SS")) - UNet train error norm = $(norm_diff!(model_result, e_true))"
end

m = 30
restrt = 20
max_iter = 4
check_model!(test_name, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=kappa_smooth)
