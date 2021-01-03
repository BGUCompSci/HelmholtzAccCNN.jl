using Statistics
using LinearAlgebra
using Flux
using Flux: @functor
using LaTeXStrings
using KrylovMethods
using Distributions: Normal
using Plots
using Augmentor
using CSV, DataFrames
pyplot()

include("../../src/multigrid/helmholtz_methods.jl")
include("../../src/unet/model.jl")
include("../../src/unet/data.jl")

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

shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)

# Network Parameters

#test_name = "UnetConv and Free Down Up"
#test_name = "UnetConv and Weighted Down Up"
#test_name = "ResidualConv and Free Down Up"
#test_name = "ResidualConv and Weighted Down Up"
test_name = "Only Residual Input 100"
generate_e_vcycle = false
lr = 0.001
opt = ADAM(lr)
iterations = 2

# Generate Data

train_size = 50
test_size = 5
batch_size = 10
train_set = generate_random_data!(train_size, n, kappa, omega, gamma; generate_e_vcycle=generate_e_vcycle)
test_set = generate_random_data!(test_size, n, kappa, omega, gamma; generate_e_vcycle=generate_e_vcycle)

# UNet

function loss!(input, output)
    model_result = model(input)
    return (sum((output - model_result).^2)) / (sum(output.^2))
end

function loss!(tuple)
    return loss!(tuple[1],tuple[2])
end

if generate_e_vcycle == true
    model = SUnet(4,2)
else
    model = SUnet(2)
end

batchs = Int64(train_size / batch_size)
test_loss = zeros(iterations, test_size)
train_loss = zeros(iterations, train_size)

for iteration in 1:iterations
    for epoch_idx in 1:batchs
        epoch_set = train_set[(epoch_idx-1)*batch_size+1:epoch_idx*batch_size]
        Flux.train!(loss!, Flux.params(model), epoch_set, opt);
        println(batchs*(iteration-1)+epoch_idx)
    end
    test_loss[iteration,:] = loss!.(test_set)
    train_loss[iteration,:] = loss!.(train_set)
    println("$(batchs*iteration)) Train loss value = $(mean(train_loss[iteration,:])), Test loss value = $(mean(test_loss[iteration,:]))")
end

iter = range(1, batchs*iterations, length = iterations)
p = plot(iter, mean(train_loss,dims=2), label="Train")
plot!(iter, mean(test_loss,dims=2), label="Test")
yaxis!("Loss", :log10)
xlabel!("Iterations")
savefig(replace("test/unet_helmholtz/results/$(test_name) Residual Graph","."=>"_"))

index = 3

# Train Check

e_true = train_set[index][2]
if generate_e_vcycle == true
    e_vcycle = train_set[index][1][:,:,1:2,:]
    r = train_set[index][1][:,:,3:4,:]
else
    r = train_set[index][1]
end

model_result = model(train_set[index][1])

heatmap(e_true[:,:,1,1], color=:grays)
title!(L"e^{true} = x - \tilde x")
savefig("test/unet_helmholtz/results/$(test_name) Train e_true")

heatmap(r[:,:,1,1], color=:grays)
title!(L"r = b^{true} - A \tilde x")
savefig("test/unet_helmholtz/results/$(test_name) Train r")

if generate_e_vcycle == true
    heatmap(real(e_vcycle[:,:,1,1]), color=:grays)
    title!(L"e^{vcycle} = Vcycle(A,r,e^{(0)}=0)")
    savefig("test/unet_helmholtz/results/$(test_name) Train e_vcycle")
    println("V-Cycle train error norm = $((sum((e_vcycle - e_true).^2))/ (sum(e_true.^2))), UNet train error norm = $((sum((model_result - e_true).^2))/ (sum(e_true.^2)))")
else
    println("UNet train error norm = $((sum((model_result - e_true).^2))/ (sum(e_true.^2)))")
end

heatmap(model_result[:,:,1,1], color=:grays)
title!(L"e^{net} = UNet(r)")
savefig("test/unet_helmholtz/results/$(test_name) Train e_unet")

# Test Check

e_true = test_set[index][2]
if generate_e_vcycle == true
    e_vcycle = test_set[index][1][:,:,1:2,:]
    r = test_set[index][1][:,:,3:4,:]
else
    r = test_set[index][1]
end

model_result = model(test_set[index][1])

heatmap(e_true[:,:,1,1], color=:grays)
title!(L"e^{true} = x - \tilde x")
savefig("test/unet_helmholtz/results/$(test_name) Test e_true")

heatmap(r[:,:,1,1], color=:grays)
title!(L"r = b^{true} - A \tilde x")
savefig("test/unet_helmholtz/results/$(test_name) Test r")

if generate_e_vcycle == true
    heatmap(real(e_vcycle[:,:,1,1]), color=:grays)
    title!(L"e^{vcycle} = Vcycle(A,r,e^{(0)}=0)")
    savefig("test/unet_helmholtz/results/$(test_name) Test e_vcycle")
    println("V-Cycle test error norm = $((sum((e_vcycle - e_true).^2))/ (sum(e_true.^2))), UNet train error norm = $((sum((model_result - e_true).^2))/ (sum(e_true.^2)))")
else
    println("UNet test error norm = $((sum((model_result - e_true).^2))/ (sum(e_true.^2)))")
end

heatmap(model_result[:,:,1,1], color=:grays)
title!(L"e^{net} = UNet(r)")
savefig("test/unet_helmholtz/results/$(test_name) Test e_unet")
