using Statistics
using LinearAlgebra
using Flux
using Flux: @functor
using LaTeXStrings
using KrylovMethods
using Distributions: Normal
using Plots
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

# UNet

helmholtz_channels = complex_helmholtz_to_channels!(helmholtz_matrix, n)

function loss!(input, output)
    model_result = model(input)
    return norm(output - helmholtz_chain_channels!(model_result, helmholtz_channels, n; h=h))/norm(output)
end

function loss!(tuple)
    return loss!(tuple[1],tuple[2])
end

train_size = 50
test_size = 5
batch_size = 5

train_set = generate_point_source_data!(train_size, n, kappa, omega, gamma; save=false, path="test/dataset/train")
test_set = generate_point_source_data!(test_size, n, kappa, omega, gamma; save=false, path ="test/dataset/test")
# train_set = read_data!("test/dataset/train",train_size,n)
# test_set = read_data!("test/dataset/test",test_size,n)

model = SUnet(2)

lr = 0.001
opt = ADAM(lr)

batchs = Int64(train_size / batch_size)
iterations = 5
test_loss = zeros(iterations, test_size)
train_loss = zeros(iterations, train_size)

for iteration in 1:iterations
    for epoch_idx in 1:batchs
        epochset = trainset[(epoch_idx-1)*batch_size+1:epoch_idx*batch_size]
        Flux.train!(loss!, Flux.params(model), epochset, opt);
    end
    test_loss[iteration,:] = loss!.(test_set)
    train_loss[iteration,:] = loss!.(train_set)
    println("$(batchs*iteration)) Train loss value = $(mean(train_loss[iteration,:])), Test loss value = $(mean(test_loss[iteration,:]))")
end

test_name = "unet point source $(SubString("$(opt)",1,findfirst("(", "$(opt)")[1]-1)) $(lr) $(iterations)"

iter = range(1, batchs*iterations, length = iterations)
p = plot(iter, mean(train_loss,dims=2), label="Train")
plot!(iter, mean(test_loss,dims=2), label="Test")
yaxis!("Loss", :log10)
xlabel!("Iterations")
savefig(replace("test/unet_helmholtz/results/$(test_name) residual graph","."=>"_"))

index = 3

# Train Check

x_channels = test_set[index][1]
b_channels = test_set[index][2]
heatmap(x_channels[:,:,1,1], color=:grays)
heatmap(b_channels[:,:,1,1], color=:grays)
model_result = model(x_channels)
heatmap(model_result[:,:,1,1], color=:grays)
savefig(replace("test/unet_helmholtz/results/$(test_name) unet result real","."=>"_"))

heatmap(model_result[:,:,2,1], color=:grays)
savefig(replace("test/unet_helmholtz/results/$(test_name) unet result imag","."=>"_"))

last_res_input = helmholtz_chain_channels!(x_channels, helmholtz_channels, n; h=h)
heatmap(real(last_res_input[:,:,1,1] - b_channels[:,:,1,1]) , color=:grays)
savefig(replace("test/unet_helmholtz/results/$(test_name) vcycle residual real","."=>"_"))

last_res_output = helmholtz_chain_channels!(model_result, helmholtz_channels, n; h=h)
heatmap(real(last_res_output[:,:,1,1]- b_channels[:,:,1,1]), color=:grays)
savefig(replace("test/unet_helmholtz/results/$(test_name) unet residual real","."=>"_"))

heatmap(real(last_res_output[:,:,2,1]- b_channels[:,:,2,1]), color=:grays)
savefig(replace("test/unet_helmholtz/results/$(test_name) unet residual imag","."=>"_"))
