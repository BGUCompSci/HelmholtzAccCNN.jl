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
include("../../src/unet/vcycle_unet.jl")
include("../../src/unet/data.jl")
include("../../src/unet/train.jl")
include("../../src/kappa_models.jl")
include("unet_test_utils.jl")

function test_train_vcycle!(n, f, opt, lr, train_size, test_size, batch_size, iterations; data_augmentetion=false, e_vcycle_input=false, cifar_kappa=true, kappa_input=true, kappa_smooth=true, gamma_input=true, kernel=(3,3), v2_iter=10, level=3)

    h = 1.0./n;
    kappa = ones(Float64,n-1,n-1)
    omega = 2*pi*f;
    gamma_val = 0.00001
    gamma = gamma_val*2*pi * ones(ComplexF64,size(kappa));
    pad_cells = [10;10]
    gamma = absorbing_layer!(gamma, pad_cells, omega);

    random_batch = false
    resnet = false
    test_name = replace("$(Dates.format(now(), "HH_MM_SS")) big thresh small lr resnet=$("$(resnet)"[1]) kernel=$(kernel[1]) gamma=$("$(gamma_input)"[1]) da=$("$(data_augmentetion)"[1]) cifar=$("$(cifar_kappa)"[1]) e_vcycle=$("$(e_vcycle_input)"[1]) n=$(n) f=$(f) m=$(train_size) bs=$(batch_size) opt=$(SubString("$(opt)",1,findfirst("(", "$(opt)")[1]-1)) lr=$(lr) iter=$(iterations)","."=>"_")

    model, train_loss, test_loss = train_vcycle!(test_name, n, kappa, omega, gamma,
                                                        train_size, test_size, batch_size, iterations, lr;
                                                        e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion=data_augmentetion, cifar_kappa=cifar_kappa,
                                                        kappa_input=kappa_input, gamma_input=gamma_input, random_batch=random_batch, kernel=kernel)

    iter = range(1, length =iterations)
    p = plot(iter, train_loss, label="Train residual")
    plot!(iter, test_loss, label="Test residual")
    yaxis!("Loss", :log10)
    xlabel!("Iterations")
    savefig("test/unet/results/$(test_name) residual graph")

    # New Example Check

    (input,e_true) = generate_random_data!(1, n, kappa, omega, gamma; e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level,
                                                              cifar_kappa=cifar_kappa, kappa_input=kappa_input, kappa_smooth=kappa_smooth)[1]
    e_vcycle = input[:,:,1:2,:]
    r = input[:,:,end-1:end,:]
    if gamma_input==true
        input = cat(input, complex_grid_to_channels!(gamma, n), dims=3)
    end
    input =input|> cgpu
    model_result = model(input)
    model_result = model_result|>cpu

    # heatmap(e_true[:,:,1,1], color=:grays)
    # title!(L"e^{true} = x - \tilde x")
    # savefig("test/unet_helmholtz/results/$(test_name) e_true")
    #
    # heatmap(r[:,:,1,1], color=:grays)
    # title!(L"r = b^{true} - A \tilde x")
    # savefig("test/unet_helmholtz/results/$(test_name) r")

    if e_vcycle_input == true
        # heatmap(real(e_vcycle[:,:,1,1]), color=:grays)
        # title!(L"e^{vcycle} = Vcycle(A,r,e^{(0)}=0)")
        # savefig("test/unet_helmholtz/results/$(test_name) e_vcycle")
        @info "$(Dates.format(now(), "HH:MM:SS")) - V-Cycle train error norm = $(norm_diff!(e_vcycle, e_true)), UNet train error norm = $(norm_diff!(model_result, e_true))"
    else
        @info "$(Dates.format(now(), "HH:MM:SS")) - UNet train error norm = $(norm_diff!(model_result, e_true))"
    end

    # heatmap(model_result[:,:,1,1], color=:grays)
    # title!(L"e^{unet} = Unet(r)")
    # savefig("test/unet_helmholtz/results/$(test_name) e_unet")

    m = 5
    restrt = 20
    max_iter = 4
    check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, cifar_kappa, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=kappa_smooth)
end

lr = 0.01
opt = ADAM(lr)
train_size = 10000
test_size = 10
batch_size = 20
iterations = 5

test_train_vcycle!(64, 5.0, opt, lr, train_size, test_size, batch_size, iterations;
                    data_augmentetion=false,
                    e_vcycle_input=false,
                    cifar_kappa=true,
                    kappa_input=true,
                    kappa_smooth=true,
                    gamma_input=true,
                    kernel=(3,3),
                    v2_iter=10, level=3)
