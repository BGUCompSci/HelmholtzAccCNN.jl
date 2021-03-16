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

r_type = Float64
c_type = ComplexF64
u_type = Float16

include("../../src/multigrid/helmholtz_methods.jl")
include("../../src/unet/model.jl")
include("../../src/unet/data.jl")
include("../../src/unet/train.jl")
include("../../src/kappa_models.jl")
include("unet_test_utils.jl")


function test_train_unet!(n, f, opt, init_lr, train_size, test_size, batch_size, iterations;
                                    is_save=false, data_augmentetion=false, e_vcycle_input=false,
                                    kappa_type=1, threshold=50, kappa_input=true, kappa_smooth=false,
                                    gamma_input=true, kernel=(3,3), smaller_lr=10, v2_iter=10, level=3,
                                    axb=false, norm_input=false, model_type=SUnet)

    h = 1.0./n;
    kappa = ones(r_type,n-1,n-1)
    omega = 2*pi*f;
    gamma_val = 0.00001
    gamma = gamma_val*2*pi * ones(c_type,size(kappa));
    pad_cells = [10;10]
    gamma = absorbing_layer!(gamma, pad_cells, omega);

    test_name = replace("$(Dates.format(now(), "HH_MM_SS")) $(model_type) axb=$("$(axb)"[1]) norm=$("$(norm_input)"[1]) t=$(u_type) k=$(kernel[1]) $(threshold) g=$("$(gamma_input)"[1]) e=$("$(e_vcycle_input)"[1]) da=$("$(data_augmentetion)"[1]) k=$(kappa_type) n=$(n) f=$(f) m=$(train_size) bs=$(batch_size) opt=$(SubString("$(opt)",1,findfirst("(", "$(opt)")[1]-1)) lr=$(init_lr) each=$(smaller_lr) i=$(iterations)","."=>"_")

    model, train_loss, test_loss = train_residual_unet!(test_name, n, kappa, omega, gamma,
                                                        train_size, test_size, batch_size, iterations, init_lr;
                                                        e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion=data_augmentetion,
                                                        kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth,
                                                        gamma_input=gamma_input, kernel=kernel, smaller_lr=smaller_lr, axb=axb, norm_input=norm_input, model_type=model_type)

    iter = range(1, length=iterations)
    p = plot(iter, train_loss, label="Train loss")
    plot!(iter, test_loss, label="Test loss")
    yaxis!("Loss", :log10)
    xlabel!("Iterations")
    savefig("test/unet/results/$(test_name) residual graph")

    # New Example Check

    (input,e_true) = generate_random_data!(1, n, kappa, omega, gamma; e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level,
                                                          kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, axb=axb, norm_input=norm_input)[1]
    e_vcycle = input[:,:,1:2,:]
    r = input[:,:,end-1:end,:]
    if gamma_input == true
        input = cat(input, complex_grid_to_channels!(gamma, n), dims=3)
    end
    input = input|> cgpu
    model_result = model(input)
    model_result = model_result|>cpu

    if is_save == true
        heatmap(e_true[:,:,1,1], color=:grays)
        title!(L"e^{true} = x - \tilde x")
        savefig("test/unet_helmholtz/results/$(test_name) e_true")

        heatmap(r[:,:,1,1], color=:grays)
        title!(L"r = b^{true} - A \tilde x")
        savefig("test/unet_helmholtz/results/$(test_name) r")

        heatmap(model_result[:,:,1,1], color=:grays)
        title!(L"e^{unet} = Unet(r)")
        savefig("test/unet_helmholtz/results/$(test_name) e_unet")
    end

    if e_vcycle_input == true
        if is_save == true
            heatmap(real(e_vcycle[:,:,1,1]), color=:grays)
            title!(L"e^{vcycle} = Vcycle(A,r,e^{(0)}=0)")
            savefig("test/unet_helmholtz/results/$(test_name) e_vcycle")
        end
        @info "$(Dates.format(now(), "HH:MM:SS")) - V-Cycle train error norm = $(norm_diff!(e_vcycle, e_true)), UNet train error norm = $(norm_diff!(model_result, e_true))"
    else
        @info "$(Dates.format(now(), "HH:MM:SS")) - UNet train error norm = $(norm_diff!(model_result, e_true))"
    end

    m = 30
    restrt = 20
    max_iter = 4
    check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=kappa_smooth, threshold=threshold, axb=axb, norm_input=norm_input)
end

init_lr = 0.005
opt = ADAM(init_lr)
train_size = 20000
test_size = 100
batch_size = 10
iterations = 200

test_train_unet!(128, 10.0, opt, init_lr, train_size, test_size, batch_size, iterations;
                    data_augmentetion = false,
                    e_vcycle_input = false,
                    kappa_type = 2,
                    kappa_input = true,
                    threshold = 25,
                    kappa_smooth = false,
                    gamma_input = true,
                    kernel = (3,3),
                    smaller_lr = 50,
                    v2_iter = 10,
                    level = 3,
                    axb = false,
                    norm_input = true,
                    model_type = DNUnet)
