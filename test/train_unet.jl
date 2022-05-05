using Statistics
using LinearAlgebra
using Flux
using Flux: @functor
using LaTeXStrings
using KrylovMethods
using Distributions: Normal
using BSON: @load
using Plots
using CSV, DataFrames
using Dates
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
r_type = Float32
c_type = ComplexF32
u_type = Float32
gmres_type = ComplexF64
a_type = CuArray{gmres_type}
a_type = Array{gmres_type}

include("../src/multigrid/helmholtz_methods.jl")
include("../src/unet/model.jl")
include("../src/unet/data.jl")
include("../src/unet/train.jl")
include("../src/kappa_models.jl")
include("test_utils.jl")

fgmres_func = KrylovMethods.fgmres # gpu_flexible_gmres #

function test_train_unet!(n, f, opt, init_lr, train_size, test_size, batch_size, iterations;
                                    is_save=false, data_augmentetion=false, e_vcycle_input=false,
                                    kappa_type=1, threshold=50, kappa_input=true, kappa_smooth=false, k_kernel=3,
                                    gamma_input=true, kernel=(3,3), smaller_lr=10, v2_iter=10, level=3,
                                    axb=false, norm_input=false, model_type=SUnet, k_type=NaN, resnet_type=SResidualBlock, k_chs=-1, indexes=3, data_path="", full_loss=false, residual_loss=false, gmres_restrt=1, σ=elu, arch=1)

    h = 1.0./n;
    gamma_val = 0.00001
    pad_cells = [10;10]
    kappa = r_type.(ones(r_type,n-1,n-1)|>pu)
    omega = r_type(2*pi*f);
    gamma = gamma_val*2*pi * ones(r_type,size(kappa))
    gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))|>pu
    test_name = replace("$(Dates.format(now(), "HH_MM_SS")) RADAM ND $(model_type) $(k_type) $(resnet_type) $(k_chs) $(σ) $(indexes) $(k_kernel) g=$(gmres_restrt) t=$(u_type) g=$("$(gamma_input)"[1]) e=$("$(e_vcycle_input)"[1]) r=$("$(residual_loss)"[1]) k=$(kappa_type) $(threshold) n=$(n) f=$(f) m=$(train_size) bs=$(batch_size) lr=$(init_lr) each=$(smaller_lr) i=$(iterations)","."=>"_")
    model = create_model!(e_vcycle_input, kappa_input, gamma_input; kernel=kernel, type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)|>cgpu

    model, train_loss, test_loss = train_residual_unet!(model, test_name, n, n, f, kappa, omega, gamma,
                                                        train_size, test_size, batch_size, iterations, init_lr;
                                                        e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion=data_augmentetion,
                                                        kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, k_kernel=k_kernel,
                                                        gamma_input=gamma_input, kernel=kernel, smaller_lr=smaller_lr, axb=axb, jac=false, norm_input=norm_input, model_type=model_type, k_type=k_type, k_chs=k_chs, indexes=indexes,
                                                        data_path=data_path, full_loss=full_loss, residual_loss=residual_loss, gmres_restrt=gmres_restrt,σ=σ)

    iter = range(1, length=iterations)
    p = plot(iter, train_loss, label="Train loss")
    plot!(iter, test_loss, label="Test loss")
    yaxis!("Loss", :log10)
    xlabel!("Iterations")
    savefig("test/unet/results/$(test_name) residual graph")

    # New Example Check

    (input,e_true) = generate_random_data!(1, n, kappa, omega, gamma; e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level,
                                                          kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, axb=axb, norm_input=norm_input, gmres_restrt=gmres_restrt)[1]
    e_vcycle = input[:,:,1:2,:]
    r = input[:,:,end-1:end,:]
    if gamma_input == true
        input = cat(input, gamma, dims=3)
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

    m = 10
    restrt = 10
    max_iter = 6
    check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, 4, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=v2_iter, level=level, smooth=kappa_smooth, threshold=threshold, axb=axb, norm_input=norm_input, before_jacobi=false, indexes=indexes, arch=arch)
end

init_lr = 0.0001
opt = RADAM(init_lr)
train_size = 25000
test_size = 100
batch_size = 20
iterations = 120
full_loss = false
gmres_restrt = -1 # 1 -Default, 5 - 5GMRES, -1 Random


test_train_unet!(128, 10.0, opt, init_lr, train_size, test_size, batch_size, iterations;
                    data_augmentetion = false,
                    e_vcycle_input = false,
                    kappa_type = 1,
                    kappa_input = true,
                    threshold = 25,
                    kappa_smooth = true,
                    k_kernel = 5,
                    gamma_input = true,
                    kernel = (3,3),
                    smaller_lr = 48,
                    v2_iter = 10,
                    level = 3,
                    axb = false,
                    norm_input = false,
                    model_type = FFSDNUnet,
                    k_type = TFFKappa,
                    resnet_type = TSResidualBlockI,
                    k_chs = 10,
                    arch = 2,
                    indexes = 3,
                    full_loss = full_loss,
                    residual_loss = false,
                    data_path = "",
                    gmres_restrt = gmres_restrt,
                    σ = elu)
