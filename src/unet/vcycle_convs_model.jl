include("../../src/unet/utils.jl")

# Conv Methods

function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2)|>cgpu,
	    BatchNorm(out_ch)|> cgpu,
	    x->squeeze(x))|> cgpu
end

ConvDown(in_chs,out_chs,kernel = (5,5)) =
    Chain(Conv(kernel,in_chs=>out_chs,stride=(2,2), pad = 1;init=_random_normal),
	    BatchNorm(out_chs),
	    x->elu.(x,0.2f0))|> cgpu

struct UNetUpBlock
  upsample
end

(u::UNetUpBlock)(input, bridge) = u.upsample(input)

@functor UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (5, 5), p = 0.5f0) =
    UNetUpBlock(Chain(x->elu.(x,0.2f0),
                ConvTranspose(kernel, in_chs=>out_chs, stride=(2, 2), pad = 1;init=_random_normal),
                BatchNorm(out_chs),
        		Dropout(p)))|> cgpu

struct ConvMatrix
    input_func
    matrix_func
    func
    bn
end

(u::ConvMatrix)(input, matrix) = u.bn(u.func(cat(u.input_func(input), u.matrix_func(matrix), dims = 3)))

@functor ConvMatrix

function ConvMatrix(in_i::Int, in_m::Int, out::Int; kernel = (3, 3), p = 0.5f0)
    input_func = Chain(Conv(kernel, in_i=>out, pad = 1;init=_random_normal),
                        BatchNorm(out),
                        x->elu.(x,0.2f0))|> cgpu
    matrix_func = Chain(Conv(kernel, in_m=>out, pad = 1;init=_random_normal),
                        BatchNorm(out),
                        x->elu.(x,0.2f0))|> cgpu
    func = Chain(Conv(kernel, (2*out)=>out, pad = 1;init=_random_normal),
                BatchNorm(out),
                x->elu.(x,0.2f0),
                Conv(kernel, out=>out, pad = 1;init=_random_normal))|> cgpu
    bn = BatchNorm(out)|> cgpu
    ConvMatrix(input_func,matrix_func,func,bn)
end

function Jacobi(func, n, x, b, matrix; max_iter=1, w=0.8, use_gmres_alpha=0)
    h = 1.0 ./ n
    for i in 1:max_iter
        y = func(matrix, x)
        residual = b - y
        d = (4.0 / h^2) .- sum(matrix, dims=4)
        alpha = w ./ d
        x = x + alpha * residual
    end
    return x
end

function BlockFilter(filter_size, kernel, channels)
    w = zeros(Float32, filter_size, filter_size, channels, channels)
    for i in 1:channels
        w[:,:,i,i] = kernel
    end
    return w
end

zeros_karnel = zeros(3,3)
laplacian_kernel = [0 -1 0;-1 4.0 -1;0 -1 0]
laplacian_block_kernel = BlockFilter(3, laplacian_kernel, 2)
smooth_up_kernel = (1/4) * [1 2 1;2 4.0 2;1 2 1]
smooth_up_block_kernel = BlockFilter(3, smooth_up_kernel, 2)
smooth_down_kernel = (1/16) * [1 2 1;2 4 2;1 2 1]
smooth_down_block_kernel = BlockFilter(3, smooth_down_kernel, 2)

struct GetMatrices
    real_func
    h_imag_func
    s_imag_func
end

@functor GetMatrices

function GetMatrices(alpha)
    real_func(kappa, omega, gamma_i) = (kappa .^ 2) .* omega .* (omega .+ gamma_i)
    h_imag_func(kappa, omega, gamma_r) = (kappa .^ 2) .* omega .* gamma_r
    s_imag_func(kappa, omega, gamma_r) = (kappa .* kappa .* omega .* omega .* alpha) .+ (kappa .* kappa .* omega .* gamma_r)
    GetMatrices(real_func, h_imag_func, s_imag_func)
end

function (gm::GetMatrices)(kappa, omega, gamma_r, gamma_i)
    i_conv = Conv(reshape([1.0],1,1,1,1),[0.0])|> cgpu

    s_h_real = i_conv(gm.real_func(i_conv(kappa), omega, i_conv(gamma_i)))
    h_imag = i_conv(gm.h_imag_func(i_conv(kappa), omega, i_conv(gamma_r)))
    s_imag = i_conv(gm.s_imag_func(i_conv(kappa), omega, i_conv(gamma_r)))

    h_matrix = cat(s_h_real, h_imag, dims=3) # cat(cat(s_h_real, h_imag, dims=3),cat(-1.0 * h_imag, s_h_real, dims=3),dims=4)
    s_matrix = cat(s_h_real, s_imag, dims=3) # cat(cat(s_h_real, s_imag, dims=3),cat(-1.0 * s_imag, s_h_real, dims=3),dims=4)
    return h_matrix, s_matrix
end

struct VUnet
    jacobi
    up_convs
    down_convs
    matrix_convs
    get_matrices
    n
    f
end

@functor VUnet

function VUnet()
    jacobi = Jacobi
    up_convs = Chain(UNetUpBlock(16, 8), UNetUpBlock(32, 16)) |> cgpu # ConvTranspose(smooth_up_block_kernel, [0.0], stride=2) |> cgpu #

    down_convs = Chain(ConvDown(1,1),ConvDown(8,16),ConvDown(16,32)) |> cgpu # Conv(smooth_down_block_kernel, [0.0], stride=2) |> cgpu #

    matrix_convs = Chain(ConvMatrix(2, 2, 8),
                        ConvMatrix(8, 2, 8),
                        ConvMatrix(16, 2, 16),
                        ConvMatrix(16, 16, 16),
                        ConvMatrix(32, 2, 32),
                      	ConvMatrix(8, 2,2))|> cgpu

    laplace_conv = nothing #Conv(laplacian_block_kernel, [0.0], pad=(1,1))|> cgpu
    helmholtz_conv(matrix, x) = nothing # (1.0 / (h^2)) .* i_conv(laplace_conv(x)) - sum(i_conv(matrix) .* i_conv(x), dims=4)

    get_matrices = GetMatrices([0.5]|> cgpu)
    n = 64
    f = n == 64 ? 5.0 : 10.0
    VUnet(jacobi,up_convs,down_convs,matrix_convs,get_matrices,n,f)|> cgpu
end

function (u::VUnet)(x::AbstractArray)

    # Parameters
    r = reshape(x[:,:,1:2,1],u.n-1,u.n-1,2,1)
    kappa = reshape(x[:,:,3,1],u.n-1,u.n-1,1,1)
    gamma_r = reshape(x[:,:,4,1],u.n-1,u.n-1,1,1)
    gamma_i = reshape(x[:,:,5,1],u.n-1,u.n-1,1,1)
    omega = reshape([2.0*pi*u.f]|> cgpu,1,1,1,1)

    h = 1.0 ./ u.n

    h_matrix, s_matrix = u.get_matrices(kappa, omega, gamma_r, gamma_i)
    # Relax on Ae = r v1_iter times with initial guess e
    e = u.matrix_convs[1](r,h_matrix) # 8 CHs
    residual_fine = u.matrix_convs[2](e,s_matrix) # 8 CHs
    #residual_fine = u.matrix_convs[2](e_helmholtz,r) # 8 CHs

    # Level 1
    r_coarse_1 = u.down_convs[2](residual_fine) # 16 CHs
    kappa_coarse_1 = u.down_convs[1](kappa) # 1 CHs
    gamma_r_coarse_1 = u.down_convs[1](gamma_r) # 1 CHs
    gamma_i_coarse_1 = u.down_convs[1](gamma_i) # 1 CHs
    n_coarse_1 = floor(Int32,u.n / 2)
    h_matrix_coarse_1, s_matrix_coarse_1 = u.get_matrices(kappa_coarse_1, omega, gamma_r_coarse_1, gamma_i_coarse_1)

    # Relax on Ae = r v1_iter times with initial guess e
    e_coarse_1 = u.matrix_convs[3](r_coarse_1,h_matrix_coarse_1) # 16 CHs
    r_coarse_1 = u.matrix_convs[3](e_coarse_1,s_matrix_coarse_1) # 16 CHs
    #r_coarse_1 = u.matrix_convs[4](e_helmholtz_1,r_coarse_1) # 16 CHs

    # Level 2
    r_coarse = u.down_convs[3](r_coarse_1) # 32 CHs
    kappa_coarse = u.down_convs[1](kappa_coarse_1) # 1 CHs
    gamma_r_coarse = u.down_convs[1](gamma_r_coarse_1) # 1 CHs
    gamma_i_coarse = u.down_convs[1](gamma_i_coarse_1) # 1 CHs
    n_coarse = floor(Int32,n_coarse_1 / 2)
    h_matrix_coarse, s_matrix_coarse = u.get_matrices(kappa_coarse, omega, gamma_r_coarse, gamma_i_coarse)

    # Solve Coarser
    e_coarse = u.matrix_convs[5](r_coarse,s_matrix_coarse) # 32 CHs
    e_coarse = u.matrix_convs[5](e_coarse,s_matrix_coarse) # 32 CHs

    # Correct
    e_coarse_1 = u.up_convs[2](e_coarse, e_coarse_1) # 16 CHs
    #e_coarse_1 = e_coarse_1 + e_fine_1 # 16 CHs
    e_helmholtz_1 = u.matrix_convs[3](e_coarse_1,h_matrix_coarse_1) # 16 CHs
    residual_coarse_1 = u.matrix_convs[4](e_helmholtz_1,r_coarse_1) # 16 CHs

    e = u.up_convs[1](residual_coarse_1,e) # 8 CHs
    #e = e + e_fine # 8 CHs
    e_helmholtz = u.matrix_convs[2](e,s_matrix) # 8 CHs
    e = u.matrix_convs[6](e_helmholtz,r) # 2 CHs

    return e
end
