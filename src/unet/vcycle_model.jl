function Jacobi(func, n, x, b, matrix; max_iter=1, w=0.8, use_gmres_alpha=0)
    h = 1.0 ./ n
    for i in 1:max_iter
        y = func(h, matrix, x)
        residual = b - y
        d = (4.0 / h^2) .- sum(matrix, dims=4)
        alpha = w ./ d
        x = x + alpha .* residual
    end
    return x
end

function BlockFilter(filter_size, kernel, channels)
    w = zeros(r_type, filter_size, filter_size, channels, channels)
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
    real_func(kappa, omega, gamma_i) = (kappa .* kappa .* omega .* omega) .+ (kappa .* kappa .* omega .* gamma_i)
    h_imag_func(kappa, omega, gamma_r) = kappa .* kappa .* omega .* gamma_r
    s_imag_func(kappa, omega, gamma_r) = (kappa .* kappa .* omega .* omega .* alpha) .+ (kappa .* kappa .* omega .* gamma_r)
    GetMatrices(real_func, h_imag_func, s_imag_func)
end

function (gm::GetMatrices)(kappa, omega, gamma_r, gamma_i)
    i_conv = Conv(reshape([1.0],1,1,1,1),[0.0])|> cgpu

    s_h_real = i_conv(gm.real_func(i_conv(kappa), omega, i_conv(gamma_i)))
    h_imag = i_conv(gm.h_imag_func(i_conv(kappa), omega, i_conv(gamma_r)))
    s_imag = i_conv(gm.s_imag_func(i_conv(kappa), omega, i_conv(gamma_r)))

    h_matrix = cat(cat(s_h_real, h_imag, dims=3),cat(-1.0 * h_imag, s_h_real, dims=3),dims=4)
    s_matrix = cat(cat(s_h_real, s_imag, dims=3),cat(-1.0 * s_imag, s_h_real, dims=3),dims=4)
    return h_matrix, s_matrix
end

struct VcycleUnet
    jacobi
    up_block_conv
    down_block_conv
    down_conv
    helmholtz_conv
    get_matrices
    n
    f
end

@functor VcycleUnet

function VcycleUnet()
    jacobi = Jacobi
    up_block_conv = ConvTranspose((5,5), 2=>2, stride=(2, 2), pad = 1)|> cgpu # ConvTranspose(smooth_up_block_kernel, [0.0], stride=2) |> cgpu #
    down_block_conv = Conv((5,5), 2=>2, stride=(2,2), pad = 1)|> cgpu # Conv(smooth_down_block_kernel, [0.0], stride=2) |> cgpu #
    down_conv = Conv((5,5),1=>1,stride=(2,2), pad = 1)|> cgpu # Conv(BlockFilter(3, smooth_down_kernel, 1), [0.0], stride=2) |> cgpu #
    i_conv = Conv((1,1),2=>2)|> cgpu
    laplace_conv  = Conv((3,3), 2=>2, pad=1)|> cgpu # Conv(laplacian_block_kernel, [0.0], pad=(1,1))
    helmholtz_conv(h, matrix, x) = (1.0 / (h^2)) .* i_conv(laplace_conv(x)) - sum(i_conv(matrix) .* i_conv(x), dims=4) # ((1.0 / (h^2)) .*
    get_matrices = GetMatrices([0.5]|> cgpu)
    n = 64
    f = n == 64 ? 5.0 : 10.0
    VcycleUnet(jacobi,up_block_conv,down_block_conv,down_conv,helmholtz_conv,get_matrices,n,f)|> cgpu
end

function (u::VcycleUnet)(x::AbstractArray)

    # Parameters
    r = reshape(x[:,:,1:2,1],u.n-1,u.n-1,2,1)
    gamma_r = reshape(x[:,:,3,1],u.n-1,u.n-1,1,1)
    gamma_i = reshape(x[:,:,4,1],u.n-1,u.n-1,1,1)
    kappa = reshape(x[:,:,5,1],u.n-1,u.n-1,1,1)
    omega = reshape([2.0*pi*u.f]|> cgpu,1,1,1,1)

    h = 1.0 ./ u.n

    h_matrix, s_matrix = u.get_matrices(kappa, omega, gamma_r, gamma_i)

    # Relax on Ae = r v1_iter times with initial guess e
    e = zeros(r_type, u.n-1, u.n-1, 2, 1)|> cgpu
    e = u.jacobi(u.helmholtz_conv, u.n, e, r, s_matrix; max_iter=1)
    residual_fine = r - u.helmholtz_conv(h, h_matrix, e)

    # Level 1
    r_coarse_1 = u.down_block_conv(residual_fine)
    kappa_coarse_1 = u.down_conv(kappa)
    gamma_r_coarse_1 = u.down_conv(gamma_r)
    gamma_i_coarse_1 = u.down_conv(gamma_i)
    n_coarse_1 = floor(Int32,u.n / 2)
    h_matrix_coarse_1, s_matrix_coarse_1 = u.get_matrices(kappa_coarse_1, omega, gamma_r_coarse_1, gamma_i_coarse_1)

    # Relax on Ae = r v1_iter times with initial guess e
    e_coarse_1 = zeros(r_type, n_coarse_1-1, n_coarse_1-1, 2, 1)|> cgpu
    e_coarse_1 = u.jacobi(u.helmholtz_conv, n_coarse_1, e_coarse_1, r_coarse_1, s_matrix_coarse_1; max_iter=1)
    residual_coarse_1 = r_coarse_1 - u.helmholtz_conv(2.0*h, h_matrix_coarse_1, e_coarse_1)

    # Level 2
    r_coarse = u.down_block_conv(r_coarse_1)
    kappa_coarse = u.down_conv(kappa_coarse_1)
    gamma_r_coarse = u.down_conv(gamma_r_coarse_1)
    gamma_i_coarse = u.down_conv(gamma_i_coarse_1)
    n_coarse = floor(Int32,n_coarse_1 / 2)
    h_matrix_coarse, s_matrix_coarse = u.get_matrices(kappa_coarse, omega, gamma_r_coarse, gamma_i_coarse)

    # Solve Coarser
    e_coarse = zeros(r_type, n_coarse-1, n_coarse-1, 2, 1)|> cgpu
    e_coarse = u.jacobi(u.helmholtz_conv, n_coarse, e_coarse, r_coarse, s_matrix_coarse; max_iter=3)

    # Correct
    e_fine_1 = u.up_block_conv(e_coarse)
    e_coarse_1 = e_coarse_1 + e_fine_1
    e_coarse_1 = u.jacobi(u.helmholtz_conv, n_coarse_1, e_coarse_1, r_coarse_1, s_matrix_coarse_1; max_iter=1)

    e_fine = u.up_block_conv(e_coarse_1)
    e = e + e_fine
    e = u.jacobi(u.helmholtz_conv, u.n, e, r, s_matrix; max_iter=1)

    return e
end
