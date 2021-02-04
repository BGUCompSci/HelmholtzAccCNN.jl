# Multigrid Flux Filters

smooth_up_filter = reshape((1/4) * [1 2 1;2 4.0 2;1 2 1],3,3,1,1)
smooth_down_filter = reshape((1/16) * [1 2 1;2 4 2;1 2 1],3,3,1,1)
laplacian_filter = reshape([0 -1 0;-1 4.0 -1;0 -1 0],3,3,1,1)

function block_filter!(filter_size, kernel, channels)
    w = zeros(filter_size, filter_size, channels, channels)
    for i in 1:channels
        w[:,:,i,i] = kernel
    end
    return w
end

block_laplacian_filter = block_filter!(3, laplacian_filter, 2)

up = ConvTranspose(smooth_up_filter, [0.0], stride=2)
down = Conv(smooth_down_filter, [0.0], stride=2)

block_up = ConvTranspose(block_filter!(3, smooth_up_filter, 2), [0.0], stride=2)
block_down = Conv(block_filter!(3, smooth_down_filter, 2), [0.0], stride=2)

function laplacian_conv!(grid; h= 1)
    filter = (1.0 / (h^2)) * laplacian_filter
    conv = Conv(filter, [0.0], pad=(1,1))
    return conv(grid)
end

function helmholtz_chain!(grid, matrix; h = 1)
    filter = (1.0 / (h^2)) * laplacian_filter
    conv = Conv(filter, [0.0], pad=(1,1))
    chain = Chain(x -> conv(x) .- (matrix .* x))
    return chain(grid)
end

function helmholtz_chain_channels!(grid, matrix, n; h = 1)
    filter = (1.0 / (h^2)) * block_laplacian_filter
    conv = Conv(filter, [0.0], pad=(1,1))
    chain = Chain(x -> conv(x) .- reshape(sum(matrix .* x, dims=3), n-1, n-1, 2, 1))
    return chain(grid)
end
