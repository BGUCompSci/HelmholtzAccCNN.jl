
smooth_up_filter = r_type.( reshape((1/4) * [1 2 1;2 4.0 2;1 2 1],3,3,1,1))
smooth_down_filter =r_type.( reshape((1/16) * [1 2 1;2 4 2;1 2 1],3,3,1,1))
laplacian_filter = r_type.(reshape([0 -1 0;-1 4.0 -1;0 -1 0],3,3,1,1))

function block_filter!(filter_size, kernel, channels)
    w = zeros(r_type, filter_size, filter_size, channels, channels)
    for i in 1:channels
        w[:,:,i,i] = kernel
    end
    return w
end

function big_block_filter!(filter_size, kernel, channels)
    w = u_type.(zeros(r_type, filter_size, filter_size, channels, channels))|>cgpu
    for i in 1:channels
        w[:,:,:,i] = u_type.(kernel)|>cgpu
    end
    return w
end

block_laplacian_filter = block_filter!(3, laplacian_filter, 2)

up = ConvTranspose(smooth_up_filter, r_type.([0.0]), stride=2)|> pu
down = Conv(smooth_down_filter, r_type.([0.0]), stride=2)|> pu

block_up = ConvTranspose(block_filter!(3, smooth_up_filter, 2), r_type.([0.0]), stride=2)
block_down = Conv(block_filter!(3, smooth_down_filter, 2), r_type.([0.0]), stride=2)

i_conv = Conv(block_filter!(1, reshape([1.0],1,1,1,1), 2),r_type.([0.0]))|> pu

function laplacian_conv!(grid; h= 1)
    filter = r_type.((1.0 / (h^2)) * laplacian_filter)
    conv = Conv(filter, r_type.([0.0]), pad=(1,1))
    return conv(grid)
end

function helmholtz_chain!(grid, matrix; h = 1)
    filter = r_type.((1.0 / (h^2)) * laplacian_filter)
    conv = Conv(filter, r_type.([0.0]), pad=(1,1))|>pu
    y = conv(grid|>pu) - matrix .* grid
    return y
end

function helmholtz_chain_channels!(grid, matrix; h = 1)
    filter = r_type.((1.0 / (h^2)) * block_laplacian_filter)
    conv = Conv(filter, [0.0], pad=(1,1))|> pu
    y = conv(grid)-sum(matrix .* grid, dims=4)
    return y
end
