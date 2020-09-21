# Multigrid Flux Filters

smooth_up_filter = reshape((1/4)*[1 2 1;2 4.0 2;1 2 1],3,3,1,1)
smooth_down_filter = reshape((1/16)*[1 2 1;2 4 2;1 2 1],3,3,1,1)
laplacian_filter = [0 -1 0;-1 4.0 -1;0 -1 0]

up = ConvTranspose(smooth_up_filter, [0.0], stride=2)
down = Conv(smooth_down_filter, [0.0], stride=2)

function laplacian_conv!(grid; h= 1)
    filter = reshape((1.0 / (h^2)) * laplacian_filter,3,3,1,1)
    conv = Conv(filter, [0.0], pad=(1,1))
    return conv(grid)
end

function helmholtz_chain!(grid, matrix; h= 1)
    filter = reshape((1.0 / (h^2)) * laplacian_filter,3,3,1,1)
    conv = Conv(filter, [0.0], pad=(1,1))
    chain = Chain(x -> conv(x) .- (matrix .* x))
    return chain(grid)
end
