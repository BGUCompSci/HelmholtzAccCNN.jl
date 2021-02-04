include("../../src/unet/utils.jl")

function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2)|>cgpu,
	  BatchNorm(out_ch)|> cgpu,
	  x->squeeze(x))|> cgpu
end

UNetConvBlock(in_chs, out_chs; kernel = (3, 3), pad=1) =
    Chain(Conv(kernel, in_chs=>out_chs, pad=pad;init=_random_normal),
	BatchNorm(out_chs),
	x->elu.(x,0.2f0))|> cgpu

ConvDown(in_chs,out_chs,kernel = (5,5)) =
  Chain(
    #Conv(block_filter!(3, smooth_down_filter, in_chs), [0.0], stride=(2,2)),
    Conv(kernel,in_chs=>out_chs,stride=(2,2), pad = 1;init=_random_normal),
	BatchNorm(out_chs),
	x->elu.(x,0.2f0))|> cgpu

struct UNetUpBlock
  upsample
end

(u::UNetUpBlock)(input, bridge) = cat(u.upsample(input), bridge, dims = 3)

@functor UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (5, 5), p = 0.5f0) =
    UNetUpBlock(Chain(x->elu.(x,0.2f0),
                #ConvTranspose(block_filter!(3, smooth_up_filter, in_chs), [0.0], stride=(2,2)),
                #ConvTranspose((1, 1),in_chs=>out_chs;init=_random_normal),
                ConvTranspose(kernel, in_chs=>out_chs, stride=(2, 2), pad = 1;init=_random_normal),
                BatchNorm(out_chs),
        		Dropout(p)))|> cgpu

struct ResidualBlock
  layers
  shortcut
  bn
end

(r::ResidualBlock)(input) = r.bn(r.layers(input) + r.shortcut(input))

@functor ResidualBlock

function ResidualBlock(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1)
    layers = Chain(Conv(kernel, in_chs=>out_chs,pad = pad;init=_random_normal),
                	BatchNorm(out_chs),
                	x->elu.(x,0.2f0),
                    Conv(kernel, out_chs=>out_chs,pad = 1;init=_random_normal))|> cgpu
    shortcut = Conv((1,1),in_chs=>out_chs)|> cgpu
    bn = BatchNorm(out_chs)|> cgpu
    ResidualBlock(layers, shortcut, bn)|> cgpu
end

struct SUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SUnet

function SUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3))
  conv_down_blocks = Chain(ConvDown(16,16),
		      ConvDown(32,32),
		      ConvDown(64,64),
		      ConvDown(128,128))|> cgpu

  conv_blocks = Chain(UNetConvBlock(channels, 8; kernel = kernel),
		 UNetConvBlock(8, 32; kernel = kernel),
		 UNetConvBlock(32, 64; kernel = kernel),
		 UNetConvBlock(64, 128; kernel = kernel),
		 UNetConvBlock(128, 128; kernel = kernel))|> cgpu

  up_blocks = Chain(UNetUpBlock(128, 64),
		UNetUpBlock(128, 32),
		Chain(x->elu.(x,0.2f0),
		Conv((1, 1), 64=>labels;init=_random_normal)))|> cgpu
  SUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SUnet)(x::AbstractArray)
    # 63 X 63 X 4 X 1 -> 63 X 63 X 32 X 1
    op = u.conv_blocks[1:2](x)
    # 63 X 63 X 32 X 1 -> 31 X 31 X 64 X 1
    x1 = u.conv_blocks[3](u.conv_down_blocks[2](op))
    # 31 X 31 X 64 X 1 -> 15 X 15 X 128 X 1
    x2 = u.conv_blocks[4](u.conv_down_blocks[3](x1))

    # 15 X 15 X 128 X 1 -> 15 X 15 X 128 X 1
    up_x2 = u.conv_blocks[5](x2)

    # 15 X 15 X 128 X 1 -> 31 X 31 X 128 X 1
    up_x1 = u.up_blocks[1](up_x2, x1)
    # 31 X 31 X 128 X 1 -> 63 X 63 X 64 X 1
    up_x3 = u.up_blocks[2](up_x1, op)
    # 63 X 63 X 64 X 1 -> 63 X 63 X 2 X 1
    tanh.(u.up_blocks[end](up_x3))
end

# function SUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3))
#   conv_down_blocks = Chain(ConvDown(16,16),
# 		      ConvDown(32,32),
# 		      ConvDown(64,64),
# 		      ConvDown(128,128))|> cgpu
#
#   conv_blocks = Chain(UNetConvBlock(channels, 8; kernel = kernel),
# 		 UNetConvBlock(8, 64; kernel = kernel),
# 		 UNetConvBlock(64, 128; kernel = kernel),
# 		 UNetConvBlock(128, 256; kernel = kernel),
# 		 UNetConvBlock(256, 256; kernel = kernel))|> cgpu
#
#   up_blocks = Chain(UNetUpBlock(256, 128),
# 		UNetUpBlock(256, 64),
# 		Chain(x->elu.(x,0.2f0),
# 		Conv((3, 3), pad=1,128=>labels;init=_random_normal)))|> cgpu
#   SUnet(conv_down_blocks, conv_blocks, up_blocks)
# end
#
# function (u::SUnet)(x::AbstractArray)
#     # 63 X 63 X 4 X 1 -> 63 X 63 X 64 X 1
#     op = u.conv_blocks[1:2](x)
#     # 63 X 63 X 64 X 1 -> 31 X 31 X 128 X 1
#     x1 = u.conv_blocks[3](u.conv_down_blocks[3](op))
#     # 31 X 31 X 128 X 1 -> 15 X 15 X 256 X 1
#     x2 = u.conv_blocks[4](u.conv_down_blocks[4](x1))
#
#     # 15 X 15 X 256 X 1 -> 15 X 15 X 256 X 1
#     up_x2 = u.conv_blocks[5](x2)
#     #up_x2 = u.conv_blocks[5](up_x2)
#
#     # 15 X 15 X 256 X 1 -> 31 X 31 X 256 X 1
#     up_x1 = u.up_blocks[1](up_x2, x1)
#     # 31 X 31 X 256 X 1 -> 63 X 63 X 128 X 1
#     up_x3 = u.up_blocks[2](up_x1, op)
#     # 63 X 63 X 128 X 1 -> 63 X 63 X 2 X 1
#     tanh.(u.up_blocks[end](up_x3))
# end

function Base.show(io::IO, u::SUnet)
    println(io, "SUnet:")
    println(io, "  UNetConvBlock: 63 X 63 X 4 X 1 -> 63 X 63 X 32 X 1")
    println(io, "  ConvDown: 63 X 63 X 32 X 1 -> 31 X 31 X 32 X 1")
    println(io, "  UNetConvBlock: 31 X 31 X 32 X 1 -> 31 X 31 X 64 X 1")
    println(io, "  ConvDown: 31 X 31 X 64 X 1 -> 15 X 15 X 64 X 1")
    println(io, "  UNetConvBlock: 15 X 15 X 64 X 1 -> 15 X 15 X 128 X 1")
    println(io, "  UNetConvBlock: 15 X 15 X 128 X 1 -> 15 X 15 X 128 X 1")
    println(io, "  UpBlock: 15 X 15 X 128 X 1 -> 31 X 31 X 128 X 1")
    println(io, "  UpBlock: 31 X 31 X 128 X 1 -> 63 X 63 X 64 X 1")
    println(io, "  Conv: 63 X 63 X 64 X 1 -> 63 X 63 X 2 X 1")
end

function create_model!(e_vcycle_input,kappa_input,gamma_input;kernel=(3,3))
    input = 2
    if e_vcycle_input == true
        input = input+2
    end
    if kappa_input == true
        input = input+1
    end
    if gamma_input == true
        input = input+2
    end
    return SUnet(input,2;kernel=kernel)
end
