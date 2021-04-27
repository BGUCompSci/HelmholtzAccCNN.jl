include("../../src/unet/utils.jl")

function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2)|>cgpu,
	  BatchNorm(out_ch)|> cgpu,
	  x->squeeze(x))|> cgpu
end

UNetConvBlock(in_chs, out_chs; kernel = (3, 3), pad=1) =
    Chain(Conv(kernel, in_chs=>out_chs, pad=pad; init=_random_normal),
	BatchNorm(out_chs),
	x->elu.(x,0.2f0))|> cgpu

ConvDown(in_chs,out_chs,kernel = (5,5)) =
  Chain(
    #Conv(block_filter!(3, smooth_down_filter, in_chs), [0.0], stride=(2,2)),
    Conv(kernel, in_chs=>out_chs, stride=(2,2), pad = 1; init=_random_normal),
	BatchNorm(out_chs),
	x->elu.(x,0.2f0))|> cgpu

ConvUp(in_chs,out_chs,kernel = (5,5)) =
        Chain(x->elu.(x,0.2f0),
            ConvTranspose(kernel, in_chs=>out_chs, stride=(2, 2), pad = 1; init=_random_normal),
            BatchNorm(out_chs),
            Dropout(p))|> cgpu

struct UNetUpBlock
  upsample
end

(u::UNetUpBlock)(input, bridge) = cat(u.upsample(input), bridge, dims = 3)

@functor UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (5, 5), p = 0.5f0) =
    UNetUpBlock(Chain(x->elu.(x,0.2f0),
                #ConvTranspose(block_filter!(3, smooth_up_filter, in_chs), [0.0], stride=(2,2)),
                #ConvTranspose((1, 1),in_chs=>out_chs;init=_random_normal),
                ConvTranspose(kernel, in_chs=>out_chs, stride=(2, 2), pad = 1; init=_random_normal),
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
                    Conv(kernel, out_chs=>out_chs,pad = pad; init=_random_normal))|> cgpu
    shortcut = Conv((1,1),in_chs=>out_chs)|> cgpu
    bn = BatchNorm(out_chs)|> cgpu
    ResidualBlock(layers, shortcut, bn)|> cgpu
end

struct SResidualBlock
  layers
  activation
end

(r::SResidualBlock)(input) = r.activation(r.layers(input) + input)

@functor SResidualBlock

function SResidualBlock(chs::Int; kernel = (3, 3), pad =1)
    layers = Chain(Conv(kernel, chs=>chs, pad = pad; init=_random_normal),
                	BatchNorm(chs),
                	x->elu.(x,0.2f0),
                    Conv(kernel, chs=>chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(chs),
                    x->elu.(x,0.2f0))|> cgpu
    SResidualBlock(layers, activation)|> cgpu
end

struct SUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SUnet

function SUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3))
    conv_down_blocks = Chain(ConvDown(16,32),
		      ConvDown(32,64),
		      ConvDown(64,128))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel),
		 ResidualBlock(8, 32; kernel = kernel),
		 ResidualBlock(64, 64; kernel = kernel),
		 ResidualBlock(128, 128; kernel = kernel))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64),
		UNetUpBlock(128, 32),
		Chain(x->elu.(x,0.2f0),
		Conv((3, 3), pad = 1, 64=>labels;init=_random_normal)))|> cgpu
    SUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SUnet)(x::AbstractArray)

    # n X n X 4 X bs -> n X n X 32 X bs
    op = u.conv_blocks[1:2](x)
    # n X n X 32 X bs -> (n/2) X (n/2) X 64 X bs
    x1 = u.conv_blocks[3](u.conv_down_blocks[2](op))
    # (n/2) X (n/2) X 64 X bs -> (n/4) X (n/4) X 128 X bs
    x2 = u.conv_blocks[4](u.conv_down_blocks[3](x1))

    # (n/4) X (n/4) X 128 X bs
    up_x2 = u.conv_blocks[4](x2)
    up_x2 = u.conv_blocks[4](up_x2)

    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 128 X bs
    up_x1 = u.up_blocks[1](up_x2, x1)
    # (n/2) X (n/2) X 128 X bs -> n X n X 64 X bs
    up_x3 = u.up_blocks[2](up_x1, op)
    # n X n X 64 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x3)

end

struct DTUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor DTUnet

function DTUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3))
    conv_down_blocks = Chain(ConvDown(16,32),
		      ConvDown(32,64),
		      ConvDown(64,128),
		      ConvDown(128,256))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel),
		 ResidualBlock(8, 32; kernel = kernel),
		 ResidualBlock(64, 64; kernel = kernel),
		 ResidualBlock(128, 128; kernel = kernel),
		 ResidualBlock(256, 256; kernel = kernel))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64),
		UNetUpBlock(128, 32),
        UNetUpBlock(256, 128),
        UNetUpBlock(256, 64),
		Chain(x->elu.(x,0.2f0),
		Conv((3, 3), pad = 1, 64=>labels;init=_random_normal)))|> cgpu

    DTUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::DTUnet)(x::AbstractArray)

    # n X n X 4 X bs -> n X n X 32 X bs
    op = u.conv_blocks[1:2](x)
    # n X n X 32 X bs -> (n/2) X (n/2) X 64 X bs
    x1 = u.conv_blocks[3](u.conv_down_blocks[2](op))
    # (n/2) X (n/2) X 64 X bs -> (n/4) X (n/4) X 128 X bs
    x2 = u.conv_blocks[4](u.conv_down_blocks[3](x1))
    # (n/4) X (n/4) X 128 X bs -> (n/8) X (n/8) X 256 X bs
    x3 = u.conv_blocks[5](u.conv_down_blocks[4](x2))

    # (n/8) X (n/8) X 256 X bs
    up_x3 = u.conv_blocks[5](x3)
    up_x3 = u.conv_blocks[5](up_x3)

    # (n/8) X (n/8) X 256 X bs -> (n/4) X (n/4) X 256 X bs
    up_x1 = u.up_blocks[3](up_x3, x2)
    # (n/4) X (n/4) X 256 X bs -> (n/2) X (n/2) X 128 X bs
    up_x2 = u.up_blocks[4](up_x1, x1)
    # (n/2) X (n/2) X 128 X bs -> n X n X 64 X bs
    up_x4 = u.up_blocks[2](up_x2, op)
    # n X n X 64 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x4)

end

struct DNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor DNUnet

function DNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3))
    conv_down_blocks = Chain(ConvDown(16,8),
              ConvDown(16,32),
              ConvDown(32,64),
              ConvDown(64,128))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel),
         ResidualBlock(8, 16; kernel = kernel),
         ResidualBlock(32, 32; kernel = kernel),
         ResidualBlock(64, 64; kernel = kernel),
         ResidualBlock(128, 128; kernel = kernel))|> cgpu

    up_blocks = Chain(UNetUpBlock(64, 32),
        UNetUpBlock(64, 16),
        UNetUpBlock(128, 64),
        UNetUpBlock(128, 32),
        Chain(x->elu.(x,0.2f0),
        Conv((3, 3), pad = 1, 32=>labels;init=_random_normal)))|> cgpu

    DNUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::DNUnet)(x::AbstractArray)

    # n X n X 4 X bs -> n X n X 16 X bs
    op = u.conv_blocks[1:2](x)
    # n X n X 16 X bs -> (n/2) X (n/2) X 32 X bs
    x1 = u.conv_blocks[3](u.conv_down_blocks[2](op))
    # (n/2) X (n/2) X 32 X bs -> (n/4) X (n/4) X 64 X bs
    x2 = u.conv_blocks[4](u.conv_down_blocks[3](x1))
    # (n/4) X (n/4) X 64 X bs -> (n/8) X (n/8) X 128 X bs
    x3 = u.conv_blocks[5](u.conv_down_blocks[4](x2))

    # (n/8) X (n/8) X 128 X bs
    up_x3 = u.conv_blocks[5](x3)
    up_x3 = u.conv_blocks[5](up_x3)

    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 128 X bs
    up_x1 = u.up_blocks[3](up_x3, x2)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 64 X bs
    up_x2 = u.up_blocks[4](up_x1, x1)
    # (n/2) X (n/2) X 128 X bs -> n X n X 32 X bs
    up_x4 = u.up_blocks[2](up_x2, op)
    # n X n X 32 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x4)

end

struct DDTUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor DDTUnet

function DDTUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3))
    conv_down_blocks = Chain(ConvDown(16,32),
		      ConvDown(32,64),
		      ConvDown(64,128),
		      ConvDown(128,256),
		      ConvDown(256,512))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel),
		 ResidualBlock(8, 32; kernel = kernel),
		 ResidualBlock(64, 64; kernel = kernel),
		 ResidualBlock(128, 128; kernel = kernel),
		 ResidualBlock(256, 256; kernel = kernel),
		 ResidualBlock(512, 512; kernel = kernel))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64),
		UNetUpBlock(128, 32),
        UNetUpBlock(256, 128),
        UNetUpBlock(256, 64),
        UNetUpBlock(512, 256),
        UNetUpBlock(512, 128),
		Chain(x->elu.(x,0.2f0),
		Conv((3, 3), pad = 1, 64=>labels;init=_random_normal)))|> cgpu

    DDTUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::DDTUnet)(x::AbstractArray)

    # n X n X 4 X bs -> n X n X 32 X bs
    op = u.conv_blocks[1:2](x)
    # n X n X 32 X bs -> (n/2) X (n/2) X 64 X bs
    x1 = u.conv_blocks[3](u.conv_down_blocks[2](op))
    # (n/2) X (n/2) X 64 X bs -> (n/4) X (n/4) X 128 X bs
    x2 = u.conv_blocks[4](u.conv_down_blocks[3](x1))
    # (n/4) X (n/4) X 128 X bs -> (n/8) X (n/8) X 256 X bs
    x3 = u.conv_blocks[5](u.conv_down_blocks[4](x2))
    # (n/8) X (n/8) X 256 X bs -> (n/16) X (n/16) X 512 X bs
    x4 = u.conv_blocks[6](u.conv_down_blocks[5](x3))

    # (n/16) X (n/16) X 512 X bs
    up_x4 = u.conv_blocks[6](x4)
    up_x4 = u.conv_blocks[6](up_x4)

    # (n/16) X (n/16) X 512 X bs -> (n/8) X (n/8) X 512 X bs
    up_x1 = u.up_blocks[5](up_x4, x3)
    # (n/8) X (n/8) X 512 X bs -> (n/4) X (n/4) X 256 X bs
    up_x2 = u.up_blocks[6](up_x1, x2)
    # (n/4) X (n/4) X 256 X bs -> (n/2) X (n/2) X 128 X bs
    up_x3 = u.up_blocks[4](up_x2, x1)
    # (n/2) X (n/2) X 128 X bs -> n X n X 64 X bs
    up_x5 = u.up_blocks[2](up_x3, op)
    # n X n X 64 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

struct SDNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SDNUnet

function SDNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3))
    conv_down_blocks = Chain(ConvDown(channels,16),
              ConvDown(16,32),
              ConvDown(32,64),
              ConvDown(64,128))|> cgpu

    conv_blocks = Chain(SResidualBlock(32, 32; kernel = kernel),
         SResidualBlock(64, 64; kernel = kernel),
         SResidualBlock(128, 128; kernel = kernel))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64),
                        UNetUpBlock(128, 32),
                        UNetUpBlock(64, 16),
                        ConvUp(32, 16),
                        Chain(x->elu.(x,0.2f0),
                                Conv((3, 3), pad = 1, 16=>labels;init=_random_normal)))|> cgpu
    SDNUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SDNUnet)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_down_blocks[1](x)
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[1](u.conv_down_blocks[2](op))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[2](u.conv_down_blocks[3](x1))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[3](u.conv_down_blocks[4](x2))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[3](x3)
    up_x3 = u.conv_blocks[3](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](up_x3, x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](up_x1, x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](up_x2, op)
    # (n/2) X (n/2) X 32 X bs -> n X n X 16 X bs
    up_x5 = u.up_blocks[4](up_x4)
    # n X n X 16 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

function Base.show(io::IO, u::SUnet)
    println(io, "SUnet:")
    println(io, "  UNetConvBlock: n X n X 4 X bs -> n X n X 32 X bs")
    println(io, "  ConvDown: n X n X 32 X bs -> (n/2) X (n/2) X 64 X bs")
    println(io, "  UNetConvBlock: (n/2) X (n/2) X 64 X bs -> (n/2) X (n/2) X 64 X bs")
    println(io, "  ConvDown: (n/2) X (n/2) X 64 X bs -> (n/4) X (n/4) X 128 X bs")
    println(io, "  UNetConvBlock: (n/4) X (n/4) X 128 X bs")
    println(io, "  UNetConvBlock: (n/4) X (n/4) X 128 X bs")
    println(io, "  UNetConvBlock: (n/4) X (n/4) X 128 X bs")
    println(io, "  UpBlock: (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 128 X bs")
    println(io, "  UpBlock: (n/2) X (n/2) X 128 X bs -> n X n X 64 X bs")
    println(io, "  Conv:  n X n X 64 X bs ->  n X n X 2 X bs")
end

function create_model!(e_vcycle_input,kappa_input,gamma_input;kernel=(3,3),type=SUnet)
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
    return type(input,2;kernel=kernel)
end
