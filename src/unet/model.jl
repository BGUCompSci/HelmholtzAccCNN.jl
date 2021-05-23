include("../../src/unet/utils.jl")

function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2)|>cgpu,
	  BatchNorm(out_ch)|> cgpu,
	  x->squeeze(x))|> cgpu
end

UNetConvBlock(in_chs, out_chs; kernel = (3, 3), pad=1, σ=elu) =
    Chain(Conv(kernel, in_chs=>out_chs, pad=pad; init=_random_normal),
	BatchNorm(out_chs),
	x-> (σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu

ConvDown(in_chs,out_chs;kernel = (5,5), σ=elu) =
  Chain(
    #Conv(block_filter!(3, smooth_down_filter, in_chs), [0.0], stride=(2,2)),
    Conv(kernel, in_chs=>out_chs, stride=(2,2), pad = 1; init=_random_normal),
	BatchNorm(out_chs),
	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu

ConvUp(in_chs,out_chs;kernel = (5,5), σ=elu) =
        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
            ConvTranspose(kernel, in_chs=>out_chs, stride=(2, 2), pad = 1; init=_random_normal),
            BatchNorm(out_chs))|> cgpu

struct UNetUpBlock
  upsample
end

(u::UNetUpBlock)(input, bridge) = cat(u.upsample(input), bridge, dims = 3)

@functor UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (5, 5), p = 0.5f0, σ=elu) =
    UNetUpBlock(Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
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

function ResidualBlock(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, in_chs=>out_chs,pad = pad;init=_random_normal),
                	BatchNorm(out_chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
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

function SResidualBlock(chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, chs=>chs, pad = pad; init=_random_normal),
                	BatchNorm(chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, chs=>chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    SResidualBlock(layers, activation)|> cgpu
end


struct SResidualBlock1
  layers
  activation
end

(r::SResidualBlock1)(input) = r.activation(r.layers(input) + input)

@functor SResidualBlock1

function SResidualBlock1(chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, chs=>chs, pad = pad; init=_random_normal),
                	BatchNorm(chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, chs=>chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(Conv(kernel, chs=>chs, pad = pad; init=_random_normal),
                    BatchNorm(chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    SResidualBlock1(layers, activation)|> cgpu
end

struct SResidualBlockI
  layers
  activation
end

(r::SResidualBlockI)(input) = r.layers(r.activation(input)) + input

@functor SResidualBlockI

function SResidualBlockI(chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, chs=>chs, pad = pad; init=_random_normal),
                	BatchNorm(chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, chs=>chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    SResidualBlockI(layers, activation)|> cgpu
end

struct SUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SUnet

function SUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(16,32;σ=σ),
		      ConvDown(32,64;σ=σ),
		      ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel, σ=σ),
		 ResidualBlock(8, 32; kernel = kernel, σ=σ),
		 ResidualBlock(64, 64; kernel = kernel, σ=σ),
		 ResidualBlock(128, 128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
		UNetUpBlock(128, 32; σ=σ),
		Chain(x-> (σ == elu ? σ.(x,0.2f0) : σ.(x)),
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

function DTUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(16,32;σ=σ),
		      ConvDown(32,64;σ=σ),
		      ConvDown(64,128;σ=σ),
		      ConvDown(128,256;σ=σ))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel, σ=σ),
		 ResidualBlock(8, 32; kernel = kernel, σ=σ),
		 ResidualBlock(64, 64; kernel = kernel, σ=σ),
		 ResidualBlock(128, 128; kernel = kernel, σ=σ),
		 ResidualBlock(256, 256; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
		UNetUpBlock(128, 32; σ=σ),
        UNetUpBlock(256, 128; σ=σ),
        UNetUpBlock(256, 64; σ=σ),
		Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
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

function DNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(16,8;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel, σ=σ),
         ResidualBlock(8, 16; kernel = kernel, σ=σ),
         ResidualBlock(32, 32; kernel = kernel, σ=σ),
         ResidualBlock(64, 64; kernel = kernel, σ=σ),
         ResidualBlock(128, 128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(64, 32; σ=σ),
        UNetUpBlock(64, 16; σ=σ),
        UNetUpBlock(128, 64; σ=σ),
        UNetUpBlock(128, 32; σ=σ),
        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
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

struct SDNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SDNUnet

function SDNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(SResidualBlock(32; kernel = kernel, σ=σ),
         SResidualBlock(64; kernel = kernel, σ=σ),
         SResidualBlock(128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
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
    # (n/2) X (n/2) X 32 X bs -> (n/2) X (n/2) X 16 X bs
    up_x5 = u.up_blocks[4](up_x4)
    # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

struct SDNUnet1
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SDNUnet1

function SDNUnet1(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(SResidualBlock(16; kernel = kernel, σ=σ),
         SResidualBlock(32; kernel = kernel, σ=σ),
         SResidualBlock(64; kernel = kernel, σ=σ),
         SResidualBlock(128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    SDNUnet1(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SDNUnet1)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](x2))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[4](x3)
    up_x3 = u.conv_blocks[4](up_x3)
    up_x3 = u.conv_blocks[4](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](up_x3, x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](up_x1, x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](up_x2, op)
    # (n/2) X (n/2) X 32 X bs -> (n/2) X (n/2) X 16 X bs
    up_x5 = u.up_blocks[4](up_x4)
    # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

struct SDNUnetI
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SDNUnetI

function SDNUnetI(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(SResidualBlockI(16; kernel = kernel, σ=σ),
         SResidualBlockI(32; kernel = kernel, σ=σ),
         SResidualBlockI(64; kernel = kernel, σ=σ),
         SResidualBlockI(128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    SDNUnetI(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SDNUnetI)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](x2))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[4](x3)
    up_x3 = u.conv_blocks[4](up_x3)
    up_x3 = u.conv_blocks[4](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](up_x3, x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](up_x1, x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](up_x2, op)
    # (n/2) X (n/2) X 32 X bs -> (n/2) X (n/2) X 16 X bs
    up_x5 = u.up_blocks[4](up_x4)
    # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

struct SDNNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SDNNUnet

function SDNNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(channels,12;σ=σ),
              ConvDown(12,24;σ=σ),
              ConvDown(24,48;σ=σ),
              ConvDown(48,96;σ=σ))|> cgpu

    conv_blocks = Chain(SResidualBlock(12; kernel = kernel, σ=σ),
         SResidualBlock(24; kernel = kernel, σ=σ),
         SResidualBlock(48; kernel = kernel, σ=σ),
         SResidualBlock(96; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(96, 48; σ=σ),
                        UNetUpBlock(96, 24; σ=σ),
                        UNetUpBlock(48, 12; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 24=>12;init=_random_normal)),
                        ConvUp(12, labels; σ=σ))|> cgpu
    SDNNUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SDNNUnet)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 12 X bs
    op = u.conv_blocks[1](u.conv_blocks[1](u.conv_down_blocks[1](x)))
    # (n/2) X (n/2) X 12 X bs -> (n/4) X (n/4) X 24 X bs
    x1 = u.conv_blocks[2](u.conv_blocks[2](u.conv_down_blocks[2](op)))
    # (n/4) X (n/4) X 24 X bs -> (n/8) X (n/8) X 48 X bs
    x2 = u.conv_blocks[3](u.conv_blocks[3](u.conv_down_blocks[3](x1)))
    # (n/8) X (n/8) X 48 X bs -> (n/16) X (n/16) X 96 X bs
    x3 = u.conv_blocks[4](u.conv_blocks[4](u.conv_down_blocks[4](x2)))

    # (n/16) X (n/16) X 96 X bs
    up_x3 = u.conv_blocks[4](x3)
    up_x3 = u.conv_blocks[4](up_x3)
    up_x3 = u.conv_blocks[4](up_x3)

    # (n/16) X (n/16) X 96 X bs -> (n/8) X (n/8) X 96 X bs
    up_x1 = u.up_blocks[1](up_x3, x2)
    # (n/8) X (n/8) X 96 X bs -> (n/4) X (n/4) X 48 X bs
    up_x2 = u.up_blocks[2](up_x1, x1)
    # (n/4) X (n/4) X 96 X bs -> (n/2) X (n/2) X 24 X bs
    up_x4 = u.up_blocks[3](up_x2, op)
    # (n/2) X (n/2) X 24 X bs -> (n/2) X (n/2) X 12 X bs
    up_x5 = u.up_blocks[4](up_x4)
    # (n/2) X (n/2) X 12 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

struct SDNUnet2
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SDNUnet2

function SDNUnet2(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(SResidualBlock1(16; kernel = kernel, σ=σ),
         SResidualBlock1(32; kernel = kernel, σ=σ),
         SResidualBlock1(64; kernel = kernel, σ=σ),
         SResidualBlock1(128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    SDNUnet2(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SDNUnet2)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](x2))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[4](x3)
    up_x3 = u.conv_blocks[4](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](up_x3, x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](up_x1, x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](up_x2, op)
    # (n/2) X (n/2) X 32 X bs -> (n/2) X (n/2) X 16 X bs
    up_x5 = u.up_blocks[4](up_x4)
    # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

struct SSUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SSUnet

function SSUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(channels,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(SResidualBlock(32; kernel = kernel, σ=σ),
         SResidualBlock(64; kernel = kernel, σ=σ),
         SResidualBlock(128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 64=>32;init=_random_normal)),
                        ConvUp(32, labels; σ=σ))|> cgpu
    SSUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SSUnet)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 32 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 32 X bs -> (n/4) X (n/4) X 64 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op))
    # (n/4) X (n/4) X 64 X bs -> (n/8) X (n/8) X 128 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1))

    # (n/8) X (n/8) X 128 X bs
    up_x2 = u.conv_blocks[3](x2)
    up_x2 = u.conv_blocks[3](up_x2)

    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 128 X bs
    up_x1 = u.up_blocks[1](up_x2, x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 64 X bs
    up_x3 = u.up_blocks[2](up_x1, op)
    # (n/2) X (n/2) X 64 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](up_x3)
    # (n/2) X (n/2) X 32 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x4)

end

struct SSUnet1
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SSUnet1

function SSUnet1(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(SResidualBlock(16; kernel = kernel, σ=σ),
         SResidualBlock(64; kernel = kernel, σ=σ),
         SResidualBlock(128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 48=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    SSUnet1(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SSUnet1)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 64 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op))
    # (n/4) X (n/4) X 64 X bs -> (n/8) X (n/8) X 128 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1))

    # (n/8) X (n/8) X 128 X bs
    up_x2 = u.conv_blocks[3](x2)
    up_x2 = u.conv_blocks[3](up_x2)

    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 128 X bs
    up_x1 = u.up_blocks[1](up_x2, x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 64 X bs
    up_x3 = u.up_blocks[2](up_x1, op)
    # (n/2) X (n/2) X 64 X bs -> (n/2) X (n/2) X 16 X bs
    up_x4 = u.up_blocks[3](up_x3)
    # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x4)

end

struct KappaNet
    conv_blocks
end

@functor KappaNet

function KappaNet(channels::Int = 1, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_blocks = Chain(ResidualBlock(channels, 32; kernel = kernel, σ=σ),
		 ResidualBlock(32, 32; kernel = kernel, σ=σ),
		 ResidualBlock(32, labels; kernel = kernel, σ=σ))|> cgpu
    KappaNet(conv_blocks)
end

function (u::KappaNet)(x::AbstractArray)
    x1 = u.conv_blocks[1:2](x)
    x2 = u.conv_blocks[2](x1)
    x3 = u.conv_blocks[2](x2)
    x4 = u.conv_blocks[2](x3)
    x5 = u.conv_blocks[2](x4)
    u.conv_blocks[3](x5)
end

struct SplitUNet
  kappa_subnet
  solve_subnet
  indexes
end

@functor SplitUNet

function SplitUNet(in_chs, k_chs, s_model, k_model; kernel = (3, 3), indexes=3, σ=elu)
    kappa_subnet = k_model(indexes-2,k_chs;kernel=kernel,σ=σ)
    solve_subnet = s_model(k_chs+in_chs,2;kernel=kernel,σ=σ)
    # @info "$(Dates.format(now(), "HH:MM:SS")) - SplitUNet $(indexes-2) -> $(k_chs), $(k_chs+in_chs) -> 2"
    SplitUNet(kappa_subnet, solve_subnet, indexes)
end

function (u::SplitUNet)(x::AbstractArray)
    kappa = reshape(x[:,:,3:u.indexes,1], size(x)[1], size(x)[1], u.indexes-2, 1)
    kappa_features = u.kappa_subnet(kappa)
    # @info "$(Dates.format(now(), "HH:MM:SS")) - SplitUNet $(size(kappa_features)) -> $(size(cat(x, kappa_features, dims=3)))"
    u.solve_subnet(cat(x, kappa_features, dims=3))
end

struct FKappa
  conv_down_blocks
  conv_blocks
end

@functor FKappa

function FKappa(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(16,32;σ=σ),
		      ConvDown(32,64;σ=σ),
		      ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel, σ=σ),
		 ResidualBlock(8, 16; kernel = kernel, σ=σ),
		 ResidualBlock(32, 32; kernel = kernel, σ=σ),
		 ResidualBlock(64, 64; kernel = kernel, σ=σ),
		 ResidualBlock(128, 128; kernel = kernel, σ=σ))|> cgpu

    FKappa(conv_down_blocks, conv_blocks)
end

function (u::FKappa)(x::AbstractArray)

    # n X n X 4 X bs -> n X n X 16 X bs
    op = u.conv_blocks[1:2](x)
    # n X n X 16 X bs -> (n/2) X (n/2) X 32 X bs
    x1 = u.conv_blocks[3](u.conv_blocks[3](u.conv_down_blocks[1](op)))
    # (n/2) X (n/2) X 32 X bs -> (n/4) X (n/4) X 64 X bs
    x2 = u.conv_blocks[4](u.conv_blocks[4](u.conv_down_blocks[2](x1)))
    # (n/4) X (n/4) X 64 X bs -> (n/8) X (n/8) X 128 X bs
    x3 = u.conv_blocks[5](u.conv_blocks[5](u.conv_down_blocks[3](x2)))

    return op, x1, x2, x3
end

struct FSDNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor FSDNUnet

function FSDNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(20,16;σ=σ),
              ConvDown(48,32;σ=σ),
              ConvDown(96,64;σ=σ),
              ConvDown(192,128;σ=σ))|> cgpu

    conv_blocks = Chain(SResidualBlock(16; kernel = kernel, σ=σ),
         SResidualBlock(32; kernel = kernel, σ=σ),
         SResidualBlock(64; kernel = kernel, σ=σ),
         SResidualBlock(128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    FSDNUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::FSDNUnet)(x::AbstractArray, features_n, features_n2, features_n4, features_n8)

    # n X n X 3 X bs + n X n X 16 X bs-> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](cat(x, features_n, dims=3)))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](cat(op, features_n2, dims=3)))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](cat(x1, features_n4, dims=3)))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](cat(x2, features_n8, dims=3)))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[4](x3)
    up_x3 = u.conv_blocks[4](up_x3)
    up_x3 = u.conv_blocks[4](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](up_x3, x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](up_x1, x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](up_x2, op)
    # (n/2) X (n/2) X 32 X bs -> (n/2) X (n/2) X 16 X bs
    up_x5 = u.up_blocks[4](up_x4)
    # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

struct FeaturesUNet
  kappa_subnet
  solve_subnet
  indexes
end

@functor FeaturesUNet

function FeaturesUNet(in_chs, k_chs, s_model, k_model; kernel = (3, 3), indexes=3, σ=elu)
    kappa_subnet = k_model(indexes-2,k_chs;kernel=kernel,σ=σ)
    solve_subnet = s_model(in_chs-1,2;kernel=kernel,σ=σ)
    FeaturesUNet(kappa_subnet, solve_subnet, indexes)
end

function (u::FeaturesUNet)(x::AbstractArray)
    kappa = reshape(x[:,:,3:u.indexes,1], size(x)[1], size(x)[1], u.indexes-2, 1)
    features_n, features_n2, features_n4, features_n8 = u.kappa_subnet(kappa)
    u.solve_subnet(x, features_n, features_n2, features_n4, features_n8)
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

function create_model!(e_vcycle_input,kappa_input,gamma_input;kernel=(3,3),type=SUnet,k_type=NaN,k_chs=-1, indexes=3, σ=elu)
    input = 2
    if e_vcycle_input == true
        input = input+2
    end
    if kappa_input == true
        input = input+1
    end
    if gamma_input == true
        input = input+1
    end
    if k_chs < 1
        return type(input,2;kernel=kernel,σ=σ)
    else
        if k_type == FKappa
            return FeaturesUNet(input,k_chs,type,k_type;kernel=kernel,indexes=indexes,σ=σ)
        else
            @info "$(Dates.format(now(), "HH:MM:SS")) - SplitUNet $(input) $(k_chs)"
            return SplitUNet(input,k_chs,type,k_type;kernel=kernel,indexes=indexes,σ=σ)
        end
    end
end
