include("../../src/unet/utils.jl")

function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2)|>cgpu,
	  BatchNorm(out_ch)|> cgpu,
	  x->squeeze(x))|> cgpu
end

ConvDown(in_chs,out_chs;kernel = (5,5), σ=elu) =
    Chain(Conv(kernel, in_chs=>out_chs, stride=(2,2), pad = 1; init=_random_normal),
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
                    ConvTranspose(kernel, in_chs=>out_chs, stride=(2, 2), pad = 1; init=_random_normal),
                    BatchNorm(out_chs)))|> cgpu

UNetConvBlock(in_chs, out_chs; kernel = (3, 3), pad=1, σ=elu) =
    Chain(Conv(kernel, in_chs=>out_chs, pad=pad; init=_random_normal),
                BatchNorm(out_chs),
                x-> (σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu

# ResNet Types:

# Preservation of input with with a 1-kernel convolution

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

# Preservation of input without convolution

struct SResidualBlock
  layers
  activation
end

(r::SResidualBlock)(input) = r.activation(r.layers(input) + input)

@functor SResidualBlock

function SResidualBlock(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, in_chs=>in_chs, pad = pad; init=_random_normal),
                	BatchNorm(in_chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, in_chs=>out_chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(out_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    SResidualBlock(layers, activation)|> cgpu
end

# Doubling the channels and reducing back between the 2 convolution actions
# C1: chs-> 2*chs, C2: 2*chs->chs

struct TSResidualBlock
  layers
  activation
end

(r::TSResidualBlock)(input) = r.activation(r.layers(input) + input)

@functor TSResidualBlock

function TSResidualBlock(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, in_chs=>(2*in_chs), pad = pad; init=_random_normal),
                	BatchNorm(2*in_chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, (2*in_chs)=>out_chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(out_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    TSResidualBlock(layers, activation)|> cgpu
end

# Convolutions with known kernels

struct WResidualBlock
  layers
  activation
end

@functor WResidualBlock

function WResidualBlock(in_chs::Int, out_chs::Int; σ=elu)
    layers(kernel) = Conv(kernel, u_type.(reshape([0.0],1,1,1,1))|> cgpu)
    activation = Chain(BatchNorm(in_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    WResidualBlock(layers, activation)|> cgpu
end

function (r::WResidualBlock)(input, kernel1, kernel2)
    c1 = Conv(u_type.(kernel1)|> cgpu, u_type.([0.0])|> cgpu, pad=1)(input)
    ac1 = r.activation(c1)
    c2 = Conv(u_type.(kernel2)|> cgpu, u_type.([0.0])|> cgpu, pad=1)(ac1)
    y = r.activation(c2 + input)
    return y
end

# 3 convolution operations in one ResNet layer

struct SResidualBlock3C
  layers
  activation
end

(r::SResidualBlock3C)(input) = r.activation(r.layers(input) + input)

@functor SResidualBlock3C

function SResidualBlock3C(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, in_chs=>in_chs, pad = pad; init=_random_normal),
                	BatchNorm(in_chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, in_chs=>out_chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(Conv(kernel, out_chs=>out_chs, pad = pad; init=_random_normal),
                    BatchNorm(in_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    SResidualBlock3C(layers, activation)|> cgpu
end

# Input as is without activation and batch normalization

struct SResidualBlockI
  layers
  activation
end

(r::SResidualBlockI)(input) = r.layers(r.activation(input)) + input

@functor SResidualBlockI

function SResidualBlockI(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, in_chs=>in_chs, pad = pad; init=_random_normal),
                	BatchNorm(in_chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, in_chs=>in_chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(in_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    SResidualBlockI(layers, activation)|> cgpu
end

# Input as is without activation and batch normalization
# and doubling the channels and reducing back between the 2 convolution actions

struct TSResidualBlockI
  layers
  activation
end

(r::TSResidualBlockI)(input) = r.layers(r.activation(input)) + input

@functor TSResidualBlockI

function TSResidualBlockI(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    layers = Chain(Conv(kernel, in_chs=>(2*in_chs), pad = pad; init=_random_normal),
                	BatchNorm(2*in_chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, (2*in_chs)=>out_chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(out_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    TSResidualBlockI(layers, activation)|> cgpu
end

# Multiply ResNet steps to create waves

struct ResidualBlock2
  c1
  c2
  activation
end

@functor ResidualBlock2

function ResidualBlock2(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    c1 = Chain(Conv(kernel, in_chs=>in_chs, pad = pad; init=_random_normal),
                	BatchNorm(in_chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, in_chs=>in_chs, pad = pad; init=_random_normal))|> cgpu
    c2 = Chain(Conv(kernel, in_chs=>in_chs, pad = pad; init=_random_normal),
                    BatchNorm(in_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, in_chs=>in_chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(in_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    ResidualBlock2(c1, c2, activation)|> cgpu
end

function (r::ResidualBlock2)(input)
    f_t = input + r.c1(r.activation(input))
    f_t1 = 2 * f_t - input + r.activation(r.c2(f_t))
    return f_t1
end

struct TResidualBlock2
  c1
  c2
  activation
end

@functor TResidualBlock2

function TResidualBlock2(in_chs::Int, out_chs::Int; kernel = (3, 3), pad =1, σ=elu)
    c1 = Chain(Conv(kernel, in_chs=> (2*in_chs), pad = pad; init=_random_normal),
                	BatchNorm(2*in_chs),
                	x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, (2*in_chs)=>in_chs, pad = pad; init=_random_normal))|> cgpu
    c2 = Chain(Conv(kernel, in_chs=>(2*in_chs), pad = pad; init=_random_normal),
                    BatchNorm(2*in_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                    Conv(kernel, (2*in_chs)=>in_chs, pad = pad; init=_random_normal))|> cgpu
    activation = Chain(BatchNorm(in_chs),
                    x->(σ == elu ? σ.(x,0.2f0) : σ.(x)))|> cgpu
    TResidualBlock2(c1, c2, activation)|> cgpu
end

function (r::TResidualBlock2)(input)
    f_t = input + r.c1(r.activation(input))
    f_t1 = 2 * f_t - input + r.activation(r.c2(f_t))
    return f_t1
end

# U-Net Types:

# SUnet - not deep, wide, without stride at the entrance
# n X n X 4 X bs -> (n/4) X (n/4) X 128 X bs

struct SUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SUnet

function SUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=ResidualBlock)
    conv_down_blocks = Chain(ConvDown(16,32;σ=σ),
		      ConvDown(32,64;σ=σ),
		      ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(channels, 8; kernel = kernel, σ=σ),
		 resnet_type(8, 32; kernel = kernel, σ=σ),
		 resnet_type(64, 64; kernel = kernel, σ=σ),
		 resnet_type(128, 128; kernel = kernel, σ=σ))|> cgpu

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

# SSUnet - not deep, wide, with stride at the entrance
# n X n X 4 X bs -> (n/4) X (n/4) X 128 X bs

struct SSUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SSUnet

function SSUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

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

# DTUnet - deep, wide, without stride at the entrance
# n X n X 4 X bs -> (n/8) X (n/8) X 256 X bs

struct DTUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor DTUnet

function DTUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=ResidualBlock)
    conv_down_blocks = Chain(ConvDown(16,32;σ=σ),
		      ConvDown(32,64;σ=σ),
		      ConvDown(64,128;σ=σ),
		      ConvDown(128,256;σ=σ))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel, σ=σ),
		 ResidualBlock(8, 32; kernel = kernel, σ=σ),
		 resnet_type(64, 64; kernel = kernel, σ=σ),
		 resnet_type(128, 128; kernel = kernel, σ=σ),
		 resnet_type(256, 256; kernel = kernel, σ=σ))|> cgpu

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

# DNUnet - deep, narrow, without stride at the entrance
# n X n X 4 X bs -> (n/8) X (n/8) X 128 X bs

struct DNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor DNUnet

function DNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=ResidualBlock)
    conv_down_blocks = Chain(ConvDown(16,8;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel, σ=σ),
         ResidualBlock(8, 16; kernel = kernel, σ=σ),
         resnet_type(32, 32; kernel = kernel, σ=σ),
         resnet_type(64, 64; kernel = kernel, σ=σ),
         resnet_type(128, 128; kernel = kernel, σ=σ))|> cgpu

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

# SDNUnet - deep, narrow, with stride at the entrance
# n X n X 4 X bs -> (n/8) X (n/8) X 128 X bs

struct SDNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SDNUnet

function SDNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
         resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

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
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](x2))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[5](x3)
    up_x3 = u.conv_blocks[6](up_x3)
    up_x3 = u.conv_blocks[7](up_x3)

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

# Old networks and their ResNet type:
# SDNUnet1 use SResidualBlock
# SDNUnet1D use SResidualBlock
# SDNUnetI use SResidualBlockI
# SDNUnet2 use ResidualBlock2
# SSDNUnet2 use SResidualBlock2
# SDNUnet21 use SResidualBlock3C

# SinUnetI - deep, narrow, with stride at the entrance, with sin operators
# n X n X 4 X bs -> (n/8) X (n/8) X 128 X bs

struct SinUnetI
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SinUnetI

function SinUnetI(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
         resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    SinUnetI(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::SinUnetI)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = sin.(u.conv_blocks[2](u.conv_down_blocks[2](op)))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = sin.(u.conv_blocks[4](u.conv_down_blocks[4](x2)))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = sin.(u.conv_blocks[5](x3))
    up_x3 = u.conv_blocks[6](up_x3)
    up_x3 = u.conv_blocks[7](up_x3)

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

# SDNNUnet - deep, narrow, with stride at the entrance, double ResNet steps
# n X n X 4 X bs -> (n/8) X (n/8) X 128 X bs

struct SDNNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor SDNNUnet

function SDNNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,12;σ=σ),
              ConvDown(12,24;σ=σ),
              ConvDown(24,48;σ=σ),
              ConvDown(48,96;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(12,12; kernel = kernel, σ=σ),
         resnet_type(24,24; kernel = kernel, σ=σ),
         resnet_type(48,48; kernel = kernel, σ=σ),
         resnet_type(96,96; kernel = kernel, σ=σ),
         resnet_type(96,96; kernel = kernel, σ=σ),
         resnet_type(96,96; kernel = kernel, σ=σ),
         resnet_type(96,96; kernel = kernel, σ=σ))|> cgpu

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
    up_x3 = u.conv_blocks[5](x3)
    up_x3 = u.conv_blocks[6](up_x3)
    up_x3 = u.conv_blocks[7](up_x3)

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

# Architecture Types:

# SplitUNet: the encoder prepares 'k_chs' channels for the entry of the solver network

struct SplitUNet
  kappa_subnet
  solve_subnet
  indexes
end

@functor SplitUNet

function SplitUNet(in_chs, k_chs, s_model, k_model; kernel = (3, 3), indexes=3, σ=elu, resnet_type=SResidualBlock)
    kappa_subnet = k_model(indexes-2,k_chs;kernel=kernel,σ=σ,resnet_type=resnet_type)
    solve_subnet = s_model(k_chs+in_chs,2;kernel=kernel,σ=σ,resnet_type=resnet_type)
    SplitUNet(kappa_subnet, solve_subnet, indexes)
end

function (u::SplitUNet)(x::AbstractArray)
    kappa = reshape(x[:,:,3:u.indexes,1], size(x)[1], size(x)[1], u.indexes-2, 1)
    kappa_features = u.kappa_subnet(kappa)
    u.solve_subnet(cat(x, kappa_features, dims=3))
end

# FeaturesUNet: the encoder prepares hierarchical input for the solver network

struct FeaturesUNet
  kappa_subnet
  solve_subnet
  indexes
end

@functor FeaturesUNet

function FeaturesUNet(in_chs, k_chs, s_model, k_model; kernel = (3, 3), indexes=3, σ=elu, resnet_type=SResidualBlock)
    kappa_subnet = k_model(indexes-2,k_chs;kernel=kernel,σ=σ,resnet_type=resnet_type)
    solve_subnet = s_model(in_chs,2;kernel=kernel,σ=σ,resnet_type=resnet_type)
    FeaturesUNet(kappa_subnet, solve_subnet, indexes)
end

function (u::FeaturesUNet)(x::AbstractArray)
    kappa = reshape(x[:,:,3:u.indexes,1], size(x)[1], size(x)[1], u.indexes-2, 1)
    features = u.kappa_subnet(kappa)
    u.solve_subnet(x, features)
end

# FKappa + FSDNUnet: an encoder that prepares hierarchical input for the restriction step

struct FKappa
  conv_down_blocks
  conv_blocks
end

@functor FKappa

function FKappa(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=ResidualBlock)
    conv_down_blocks = Chain(ConvDown(16,32;σ=σ),
		      ConvDown(32,64;σ=σ),
		      ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(channels, 8; kernel = kernel, σ=σ),
		 resnet_type(8, 16; kernel = kernel, σ=σ),
		 resnet_type(32, 32; kernel = kernel, σ=σ),
		 resnet_type(64, 64; kernel = kernel, σ=σ),
		 resnet_type(128, 128; kernel = kernel, σ=σ))|> cgpu

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

    return [op, x1, x2, x3]
end

struct FSDNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor FSDNUnet

function FSDNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlockI)
    conv_down_blocks = Chain(ConvDown(20,16;σ=σ),
              ConvDown(48,32;σ=σ),
              ConvDown(96,64;σ=σ),
              ConvDown(192,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
         resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    FSDNUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::FSDNUnet)(x::AbstractArray, features)

    # n X n X 3 X bs + n X n X 16 X bs-> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](cat(x, features[1], dims=3)))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](cat(op, features[2], dims=3)))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](cat(x1, features[3], dims=3)))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](cat(x2, features[4], dims=3)))

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

# FFKappa + FFSDNUnet: an encoder that prepares hierarchical input

struct FFKappa
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor FFKappa

function FFKappa(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
         resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    FFKappa(conv_down_blocks, conv_blocks, up_blocks)
end


function (u::FFKappa)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](x2))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[5](x3)
    up_x3 = u.conv_blocks[6](up_x3)
    up_x3 = u.conv_blocks[7](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](up_x3, x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](up_x1, x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](up_x2, op)

    return [op, x1, x2, up_x3, up_x1, up_x2, up_x4]
end

struct FFSDNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor FFSDNUnet

function FFSDNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(32,32;σ=σ),
              ConvDown(64,64;σ=σ),
              ConvDown(128,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
         resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(256, 64; σ=σ),
                        UNetUpBlock(256, 32; σ=σ),
                        UNetUpBlock(128, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 64=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    FFSDNUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::FFSDNUnet)(x::AbstractArray, features)

    # n X n X 4 X bs + n X n X 16 X bs-> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](cat(op, features[1], dims=3)))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](cat(x1, features[2], dims=3)))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](cat(x2, features[3], dims=3)))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[5](x3)
    up_x3 = u.conv_blocks[6](up_x3)
    up_x3 = u.conv_blocks[7](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](cat(up_x3, features[4], dims=3), x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](cat(up_x1, features[5], dims=3), x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](cat(up_x2, features[6], dims=3), op)
    # (n/2) X (n/2) X 32 X bs -> (n/2) X (n/2) X 16 X bs
    up_x5 = u.up_blocks[4](cat(up_x4, features[7], dims=3))
    # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

# TFFKappa: an encoder that prepares hierarchical inputand uses double ResNet steps

struct TFFKappa
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor TFFKappa

function TFFKappa(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
         resnet_type(16,16; kernel = kernel, σ=σ),
         resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(16,16; kernel = kernel, σ=σ),
         resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    TFFKappa(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::TFFKappa)(x::AbstractArray)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[2](u.conv_blocks[1](u.conv_down_blocks[1](x)))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[4](u.conv_blocks[3](u.conv_down_blocks[2](op)))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[6](u.conv_blocks[5](u.conv_down_blocks[3](x1)))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[8](u.conv_blocks[7](u.conv_down_blocks[4](x2)))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[9](x3)
    up_x3 = u.conv_blocks[10](up_x3)
    up_x3 = u.conv_blocks[11](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.conv_blocks[15](u.up_blocks[1](up_x3, x2))
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.conv_blocks[14](u.up_blocks[2](up_x1, x1))
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.conv_blocks[13](u.up_blocks[3](up_x2, op))

    return [op, x1, x2, up_x3, up_x1, up_x2, up_x4]
end

# MFFSDNUnet: solver network which multiplies the hierarchical vectors with the input of each layer

struct MFFSDNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor MFFSDNUnet

function MFFSDNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu, resnet_type=SResidualBlock)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(resnet_type(16,16; kernel = kernel, σ=σ),
         resnet_type(32,32; kernel = kernel, σ=σ),
         resnet_type(64,64; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ),
         resnet_type(128,128; kernel = kernel, σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    MFFSDNUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::MFFSDNUnet)(x::AbstractArray, features)

    # n X n X 4 X bs + n X n X 16 X bs-> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x))
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op .* features[1]))
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1 .* features[2]))
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](x2 .* features[3]))

    # (n/16) X (n/16) X 128 X bs
    up_x3 = u.conv_blocks[5](x3)
    up_x3 = u.conv_blocks[6](up_x3)
    up_x3 = u.conv_blocks[7](up_x3)

    # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
    up_x1 = u.up_blocks[1](up_x3 .* features[4], x2)
    # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
    up_x2 = u.up_blocks[2](up_x1 .* features[5], x1)
    # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
    up_x4 = u.up_blocks[3](up_x2 .* features[6], op)
    # (n/2) X (n/2) X 32 X bs -> (n/2) X (n/2) X 16 X bs
    up_x5 = u.up_blocks[4](up_x4 .* features[7])
    # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
    u.up_blocks[end](up_x5)

end

# Old encoder-solver and their ResNet type:
# FFSDNUnet,FFKappa = SResidualBlockI
# FFSDNUnet1,FFKappa1 = SResidualBlock

# WKappa + WSDNUnet: an encoder that learns weights for the ResNet steps of the solver network

struct WKappa
  conv_down_blocks
  conv_blocks
end

@functor WKappa

function WKappa(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(16,32;σ=σ),
		      ConvDown(32,64;σ=σ),
		      ConvDown(64,128;σ=σ),
		      ConvDown(128,256;σ=σ),
		      ConvDown(256,480;σ=σ),
		      ConvDown(480,480;σ=σ))|> cgpu

    conv_blocks = Chain(ResidualBlock(channels, 8; kernel = kernel, σ=σ),
		 ResidualBlock(8, 16; kernel = kernel, σ=σ),
		 ResidualBlock(32, 32; kernel = kernel, σ=σ),
		 ResidualBlock(64, 64; kernel = kernel, σ=σ),
		 ResidualBlock(128, 128; kernel = kernel, σ=σ),
		 ResidualBlock(256, 256; kernel = kernel, σ=σ),
		 ResidualBlock(480, 480; kernel = kernel, σ=σ))|> cgpu

    WKappa(conv_down_blocks, conv_blocks)
end

function (u::WKappa)(x::AbstractArray)

    # n X n X 4 X bs -> n X n X 16 X bs
    op = u.conv_blocks[1:2](x)
    # n X n X 16 X bs -> (n/2) X (n/2) X 32 X bs
    x1 = u.conv_blocks[3](u.conv_blocks[3](u.conv_down_blocks[1](op)))
    # (n/2) X (n/2) X 32 X bs -> (n/4) X (n/4) X 64 X bs
    x2 = u.conv_blocks[4](u.conv_blocks[4](u.conv_down_blocks[2](x1)))
    # (n/4) X (n/4) X 64 X bs -> (n/8) X (n/8) X 128 X bs
    x3 = u.conv_blocks[5](u.conv_blocks[5](u.conv_down_blocks[3](x2)))
    #  (n/8) X (n/8) X 128 X bs -> (n/16) X (n/16) X 256 X bs
    x4 = u.conv_blocks[6](u.conv_blocks[6](u.conv_down_blocks[4](x3)))
    # (n/16) X (n/16) X 256 X bs -> (n/32) X (n/32) X 480 X bs
    x5 = u.conv_blocks[7](u.conv_blocks[7](u.conv_down_blocks[5](x4)))

    return [big_block_filter!(3, x5[1:3,1:3,1:16,:], 16), big_block_filter!(3, x5[1:3,1:3,17:32,:], 16),
            big_block_filter!(3, x5[1:3,1:3,33:64,:], 32), big_block_filter!(3, x5[1:3,1:3,65:96,:], 32),
            big_block_filter!(3, x5[1:3,1:3,97:160,:], 64), big_block_filter!(3, x5[1:3,1:3,161:224,:], 64),
            big_block_filter!(3, x5[1:3,1:3,225:352,:], 128), big_block_filter!(3, x5[1:3,1:3,353:480,:], 128)]
end

struct WSDNUnet
  conv_down_blocks
  conv_blocks
  up_blocks
end

@functor WSDNUnet

function WSDNUnet(channels::Int = 2, labels::Int = channels; kernel = (3, 3), σ=elu)
    conv_down_blocks = Chain(ConvDown(channels,16;σ=σ),
              ConvDown(16,32;σ=σ),
              ConvDown(32,64;σ=σ),
              ConvDown(64,128;σ=σ))|> cgpu

    conv_blocks = Chain(WResidualBlock(16; σ=σ),
         WResidualBlock(32; σ=σ),
         WResidualBlock(64; σ=σ),
         WResidualBlock(128; σ=σ))|> cgpu

    up_blocks = Chain(UNetUpBlock(128, 64; σ=σ),
                        UNetUpBlock(128, 32; σ=σ),
                        UNetUpBlock(64, 16; σ=σ),
                        Chain(x->(σ == elu ? σ.(x,0.2f0) : σ.(x)),
                                Conv((3, 3), pad = 1, 32=>16;init=_random_normal)),
                        ConvUp(16, labels; σ=σ))|> cgpu
    WSDNUnet(conv_down_blocks, conv_blocks, up_blocks)
end

function (u::WSDNUnet)(x::AbstractArray, kernels)

    # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
    op = u.conv_blocks[1](u.conv_down_blocks[1](x), kernels[1], kernels[2])
    # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
    x1 = u.conv_blocks[2](u.conv_down_blocks[2](op), kernels[3], kernels[4])
    # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
    x2 = u.conv_blocks[3](u.conv_down_blocks[3](x1), kernels[5], kernels[6])
    # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
    x3 = u.conv_blocks[4](u.conv_down_blocks[4](x2), kernels[7], kernels[8])

    # (n/16) X (n/16) X 128 X bs
    up_x3 = x3 # u.conv_blocks[4](x3)

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

function create_model!(e_vcycle_input,kappa_input,gamma_input;kernel=(3,3),type=SUnet,k_type=NaN,resnet_type=SResidualBlock,k_chs=-1, indexes=3, σ=elu, arch=0)
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

    if arch == 0 # A stand-alone U-Net
        return type(input,2;kernel=kernel,σ=σ,resnet_type=resnet_type)
    else
        if arch == 2 # Encoder with a hierarchical context
            @info "$(Dates.format(now(), "HH:MM:SS")) - FeaturesUNet"
            return FeaturesUNet(input,k_chs,type,k_type;kernel=kernel,indexes=indexes,σ=σ,resnet_type=resnet_type)
        else # Encoder with a simple context
            @info "$(Dates.format(now(), "HH:MM:SS")) - SplitUNet $(input) $(k_chs)"
            return SplitUNet(input,k_chs,type,k_type;kernel=kernel,indexes=indexes,σ=σ,resnet_type=resnet_type)
        end
    end
end
