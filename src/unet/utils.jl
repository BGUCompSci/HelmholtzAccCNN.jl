
function _random_normal(shape...)
  return u_type.(rand(Normal(0.f0,0.02f0),shape...))|> gpu
end

function extract_bboxes(mask)
  nth = last(size(mask))
  boxes = zeros(Integer, nth, 4)
  for i =1:nth
    m = mask[:,:,i]
    cluster = findall(!iszero, m)
    if length(cluster) > 0
      Is = map(x -> [x.I[1], x.I[2]], cluster) |> x -> hcat(x...)'
      x1, x2 = extrema(Is[:,1])
      y1, y2 = extrema(Is[:,2])
    else
      x1 ,x2, y1, y2 = 0, 0, 0, 0
    end
      boxes[i,:] = [y1, x1, y2, x2]
  end
  boxes
end

function extract_bboxes(masks::AbstractArray{T,4}) where T
  bs = []
  for i in 1:size(masks, 4)
    b = extract_bboxes(masks[:,:,:,i])
    push!(bs, b)
  end
  reduce(vcat, bs)
end

expand_dims(x,n::Int) = reshape(x,ones(Int32,n)...,size(x)...)
function squeeze(x)
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # For the case BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)...,1)
    end
end

function bce(ŷ, y; ϵ=gpu(fill(eps(first(ŷ)), size(ŷ)...)))
  l1 = -y.*log.(ŷ .+ ϵ)
  l2 = (1 .- y).*log.(1 .- ŷ .+ ϵ)
  l1 .- l2
end

function loss_bce(x, y)
  op = clamp.(model(x), 0.001f0, 1.f0)
  mean(bce(op, y))
end

function complex_grid_to_channels!(grid)
    grid_channels = CuArray{r_type}(undef, size(grid,1), size(grid,2), 2, 1)
    grid_channels[:, :, 1, :] = real(grid)
    grid_channels[:, :, 2, :] = imag(grid)
    return grid_channels
end

function complex_helmholtz_to_channels!(helmholtz_matrix)
    helmholtz_channels = CuArray{r_type}(undef, size(grid,1), size(grid,2), 2, 2)
    helmholtz_channels[:,:,1,1] = real(helmholtz_matrix)
    helmholtz_channels[:,:,2,1] = -imag(helmholtz_matrix)
    helmholtz_channels[:,:,1,2] = imag(helmholtz_matrix)
    helmholtz_channels[:,:,2,2] = real(helmholtz_matrix)
    return helmholtz_channels
end

function check_helmholtz_channels!(helmholtz_matrix, x, n)
    helmholtz_channels = complex_helmholtz_to_channels!(helmholtz_matrix, n)|> gpu
    x_channels = complex_grid_to_channels!(x, n)|> gpu

    original_result = helmholtz_chain!(x, helmholtz_matrix; h=h)|> gpu
    channels_result = helmholtz_chain_channels!(x_channels, helmholtz_channels, n; h=h)
    channels_result = channels_result[:,:,1,:]+im*channels_result[:,:,2,:]

    @info "$(Dates.format(now(), "HH:MM:SS")) - Check Helmholtz Channels $(norm(channels_result .- original_result)), $(original_result[1,1,1,1]), $(channels_result[1,1,1,1])"
    println(norm(channels_result - original_result))
end
