using TestImages
using ImageView
using ImageTransformations: imresize
using Flux
using LinearAlgebra
using Plots
pyplot()

include("../../src/multigrid/laplacian_methods.jl");

img = Gray.(testimage("mandrill"));
img = imresize(img, ratio=1/2);
img_mat = convert(Array{Float64}, img);
n = size(img_mat,1)
h = 1.0/n
img_mat = reshape(img_mat[1:end-1,1:end-1], n-1, n-1, 1, 1);
ImageView.imshow(img_mat[:,:,1,1])

result_laplacian = laplacian_conv!(img_mat)
ImageView.imshow(result_laplacian[:,:,1,1])

result_down = down(img_mat)
ImageView.imshow(result_down[:,:,1,1])

result_up = up(result_down)
ImageView.imshow(result_up[:,:,1,1])

result_laplacian = laplacian_conv!(img_mat;h=h)
b = reshape(result_laplacian, length(result_laplacian), 1)
init = randn(tuple(((n-1)^2)...))

iterations = 30
jacobi_iterations = 600
residual = zeros(iterations,2);

# V-cycle

v1_iter = 1
v2_iter = floor(Int,jacobi_iterations/(iterations))-2*log2(n)

x = init
x_matrix = reshape(x, n-1, n-1, 1, 1)

for i = 1:iterations

    global x = v_cycle_laplacian!(n, h, x, b; u=1, v1_iter=v1_iter, v2_iter=v2_iter,log=0)
    global x_matrix = reshape(x, n-1, n-1, 1, 1)
    result_laplacian = laplacian_conv!(x_matrix;h=h)
    residual[i,1] = norm(reshape(result_laplacian, length(result_laplacian), 1) - b)

end

heatmap(x_matrix[:,:,1,1], color=:greys, yflip=true)
savefig("test/multigrid_laplacian/results/V-Cycle Result")

# Jacobi

x = init
x_matrix = reshape(x, n-1, n-1, 1, 1)

result_laplacian = laplacian_conv!(img_mat)
b = reshape(result_laplacian, length(result_laplacian), 1)

for i = 1:iterations

    global x = jacobi_laplacian_method!(n, 1.0, x, b; max_iter=floor(Int,jacobi_iterations/(iterations)))
    global x_matrix = reshape(x, n-1, n-1, 1, 1)
    result_laplacian = laplacian_conv!(x_matrix)
    residual[i,2] = norm(reshape(result_laplacian, length(result_laplacian), 1) - b)

end

heatmap(x_matrix[:,:,1,1], color=:greys, yflip=true)
savefig("test/multigrid_laplacian/results/Jacobi Result")

iter = range(1, length=iterations, jacobi_iterations)
p = plot(iter,residual[:,1],label="V cycle")
plot!(iter,residual[:,2],label="Jacobi")
yaxis!("|| Error ||", :log10)
xlabel!("Iterations")
title!("Comperation between V Cycle and Jacobi")
savefig("test/multigrid_laplacian/results/Residual Graph")
