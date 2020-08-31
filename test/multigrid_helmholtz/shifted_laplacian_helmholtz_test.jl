using Statistics
using LinearAlgebra
using Flux
using LaTeXStrings
using Plots
pyplot()

include("../../src/multigrid/helmholtz_methods.jl");

# Grid

n=1024
n_cells = [n,n];
h = 1.0./n;
src = [512,512]

b = zeros(ComplexF64,tuple((n_cells.-1)...))
b[src[1],src[2]] = 1.0 ./mean(h.^2);

# Parameters

gamma_val = 0.00001*2*pi;
f = 40.0;
kappa = ones(Float64,tuple(n_cells .-1...))
omega = 2*pi*f;
gamma = gamma_val * ones(ComplexF64,size(kappa));

pad_cells = [20;20]
gamma = absorbing_layer!(gamma, pad_cells, omega);

# V cycle

v_cycle_iter = 120
v1_iter = 1
v2_iter = 10

residual = zeros(v_cycle_iter)
x = zeros(n-1,n-1)
res = zeros(n-1,n-1)

for i = 1:v_cycle_iter
    global x, helmholtz_matrix = v_cycle_helmholtz!(n, h, x, b, kappa, omega, gamma;
                    u = 1, v1_iter = v1_iter, v2_iter = v2_iter, alpha=0.5, log = 0, use_gmres = 0)
    global res = helmholtz_chain!(reshape(x, n-1, n-1, 1, 1), helmholtz_matrix; h=h)
    residual[i] = norm(b .- res[:,:,1,1])
end

print(residual)

iter = range(1, length=v_cycle_iter)
p = plot(iter,residual,label="V cycle")
yaxis!(L"\Vert b - Hx \Vert_2^2", :log10)
xlabel!("Iterations")
savefig("test/multigrid_helmholtz/Residual Graph")

heatmap(real(x), color=:grays)
savefig("test/multigrid_helmholtz/V-Cycle Shifted Laplacian Helmholtz")
