using Statistics
using LinearAlgebra
using Flux
using LaTeXStrings
using KrylovMethods
using Plots
pyplot()

include("../../src/multigrid/helmholtz_methods.jl");

# Grid

n = 256
n_cells = [n,n];
h = 1.0./n;
src = [128,128]

b = zeros(ComplexF64,n-1,n-1)
b[src[1],src[2]] = 1.0 ./mean(h.^2);

# Parameters

f = 20.0
kappa = ones(Float64,tuple(n_cells .-1...))
omega = 2*pi*f;

# Gamma

gamma_val = 0.00001
gamma = gamma_val*2*pi * ones(Float64,size(kappa));
pad_cells = [20;20]
gamma = absorbing_layer!(gamma, pad_cells, omega);

# V cycle

x = x0 = zeros(ComplexF64,n-1,n-1)
x = fgmres_v_cycle_helmholtz!(n, h, b, kappa, omega, gamma; restrt=30, maxIter=1)

v1_iter = 1
v2_iter = 20
iterations = 1
use_gmres = 1
level = 3

if use_gmres == 1
    # GMRES V cycle

    iterations = 1
    restrt = 30
    max_iter = 5
    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))
    function M(v)
        v = reshape(v, n-1, n-1)
        x = zeros(ComplexF64,n-1,n-1)
        for i = 1:iterations
            x, = v_cycle_helmholtz!(n, h, x, v, kappa, omega, gamma; u=1,
                    v1_iter = v1_iter, v2_iter = v2_iter, alpha=0.5, log = 0, level = level)
        end
        return vec(x)
    end
    x,flag,err,iter,resvec = KrylovMethods.fgmres(A, vec(b), restrt, tol=1e-10, maxIter=max_iter,
                                                    M=M, x=vec(x), out=2, flexible=true)

    residual = resvec
    x = reshape(x,n-1,n-1)
else
    # V cycle
    v_cycle_iter = 100
    residual = zeros(v_cycle_iter)

    res = zeros(n-1,n-1)
    bnorm = norm(b)
    for i = 1:v_cycle_iter
        global x, helmholtz_matrix = v_cycle_helmholtz!(n, h, x, b, kappa, omega, gamma; u = 1,
                                        v1_iter = v1_iter, v2_iter = v2_iter, alpha=0.5, log = 0, level = level)
        global res = helmholtz_chain!(reshape(x, n-1, n-1, 1, 1), helmholtz_matrix; h=h)
        residual[i] = norm(b .- res[:,:,1,1]) / bnorm
        println(residual[i])
    end

end

q=M(vec(b))

heatmap(reshape(real(q),n-1,n-1), color=:grays)
print(residual)

iter = range(1, length=length(residual), iterations*length(residual))
p = plot(iter,residual,label="V cycle")
yaxis!(L"\Vert b - Hx \Vert_2", :log10)
xlabel!("Iterations")
savefig(replace("test/multigrid_helmholtz/results/f=$(f) gamma_val=$(gamma_val) residual graph","."=>"_"))

heatmap(reshape(real(x), n-1, n-1), color=:jet,size=(240,160))
savefig(replace("test/multigrid_helmholtz/results/without gamma n=$(n) f=$(f) gamma_val=$(gamma_val) result","."=>"_"))

show_last_residual = 0
if show_last_residual == 1
    last_res = helmholtz_chain!(reshape(x, n-1, n-1, 1, 1), helmholtz_matrix; h=h)
    heatmap(real(last_res[:,:,1,1] - b), color=:grays)
end
