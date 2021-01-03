using CSV, DataFrames

function generate_point_source_data!(m, n, kappa, omega, gamma; save=false, path="results")
    h = 1.0./n;
    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)

    dataset = Tuple[]
    for i = 1:m
        src = [rand(1:n-1),rand(1:n-1)]
        b = zeros(ComplexF64,n-1,n-1, 1, 1)
        b[src[1],src[2],1,1] = 1.0 ./mean(h.^2)

        # V-cycle
        x = zeros(ComplexF64,n-1,n-1,1,1)
        A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))
        function M(v)
            v = reshape(v, n-1, n-1)
            x = zeros(ComplexF64,n-1,n-1)
            x, = v_cycle_helmholtz!(n, h, x, v, kappa, omega, gamma; u=1,
                    v1_iter = 1, v2_iter = 20, alpha=0.5, log = 0, level = 3)
            return vec(x)
        end
        x,flag,err,iter,residual = KrylovMethods.fgmres(A, vec(b), 5, tol=1e-10, maxIter=1,
                                                        M=M, x=vec(x), out=2, flexible=true)

        x = reshape(x,n-1,n-1,1,1)
        x_channels = complex_grid_to_channels!(x, n)
        b_channels = complex_grid_to_channels!(b, n)
        append!(dataset,[(x_channels, b_channels)])
        if save == true
            CSV.write("$(path)/real_x_$(i).csv", DataFrame(real(x[:,:,1,1])), writeheader=false)
            CSV.write("$(path)/imag_x_$(i).csv", DataFrame(imag(x[:,:,1,1])), writeheader=false)
            CSV.write("$(path)/real_b_$(i).csv", DataFrame(real(b[:,:,1,1])), writeheader=false)
            CSV.write("$(path)/imag_b_$(i).csv", DataFrame(imag(b[:,:,1,1])), writeheader=false)
        end
    end
    return dataset
end

function read_data!(path, m, n)
    dataset = Tuple[]
    for i = 1:m
        x_real = Matrix(CSV.File("$(path)/real_x_$(i).csv", header=false)|> DataFrame)
        x_imag = Matrix(CSV.File("$(path)/imag_x_$(i).csv", header=false)|> DataFrame)
        x_channels = complex_grid_to_channels!(reshape(x_real+im*x_imag,n-1,n-1,1,1), n)
        b_real = Matrix(CSV.File("$(path)/real_b_$(i).csv", header=false)|> DataFrame)
        b_imag = Matrix(CSV.File("$(path)/imag_b_$(i).csv", header=false)|> DataFrame)
        b_channels = complex_grid_to_channels!(reshape(b_real+im*b_imag,n-1,n-1,1,1), n)
        append!(dataset,[(x_channels, b_channels)])
    end
    return dataset
end

function generate_random_data!(m, n, kappa, omega, gamma; generate_e_vcycle=false)
    h = 1.0./n;
    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)

    dataset = Tuple[]
    for i = 1:m
        x_true = randn(ComplexF64,n-1,n-1, 1, 1)
        b_true = helmholtz_chain!(x_true, helmholtz_matrix; h=h)

        # V-cycle
        A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))
        function M(v)
            v = reshape(v, n-1, n-1)
            x = zeros(ComplexF64,n-1,n-1)
            x, = v_cycle_helmholtz!(n, h, x, v, kappa, omega, gamma; u=1,
                    v1_iter = 1, v2_iter = 3, alpha=0.5, log = 0, level = 3)
            return vec(x)
        end

        x0 = zeros(ComplexF64,n-1,n-1,1,1)
        x_vcycle,flag,err,iter,residual = KrylovMethods.fgmres(A, vec(b_true), 1, tol=1e-10, maxIter=1,
                                                        M=M, x=vec(x0), out=2, flexible=true)
        x_vcycle = reshape(x_vcycle, n-1, n-1, 1, 1)
        e_true = x_true - x_vcycle
        r_vcycle = b_true - helmholtz_chain!(x_vcycle, helmholtz_matrix; h=h)

        r_vcycle_channels = complex_grid_to_channels!(r_vcycle, n)
        e_true_channels = complex_grid_to_channels!(e_true, n)

        if generate_e_vcycle == true
            e0 = zeros(ComplexF64,n-1,n-1,1,1)
            e_vcycle,flag,err,iter,residual = KrylovMethods.fgmres(A, vec(r_vcycle), 1, tol=1e-10, maxIter=1,
                                                            M=M, x=vec(e0), out=2, flexible=true)

            e_vcycle_channels = complex_grid_to_channels!(e_vcycle, n)
            append!(dataset,[(cat(e_vcycle_channels, r_vcycle_channels, dims=3),e_true_channels)])
        else
            append!(dataset,[(r_vcycle_channels, e_true_channels)])
        end
    end
    return dataset
end
