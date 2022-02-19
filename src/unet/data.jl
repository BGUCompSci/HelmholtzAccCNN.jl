using CSV, DataFrames

# x ← FGMRES(A=Helmholtz, M=V-Cycle, b, x = 0, maxIter = 1)
function generate_vcycle!(n, m, kappa, omega, gamma, b; v2_iter=10, level=3, restrt=1)
    h = r_type(2.0 ./ (n+m))

    sl_matrix_level3, h_matrix_level3 = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    kappa_coarse = down(reshape(kappa, n-1, m-1, 1, 1)|>pu)[:,:,1,1]
    gamma_coarse = down(reshape(gamma, n-1, m-1, 1, 1)|>pu)[:,:,1,1]
    sl_matrix_level2, h_matrix_level2 = get_helmholtz_matrices!(kappa_coarse, omega, gamma_coarse; alpha=r_type(0.5))
    kappa_coarse = down(reshape(kappa_coarse, Int64((n/2)-1),  Int64((m/2)-1), 1, 1)|>pu)[:,:,1,1]
    gamma_coarse = down(reshape(gamma_coarse,  Int64((n/2)-1),  Int64((m/2)-1), 1, 1)|>pu)[:,:,1,1]
    sl_matrix_level1, h_matrix_level1 = get_helmholtz_matrices!(kappa_coarse, omega, gamma_coarse; alpha=r_type(0.5))

    A(v::a_type) = vec(helmholtz_chain!(reshape(real(v), n-1, m-1, 1, 1), h_matrix_level3; h=h) + im*helmholtz_chain!(reshape(imag(v), n-1, m-1, 1, 1), h_matrix_level3; h=h))
    function M(v::a_type)
        v = reshape(v, n-1, m-1)
        x = zeros(c_type,n-1,m-1)|>pu
        x, = v_cycle_helmholtz!(n, m, h, x, v, h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3;  v2_iter=v2_iter, level=level)
        return vec(x)
    end

    x0 = zeros(c_type,n-1,m-1,1,1)|>pu
    if restrt == -1
        restrt = rand(1:10)
    end
    x_vcycle, = fgmres_func(A, vec(b), restrt, tol=1e-10, maxIter=1, M=M, x=vec(x0), out=-1, flexible=true)
    x_vcycle_channels = complex_grid_to_channels!(x_vcycle)
    return x_vcycle, x_vcycle_channels
end

# x ← FGMRES(A=Helmholtz, M=Jacobi, b, x = 0, maxIter = 1)
function generate_jacobi!(n, m, kappa, omega, gamma, b; v2_iter=10, level=3, restrt=1)
    h = r_type(2.0 ./ (n+m))
    sl_matrix, h_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))

    A(v::a_type) = vec(helmholtz_chain!(reshape(v, n-1, m-1, 1, 1), h_matrix; h=h))
    function M(v::a_type)
        v = reshape(v, n-1, m-1)
        x = zeros(c_type,n-1,m-1)|>pu
        x = jacobi_helmholtz_method!(n, m, h, x, v, sl_matrix)
        return vec(x)
    end

    x0 = zeros(c_type,n-1,m-1,1,1)|>pu
    if restrt == -1
        restrt = rand(1:10)
    end
    x_vcycle, = fgmres_func(A, vec(b), restrt, tol=1e-10, maxIter=1,
                                                    M=M, x=vec(x0), out=-1, flexible=true)
    x_vcycle_channels = complex_grid_to_channels!(x_vcycle)
    return x_vcycle, x_vcycle_channels
end

# r ← Ax - A(FGMRES(A=Helmholtz, M=V-Cycle, b, x = 0, maxIter = 1))
function generate_r_vcycle!(n, m, kappa, omega, gamma, x_true; v2_iter=10, level=3, restrt=1, jac=false)
    h = r_type(2.0 ./ (n+m))
    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    b_true = helmholtz_chain!(real(x_true), helmholtz_matrix; h=h) + im*helmholtz_chain!(imag(x_true), helmholtz_matrix; h=h)

    if jac == true
        x_vcycle, _ = generate_jacobi!(n, m, kappa, omega, gamma, b_true; v2_iter=v2_iter, level=level, restrt=restrt)
    else
        x_vcycle, _ = generate_vcycle!(n, m, kappa, omega, gamma, b_true; v2_iter=v2_iter, level=level, restrt=restrt)
    end
    x_vcycle = reshape(x_vcycle, n-1, m-1, 1, 1)
    e_true = x_true .- x_vcycle
    r_vcycle = b_true .- (helmholtz_chain!(real(x_vcycle), helmholtz_matrix; h=h) + im*helmholtz_chain!(imag(x_vcycle), helmholtz_matrix; h=h))

    return r_vcycle, e_true
end

function generate_random_data!(data_set_m, n, m, kappa, omega, gamma; e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=false,
                                                          kappa_type=1, threshold=50, kappa_input=true, kappa_smooth=false, k_kernel=3, axb=false, jac=false, norm_input=false, gmres_restrt=1)

    h = r_type(2.0 ./ (n+m))
    dataset = Tuple[]
    data_set_m = data_augmentetion == true ? floor(Int32,0.75*data_set_m) : data_set_m
    for i = 1:data_set_m
        # Generate Model
        kappa = generate_kappa!(n,m; type=kappa_type, smooth=kappa_smooth, threshold=threshold, kernel=k_kernel)|>pu

        # Generate Random Sample
        x_true = randn(c_type,n-1,m-1, 1, 1)|>pu

        if axb == true
            # Generate b
            _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
            r_vcycle = helmholtz_chain!(x_true, helmholtz_matrix; h=h)
            e_true = x_true
        else
            # Generate r,e
            r_vcycle, e_true = generate_r_vcycle!(n, m, kappa, omega, gamma, x_true;restrt=gmres_restrt, jac=jac)
        end

        r_vcycle = r_type(h^2) .* r_vcycle
        if norm_input == true
            norm_r = r_type(norm(r_vcycle))
            r_vcycle = r_vcycle ./ norm_r
            e_true = e_true ./ norm_r
        end

        # Print scale information
        if mod(i,5000) == 0
            if axb == true
                @info "$(Dates.format(now(), "HH:MM:SS")) - i = $(i) norm b = $(norm(r_vcycle)) norm x = $(norm(e_true))"
            else
                @info "$(Dates.format(now(), "HH:MM:SS")) - i = $(i) norm r = $(norm(r_vcycle)) norm e = $(norm(e_true))"
            end
        end

        r_vcycle_channels = complex_grid_to_channels!(r_vcycle)
        e_true_channels = complex_grid_to_channels!(e_true)

        # Generate e-vcycle
        if e_vcycle_input == true
            e_vcycle, e_vcycle_channels = generate_vcycle!(n, m, kappa, omega, gamma, r_vcycle; v2_iter=v2_iter, level=level)
            input = cat(e_vcycle_channels, r_vcycle_channels, dims=3)
        else
            input = r_vcycle_channels
        end

        input = kappa_input == true ? cat(input, reshape(kappa, n-1, m-1, 1, 1), dims=3) : input
        append!(dataset,[(input, e_true_channels)])

        # Data Augmentetion
        if data_augmentetion == true && mod(i,3) == 0
            (input_2,e_2) = dataset[rand(1:size(dataset,1))]
            r_index = e_vcycle_input == true ? 3 : 1
            r_2 = input_2[:,:,r_index:r_index+1,:]

            scalar = abs(rand(r_type))
            r_t = scalar*r_vcycle_channels+(1-scalar)*r_2
            scale = (scalar*norm(r_vcycle_channels) + (1-scalar)*norm(r_2))/norm(r_t)

            input_t = (scalar*input+(1-scalar)*input_2)*scale
            e_t = (scalar*e_true_channels+(1-scalar)*e_2)*scale
            append!(dataset,[(input_t, e_t)])
        end
    end
    return dataset
end

function generate_gmres_data!(data_set_m, n, m, kappa, omega, gamma; e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=false,
                                                          kappa_type=1, threshold=50, kappa_input=true, kappa_smooth=false, axb=false, norm_input=false)
    h = r_type(2.0 ./ (n+m))
    coefficient = norm_input == true ? h^2 : 1.0
    dataset = Tuple[]
    data_set_m = floor(Int64,data_set_m / 5)
    for i = 1:data_set_m
        # Generate Model
        kappa = generate_kappa!(n, m; type=kappa_type, smooth=kappa_smooth, threshold=threshold)

        # Generate Random Sample
        x_true = randn(c_type,n-1,m-1, 1, 1)|>pu
        s_matrix, h_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))

        if axb == true
            # Generate b
            b_true = helmholtz_chain!(x_true, h_matrix; h=h)
        else
            # Generate r,e
            b_true, x_true = generate_r_vcycle!(n, m, kappa, omega, gamma, x_true)
        end


        # append!(dataset,[(cat(complex_grid_to_channels!(coefficient .* b_true, n), reshape(kappa, n-1, n-1, 1, 1), dims=3),
        #                 complex_grid_to_channels!(x_true, n))])

        # if mod(i,500) == 0
        #     @info "$(Dates.format(now(), "HH:MM:SS")) - i = $(i) norm b = $(norm(b_true)) $(norm((h^2) .* b_true)) norm x = $(norm(x_true))"
        # end

        index = 1
        A(r) = vec(helmholtz_chain!(reshape(r, n-1, m-1, 1, 1), h_matrix; h=h))
        function M(r)
            r = reshape(r, n-1, m-1)
            if mod(index,4) == 0 # == 3 || index == 13 || index == 18
                e_true = fgmres_v_cycle_helmholtz!(n, m, h, r, kappa, omega, gamma)
                append!(dataset,[(cat(complex_grid_to_channels!(r), reshape(kappa, n-1, m-1, 1, 1), dims=3),
                                complex_grid_to_channels!(e_true ./ coefficient))])

                # Print scale information
                if mod(i,500) == 0
                    @info "$(Dates.format(now(), "HH:MM:SS")) - $(i) $(index) norm r = $(norm(r)) norm e = $(norm(e_true)) $(norm(e_true ./ (h^2)))"
                end
            end
            index = index + 1
            x = zeros(gmres_type,n-1,m-1)|>pu
            x, = v_cycle_helmholtz!(n, m, h, x, r, kappa, omega, gamma; u=1,
                        v1_iter = 1, v2_iter = 20, alpha=r_type(0.5), log = 0, level = 3)
            return vec(x)
        end
        x = zeros(gmres_type,n-1,m-1)|>pu
        x,flag,err,iter,resvec = gpu_flexible_gmres(A, vec(b_true), 10, tol=1e-20, maxIter=2,
                                                        M=M, x=vec(x), out=-1, flexible=true)

    end
    return dataset
end

function generate_point_source_data!(data_set_m, n, m, kappa, omega, gamma; v2_iter=10, level=3, save=false, path="results")
    h = r_type(2.0 ./ (n+m))
    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))

    dataset = Tuple[]
    for i = 1:data_set_m
        src = [rand(1:n-1),rand(1:m-1)]
        b = zeros(c_type,n-1,m-1, 1, 1)|>pu
        b[src[1],src[2],1,1] = 1.0 ./mean(h.^2)

        # V-cycle
        x = zeros(c_type,n-1,m-1,1,1)|>pu
        A(v) = vec(helmholtz_chain!(reshape(v, n-1, m-1, 1, 1), helmholtz_matrix; h=h))
        function M(v)
            v = reshape(v, n-1, m-1)
            x = zeros(c_type,n-1,m-1)|>pu
            x, = v_cycle_helmholtz!(n, m, h, x, v, kappa, omega, gamma; v2_iter = v2_iter, level = level)
            return vec(x)
        end
        x,flag,err,iter,residual = flexible_gmres(A, vec(b), 5, tol=1e-10, maxIter=1,
                                                        M=M, x=vec(x), out=-1, flexible=true)

        x = reshape(x,n-1,m-1,1,1)
        x_channels = complex_grid_to_channels!(x)
        b_channels = complex_grid_to_channels!(b)
        append!(dataset,[(x_channels, b_channels)])
    end
    return dataset
end

function get_csv_set!(path, data_set_m, n, m)
    h = r_type(2.0 ./ (n+m))
    df_training = CSV.File(path)|> DataFrame
    dataset = Tuple[]
    for i = 1:data_set_m
        input = cat(reshape(df_training.RR[(i-1)*(n-1)*(m-1)+1:i*(n-1)*(m-1)],n-1,m-1,1,1),
                    reshape(df_training.RI[(i-1)*(n-1)*(m-1)+1:i*(n-1)*(m-1)],n-1,m-1,1,1),
                    reshape(df_training.KAPPA[(i-1)*(n-1)*(m-1)+1:i*(n-1)*(m-1)],n-1,n-1,1,1), dims=3)

        output = cat(reshape(df_training.ER[(i-1)*(n-1)*(m-1)+1:i*(n-1)*(m-1)],n-1,n-1,1,1) ./ h^2,
                    reshape(df_training.EI[(i-1)*(n-1)*(m-1)+1:i*(n-1)*(m-1)],n-1,n-1,1,1) ./ h^2, dims=3)

        append!(dataset,[(input, output)])
    end
    return dataset
end
