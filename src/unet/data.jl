using CSV, DataFrames

# x ← FGMRES(A=Helmholtz, M=V-Cycle, b, x = 0, maxIter = 1)
function generate_vcycle!(n, kappa, omega, gamma, b; v2_iter=10, level=3)
    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    h = 1.0 ./ n

    A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))
    function M(v)
        v = reshape(v, n-1, n-1)
        x = zeros(c_type,n-1,n-1)
        x, = v_cycle_helmholtz!(n, h, x, v, kappa, omega, gamma; v2_iter=v2_iter, level=level)
        return vec(x)
    end

    x0 = zeros(c_type,n-1,n-1,1,1)
    x_vcycle, = KrylovMethods.fgmres(A, vec(b), 1, tol=1e-10, maxIter=1,
                                                    M=M, x=vec(x0), out=-1, flexible=true)
    x_vcycle_channels = complex_grid_to_channels!(x_vcycle, n)
    return x_vcycle, x_vcycle_channels
end

# r ← Ax - A(FGMRES(A=Helmholtz, M=V-Cycle, b, x = 0, maxIter = 1))
function generate_r_vcycle!(n, kappa, omega, gamma, x_true; v2_iter=10, level=3)
    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    h = 1.0 ./ n
    b_true = helmholtz_chain!(x_true, helmholtz_matrix; h=h)

    x_vcycle, _ = generate_vcycle!(n, kappa, omega, gamma, b_true; v2_iter=v2_iter, level=level)
    x_vcycle = reshape(x_vcycle, n-1, n-1, 1, 1)
    e_true = x_true .- x_vcycle
    r_vcycle = b_true .- helmholtz_chain!(x_vcycle, helmholtz_matrix; h=h)

    return r_vcycle, e_true
end

function generate_random_data!(m, n, kappa, omega, gamma; e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=false,
                                                          kappa_type=1, threshold=50, kappa_input=true, kappa_smooth=false, axb=false, norm_input=false)
    h = 1.0./n;

    dataset = Tuple[]
    m = data_augmentetion == true ? floor(Int32,0.75*m) : m
    for i = 1:m
        # Generate Model
        kappa = generate_kappa!(n; type=kappa_type, smooth=kappa_smooth, threshold=threshold)

        # Generate Random Sample
        x_true = randn(c_type,n-1,n-1, 1, 1)

        if axb == true
            # Generate b
            _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
            r_vcycle = helmholtz_chain!(x_true, helmholtz_matrix; h=h)
            e_true = x_true
        else
            # Generate r,e
            r_vcycle, e_true = generate_r_vcycle!(n, kappa, omega, gamma, x_true)
        end

        if norm_input == true
            r_vcycle = (h^2) .* r_vcycle

            # TODO - Not work well
            # norm_r = norm(r_vcycle)
            # r_vcycle = r_vcycle ./ norm_r
            # e_true = e_true ./ norm_r
        end

        # Print scale information
        if mod(i,5000) == 0
            if axb == true
                @info "$(Dates.format(now(), "HH:MM:SS")) - i = $(i) norm b = $(norm(r_vcycle)) norm x = $(norm(e_true))"
            else
                @info "$(Dates.format(now(), "HH:MM:SS")) - i = $(i) norm r = $(norm(r_vcycle)) norm e = $(norm(e_true))"
            end
        end

        r_vcycle_channels = complex_grid_to_channels!(r_vcycle, n)
        e_true_channels = complex_grid_to_channels!(e_true, n)

        # Generate e-vcycle
        if e_vcycle_input == true
            e_vcycle, e_vcycle_channels = generate_vcycle!(n, kappa, omega, gamma, r_vcycle; v2_iter=v2_iter, level=level)
            input = cat(e_vcycle_channels, r_vcycle_channels, dims=3)
        else
            input = r_vcycle_channels
        end

        input = kappa_input == true ? cat(input, reshape(kappa, n-1, n-1, 1, 1), dims=3) : input
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

function generate_point_source_data!(m, n, kappa, omega, gamma; v2_iter=10, level=3, save=false, path="results")
    h = 1.0./n;
    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)

    dataset = Tuple[]
    for i = 1:m
        src = [rand(1:n-1),rand(1:n-1)]
        b = zeros(c_type,n-1,n-1, 1, 1)
        b[src[1],src[2],1,1] = 1.0 ./mean(h.^2)

        # V-cycle
        x = zeros(c_type,n-1,n-1,1,1)
        A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))
        function M(v)
            v = reshape(v, n-1, n-1)
            x = zeros(c_type,n-1,n-1)
            x, = v_cycle_helmholtz!(n, h, x, v, kappa, omega, gamma; v2_iter = v2_iter, level = level)
            return vec(x)
        end
        x,flag,err,iter,residual = KrylovMethods.fgmres(A, vec(b), 5, tol=1e-10, maxIter=1,
                                                        M=M, x=vec(x), out=-1, flexible=true)

        x = reshape(x,n-1,n-1,1,1)
        x_channels = complex_grid_to_channels!(x, n)
        b_channels = complex_grid_to_channels!(b, n)
        append!(dataset,[(x_channels, b_channels)])
    end
    return dataset
end
