using CSV, DataFrames

function generate_vcycle!(n, kappa, omega, gamma, b; v2_iter=10, level=3)
    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    h = 1.0 ./ n

    A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))
    function M(v)
        v = reshape(v, n-1, n-1)
        x = zeros(ComplexF64,n-1,n-1)
        x, = v_cycle_helmholtz!(n, h, x, v, kappa, omega, gamma; v2_iter=v2_iter, level=level)
        return vec(x)
    end

    x0 = zeros(ComplexF64,n-1,n-1,1,1)
    x_vcycle, = KrylovMethods.fgmres(A, vec(b), 1, tol=1e-10, maxIter=1,
                                                    M=M, x=vec(x0), out=-1, flexible=true)
    x_vcycle_channels = complex_grid_to_channels!(x_vcycle, n)
    return x_vcycle, x_vcycle_channels
end

function generate_r_vcycle!(n, kappa, omega, gamma, x_true; v2_iter=10, level=3)
    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    h = 1.0 ./ n
    b_true = helmholtz_chain!(x_true, helmholtz_matrix; h=h)

    x_vcycle, x_vcycle_channels = generate_vcycle!(n, kappa, omega, gamma, b_true; v2_iter=v2_iter, level=level)
    x_vcycle = reshape(x_vcycle, n-1, n-1, 1, 1)
    e_true = x_true .- x_vcycle
    r_vcycle = b_true .- helmholtz_chain!(x_vcycle, helmholtz_matrix; h=h)

    return r_vcycle, e_true
end

function generate_random_data!(m, n, kappa, omega, gamma; v2_iter=10, level=3,
                                                          e_vcycle_input=true, data_augmentetion=true,
                                                          cifar_kappa=true, kappa_input=true, kappa_smooth=true)
    h = 1.0./n;

    dataset = Tuple[]
    m = data_augmentetion == true ? Int(0.5*m) : m
    for i = 1:m

        # Generate Model
        kappa = cifar_kappa == true ? cifar_model!(n;smooth=kappa_smooth) : kappa

        # Generate Random Sample
        x_true = randn(ComplexF64,n-1,n-1, 1, 1)

        # Generate r,e
        r_vcycle, e_true = generate_r_vcycle!(n, kappa, omega, gamma, x_true)
        r_vcycle_channels = complex_grid_to_channels!(r_vcycle, n)
        e_true_channels = complex_grid_to_channels!(e_true, n)

        # Generate e-vcycle
        if e_vcycle_input == true
            e_vcycle, e_vcycle_channels = generate_vcycle!(n, kappa, omega, gamma, r_vcycle; v2_iter=v2_iter, level=level)
            input = cat(e_vcycle_channels, r_vcycle_channels, dims=3)
        else
            input = r_vcycle_channels
        end

        # Append Model
        input = kappa_input == true ? cat(input, reshape(kappa, n-1, n-1, 1, 1), dims=3) : input

        append!(dataset,[(input, e_true_channels)])

        if data_augmentetion == true
            (er2,e2) = dataset[rand(1:size(dataset,1))]

            r2 = er2[:,:,end-1:end,:]
            scalar = abs(rand(Float64))
            rt = scalar*r_vcycle_channels+(1-scalar)*r2

            scale = (scalar*norm(r_vcycle_channels) + (1-scalar)*norm(r2))/norm(rt)
            rt = rt * scale
            et = (scalar*e_true_channels+(1-scalar)*e2)*scale

            if e_vcycle_input == true
                et_vcycle, et_vcycle_channels = generate_vcycle!(n, kappa, omega, gamma, rt[:,:,1,1] + im*rt[:,:,2,1])
                input = cat(et_vcycle_channels, rt, dims=3)
            else
                input = rt
            end
            if kappa_input == true
                kappa = cifar_model!(n;smooth=kappa_smooth)
                input = cat(input, reshape(kappa, n-1, n-1, 1, 1), dims=3)
            end
            append!(dataset,[(input, et)])
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
        b = zeros(ComplexF64,n-1,n-1, 1, 1)
        b[src[1],src[2],1,1] = 1.0 ./mean(h.^2)

        # V-cycle
        x = zeros(ComplexF64,n-1,n-1,1,1)
        A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))
        function M(v)
            v = reshape(v, n-1, n-1)
            x = zeros(ComplexF64,n-1,n-1)
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
