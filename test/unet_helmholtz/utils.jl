function unet_vs_vcycle_title!(after_vcycle,e_vcycle_input)
    if after_vcycle == true
        if e_vcycle_input == true
            return "M=Vcycle(r,Unet(Vcycle(r,0),r)) vs M=Vcycle(r,Vcycle(r,0))"
        else
            return "M=Vcycle(r,Unet(r)) vs M=Vcycle(r,Vcycle(r,0))"
        end
    else
        if e_vcycle_input == true
            return "M=Unet(Vcycle(r,0),r) vs M=Vcycle(r,0)"
        else
            return "M=Unet(r) vs M=Vcycle(r,0)"
        end
    end
end

function unet_vs_vcycle_graph!(title, resvec1, resvec2, after_vcycle, e_vcycle_input)
    iter = range(1, length=length(resvec1))
    sub_title = "residual graph"
    if after_vcycle == true
        sub_title = "with after vcycle $(sub_title)"
    end
    graph_title = unet_vs_vcycle_title!(after_vcycle,e_vcycle_input)
    p = plot(iter,resvec1,label="With UNet")
    plot!(iter,resvec2,label="Without UNet")
    yaxis!(L"\Vert b - Hx \Vert_2", :log10)
    xlabel!("Iterations")
    title!("$(graph_title)")
    savefig("$(title) $(sub_title)")
end

function unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, after_vcycle, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter; v2_iter=10, level=3)

    _, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    h = 1.0./n

    r_vcycle, e_true = generate_r_vcycle!(n, kappa, omega, gamma, x_true)

    A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))

    function M_Unet(r)
        r = reshape(r, n-1, n-1)
        if e_vcycle_input == true
            e_vcycle = zeros(ComplexF64,n-1,n-1)
            e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)

            input = cat(complex_grid_to_channels!(reshape(e_vcycle, n-1, n-1, 1, 1), n),
                                complex_grid_to_channels!(reshape(r, n-1, n-1, 1, 1), n), dims=3)
        else
            input = complex_grid_to_channels!(reshape(r, n-1, n-1, 1, 1), n)
        end

        input = kappa_input == true ? cat(input, reshape(kappa, n-1, n-1, 1, 1), dims=3) : input
        input = gamma_input == true ? cat(input, complex_grid_to_channels!(gamma, n), dims=3) : input

        e_unet = model(input|>cgpu)|>cpu

        e_vcycle = e_unet[:,:,1,1] +im*e_unet[:,:,2,1]
        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        end
        return vec(e_vcycle)
    end

    function M(r)
        r = reshape(r, n-1, n-1)
        e_vcycle = zeros(ComplexF64,n-1,n-1)
        e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        end
        return vec(e_vcycle)
    end

    x = zeros(ComplexF64,n-1,n-1)
    x1,flag1,err1,iter1,resvec1 = KrylovMethods.fgmres(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                    M=M_Unet, x=vec(x), out=-1, flexible=true)
    @info "$(Dates.format(now(), "HH:MM:SS")) - Error with UNet: $(norm_diff!(reshape(x1,n-1,n-1),e_true[:,:,1,1]))"

    x = zeros(ComplexF64,n-1,n-1)
    x2,flag2,err2,iter2,resvec2 = KrylovMethods.fgmres(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                    M=M, x=vec(x), out=-1, flexible=true)

    @info "$(Dates.format(now(), "HH:MM:SS")) - Error without UNet: $(norm_diff!(reshape(x2,n-1,n-1),e_true[:,:,1,1]))"

    return resvec1, resvec2
end

function check_model!(test_name, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=10, level=3, smooth=true)
    model = create_model!(e_vcycle_input,kappa_input,gamma_input;kernel = kernel)

    model = model|>cpu
    @load "$(test_name).bson" model
    @info "$(Dates.format(now(), "HH:MM:SS")) - Load Model"
    model = model|>cgpu

    unet_results = zeros(m,2,restrt*max_iter)
    vcycle_results = zeros(m,2,restrt*max_iter)
    for i=1:m
        x_true = randn(ComplexF64,n-1,n-1, 1, 1)
        kappa = cifar_model!(n;smooth=smooth)
        resvec1, resvec2 = unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, true, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter; v2_iter=v2_iter, level=level)
        unet_results[i,1,:] = resvec1
        vcycle_results[i,1,:] = resvec2
        resvec1, resvec2 = unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, false, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter; v2_iter=v2_iter, level=level)
        unet_results[i,2,:] = resvec1
        vcycle_results[i,2,:] = resvec2
    end

    unet_vs_vcycle_graph!("$(test_name) test_n=$(n) f=$(f)", mean(unet_results[:,1,:],dims=1)', mean(vcycle_results[:,1,:],dims=1)', true, e_vcycle_input)
    unet_vs_vcycle_graph!("$(test_name) test_n=$(n) f=$(f)", mean(unet_results[:,2,:],dims=1)', mean(vcycle_results[:,2,:],dims=1)', false, e_vcycle_input)
end
