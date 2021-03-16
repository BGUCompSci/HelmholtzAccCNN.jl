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

function convergence_factor!(vector)
    length = argmin(vector)[1]
    return round(((vector[length] / vector[1])^(1.0 / length)), digits=3)
end

function unet_vs_vcycle_graph!(title, vc_unet_res, unet_res, vc_vc_res, vc_res, j_unet_res, after_vcycle, e_vcycle_input)
    iterations = length(vc_unet_res)
    iter = range(1, length=iterations)

    # Unet
    factor = convergence_factor!(unet_res)
    factor_text = "u=$(factor)"
    p = plot(iter,unet_res,label="UNet(r) $(factor)")

    # Jacobi Unet
    factor = convergence_factor!(j_unet_res)
    factor_text = "$(factor_text) ju=$(factor)"
    plot!(iter,j_unet_res,label="J(UNet(r)) $(factor)")

    # Vcycle
    factor = convergence_factor!(vc_res)
    factor_text = "$(factor_text) v=$(factor)"
    plot!(iter,vc_res,label="V(r,0) $(factor)")

    # Vcycle Vcycle
    factor = convergence_factor!(vc_vc_res)
    factor_text = "$(factor_text) vv=$(factor)"
    plot!(iter,vc_vc_res,label="V(r,V(r,0)) $(factor)")

    # Vcycle Unet
    factor = convergence_factor!(vc_unet_res)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(iter,vc_unet_res,label="V(r,UNet(r)) $(factor)")

    yaxis!(L"\Vert b - Hx \Vert_2", :log10)
    xlabel!("Iterations")

    savefig("test/unet/results/$(title) $(factor_text)")
    @info "$(Dates.format(now(), "HH:MM:SS")) - Convergence Factors : $(factor_text)"
end

function unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, after_vcycle, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, relax_jacobi; v2_iter=10, level=3, axb=false, norm_input=false)

    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    h = 1.0./n

    if axb == true
        r_vcycle = helmholtz_chain!(x_true, helmholtz_matrix; h=h)
    else
        r_vcycle, _ = generate_r_vcycle!(n, kappa, omega, gamma, x_true)
    end
    coefficient = norm_input == true ? h^2 : 1.0

    A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))

    function M_Unet(r)
        r = reshape(r, n-1, n-1)
        if e_vcycle_input == true
            e_vcycle = zeros(c_type,n-1,n-1)
            e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, coefficient .* r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
            input = cat(complex_grid_to_channels!(reshape(e_vcycle, n-1, n-1, 1, 1), n),
                                complex_grid_to_channels!(reshape(coefficient .* r, n-1, n-1, 1, 1), n), dims=3)
        else
            input = complex_grid_to_channels!(reshape(coefficient .* r, n-1, n-1, 1, 1), n)
        end

        input = kappa_input == true ? cat(input, reshape(kappa, n-1, n-1, 1, 1), dims=3) : input
        input = gamma_input == true ? cat(input, complex_grid_to_channels!(gamma, n), dims=3) : input

        e_unet = model(input|>cgpu)|>cpu

        e_vcycle = (e_unet[:,:,1,1] +im*e_unet[:,:,2,1])

        if relax_jacobi == true
            e_vcycle = jacobi_helmholtz_method!(n, h, e_vcycle, r, helmholtz_matrix)
        end

        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        end

        return vec(e_vcycle)
    end

    function M(r)
        r = reshape(r, n-1, n-1)
        e_vcycle = zeros(c_type,n-1,n-1)
        e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        end
        return vec(e_vcycle)
    end

    x_init = zeros(c_type,n-1,n-1)
    x3,flag3,err3,iter3,resvec3= KrylovMethods.fgmres(A, vec(r_vcycle), 3, tol=1e-30, maxIter=1,
                                                    M=M, x=vec(x_init), out=-1, flexible=true)
    x1,flag1,err1,iter1,resvec1 = KrylovMethods.fgmres(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                    M=M_Unet, x=vec(x3), out=-1, flexible=true)

    x_init = zeros(c_type,n-1,n-1)
    x3,flag3,err3,iter3,resvec3= KrylovMethods.fgmres(A, vec(r_vcycle), 3, tol=1e-30, maxIter=1,
                                                    M=M, x=vec(x_init), out=-1, flexible=true)
    x2,flag2,err2,iter2,resvec2 = KrylovMethods.fgmres(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                    M=M, x=vec(x3), out=-1, flexible=true)

    return resvec1, resvec2
end

function load_model!(test_name, e_vcycle_input, kappa_input, gamma_input;kernel=(3,3), model_type=SUnet)
    model = create_model!(e_vcycle_input, kappa_input, gamma_input; kernel=kernel, type=model_type)

    model = model|>cpu
    @load "$(test_name).bson" model
    @info "$(Dates.format(now(), "HH:MM:SS")) - Load Model"
    model = model|>cgpu

    return model
end

function check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=10, level=3, smooth=false, threshold=50, axb=false, norm_input=false)
    unet_results = zeros(m,3,restrt*max_iter)
    vcycle_results = zeros(m,2,restrt*max_iter)
    for i=1:m
        x_true = randn(c_type,n-1,n-1, 1, 1)
        kappa = generate_kappa!(n; type=kappa_type, smooth=smooth, threshold=threshold)
        resvec1, resvec2 = unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, true, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, false; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input)
        unet_results[i,1,:] = resvec1
        vcycle_results[i,1,:] = resvec2
        resvec1, resvec2 = unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, false, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, false; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input)
        unet_results[i,2,:] = resvec1
        vcycle_results[i,2,:] = resvec2
        resvec1, resvec2 = unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, false, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, true; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input)
        unet_results[i,3,:] = resvec1
    end
    unet_vs_vcycle_graph!("$(test_name) t_n=$(n) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1])",
                    mean(unet_results[:,1,:],dims=1)', mean(unet_results[:,2,:],dims=1)', mean(vcycle_results[:,1,:],dims=1)', mean(vcycle_results[:,2,:],dims=1)', mean(unet_results[:,3,:],dims=1)',
                    true, e_vcycle_input)
end

function check_point_source_problem!(test_name, model, n, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=10, level=3)
    b = zeros(c_type,n-1,n-1)
    h = 1.0 / n
    b[floor(Int32,n / 2.0),floor(Int32,n / 2.0)] = 1.0 ./ mean(h.^2);
    coefficient = h^2

    if e_vcycle_input == true
        x = zeros(c_type,n-1,n-1)
        x, = v_cycle_helmholtz!(n, h, x, coefficient .* b, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        input = cat(complex_grid_to_channels!(reshape(x, n-1, n-1, 1, 1), n),
                            complex_grid_to_channels!(reshape(coefficient .* b, n-1, n-1, 1, 1), n), dims=3)
    else
        input = complex_grid_to_channels!(reshape(coefficient .* b, n-1, n-1, 1, 1), n)
    end
    input = kappa_input == true ? cat(input, reshape(kappa, n-1, n-1, 1, 1), dims=3) : input
    input = gamma_input == true ? cat(input, complex_grid_to_channels!(gamma, n), dims=3) : input

    e_unet = model(input|>cgpu)
    heatmap(e_unet[:,:,1,1]|>cpu, color=:grays)
    savefig("test/unet/results/$(test_name) $(n) e unet")

    x = zeros(c_type,n-1,n-1)
    x, = v_cycle_helmholtz!(n, h, x, b, kappa, omega, gamma; v2_iter=v2_iter, level=level)
    heatmap(real(x), color=:grays)
    savefig("test/unet/results/$(test_name) $(n) e vcycle")

    x, = v_cycle_helmholtz!(n, h, x, b, kappa, omega, gamma; v2_iter=v2_iter, level=level)
    heatmap(real(x), color=:grays)
    savefig("test/unet/results/$(test_name) $(n) e vcycle vcycle")

    x = e_unet[:,:,1,1]+im*e_unet[:,:,2,1]
    x, = v_cycle_helmholtz!(n, h, x|>cpu, b, kappa, omega, gamma; v2_iter=v2_iter, level=level)
    heatmap(real(x), color=:grays)
    savefig("test/unet/results/$(test_name) $(n) e vcycle unet")
end
