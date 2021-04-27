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

function unet_vs_vcycle_graph!(title, vc_unet_res, unet_res, vc_vc_res, vc_res, j_unet_res; after_vcycle=false, e_vcycle_input=false)
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

function unet_outof_gmres_graph!(title, vu_res, v_res, vv_res, vuv_res, vuj_res; after_vcycle=false, e_vcycle_input=false)
    iterations = length(v_res)
    iter = range(1, length=iterations)

    factor = convergence_factor!(v_res)
    factor_text = "v=$(factor)"
    p = plot(iter,v_res,label="V $(factor)")

    factor = convergence_factor!(vu_res)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(iter,vu_res,label="VU $(factor)")

    factor = convergence_factor!(vuv_res)
    factor_text = "$(factor_text) vuv=$(factor)"
    plot!(iter,vuv_res,label="VUV $(factor)")

    factor = convergence_factor!(vuj_res)
    factor_text = "$(factor_text) vuj=$(factor)"
    plot!(iter,vuj_res,label="VUJ $(factor)")

    factor = convergence_factor!(vv_res)
    factor_text = "$(factor_text) vv=$(factor)"
    plot!(iter,vv_res,label="VV $(factor)")

    yaxis!(L"\Vert b - Hx \Vert_2", :log10)
    xlabel!("Iterations")

    savefig("test/unet/results/$(title) $(factor_text)")
    @info "$(Dates.format(now(), "HH:MM:SS")) - Convergence Factors : $(factor_text)"
end

function loss!(x, y)
    return norm(x .- y) / norm(y)
end

function unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, after_vcycle, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, relax_jacobi; v2_iter=10, level=3, axb=false, norm_input=false, log_error=true, test_name="", before_jacobi=false)

    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    h = 1.0./n
    if axb == true
        r_vcycle = helmholtz_chain!(x_true, helmholtz_matrix; h=h)
    else
        r_vcycle, _ = generate_r_vcycle!(n, kappa, omega, gamma, x_true)
    end

    coefficient = h^2 # norm_input == true ? h^2 : 1.0
    i = 0
    df = DataFrame(Title=["Title"],EB=[1.0],RB=[1.0],EA=[1.0],RA=[1.0])
    df_unet = DataFrame(Title=["Title"],E=[1.0],R=[1.0])
    # dt_training = DataFrame(RR=vec(zeros(n-1,n-1)), RI=vec(zeros(n-1,n-1)), KAPPA=vec(kappa), ER=vec(zeros(n-1,n-1)), EI=vec(zeros(n-1,n-1)))

    A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), helmholtz_matrix; h=h))

    function M_Unet(r)
        # dt_training.RR = real(r)
        # dt_training.RI = imag(r)
        r = reshape(r, n-1, n-1)
        rj = reshape(r, n-1, n-1)
        e = zeros(c_type, n-1, n-1)
        ej = zeros(c_type, n-1, n-1)

        if before_jacobi == true
            ej = jacobi_helmholtz_method!(n, h, e, r, helmholtz_matrix)
            rj = r - reshape(A(ej), n-1, n-1)
            if log_error == true && i == 5
                @info "$(Dates.format(now(), "HH:MM:SS")) - r = $(norm(rj)) $(size(rj)) e= $(norm(ej)) $(size(ej))"
            end
        end

        if e_vcycle_input == true
            e, = v_cycle_helmholtz!(n, h, e, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
            input = cat(complex_grid_to_channels!(reshape(e, n-1, n-1, 1, 1), n),
                                complex_grid_to_channels!(reshape(r, n-1, n-1, 1, 1), n), dims=3)
        else
            input = complex_grid_to_channels!(reshape(rj , n-1, n-1, 1, 1) , n)
        end

        input = kappa_input == true ? cat(input, reshape(kappa, n-1, n-1, 1, 1), dims=3) : input
        input = gamma_input == true ? cat(input, reshape(gamma, n-1, n-1, 1, 1), dims=3) : input

        e_unet = (model(u_type.(input)|>cgpu)|>cpu)
        e_unet = reshape((e_unet[:,:,1,1] + im*e_unet[:,:,2,1]),n-1,n-1,1,1) .* coefficient
        e = reshape(e_unet, n-1,n-1)

        e = ej + e
        if log_error == true && i == 5
            e_true = fgmres_v_cycle_helmholtz!(n, h, rj, kappa, omega, gamma)
            loss_e = loss!(e_unet, complex_grid_to_channels!(e_true, n))
            df.EB = [loss_e]

            r_unet = reshape(A(e), n-1, n-1)
            loss_r = loss!(r_unet, rj)
            df.RB = [loss_r]

            df.Title = ["$(i) j=$("$(relax_jacobi)"[1]) v=$("$(after_vcycle)"[1])"]
            @info "$(Dates.format(now(), "HH:MM:SS")) - $(i) Before loss_e $(loss_e) $(loss!(e, e_true)) $(loss!(e ./ norm(e), e_true ./ norm(e_true))) $(size(e)) loss_r $(loss_r) $(loss!(r_unet ./ norm(r_unet), rj)) $(size(r_unet)) e_true $(norm(e_true)) $(size(e_true)) e $(norm(e)) r $(norm(rj)) $(size(r))"
        end

        if relax_jacobi == true
            e = jacobi_helmholtz_method!(n, h, e, r, helmholtz_matrix)
        end

        if after_vcycle == true
            e, = v_cycle_helmholtz!(n, h, e, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        end

        if log_error == true && i == 5
            e_true = fgmres_v_cycle_helmholtz!(n, h, r, kappa, omega, gamma)
            # dt_training.ER = real(vec(e_true))
            # dt_training.EI = imag(vec(e_true))
            # CSV.write("training set.csv", dt_training, delim = ';',append=true)
            loss_e = loss!(complex_grid_to_channels!(e, n), complex_grid_to_channels!(e_true, n))
            df.EA = [loss_e]

            r_unet = reshape(A(e), n-1, n-1)
            loss_r = loss!(r_unet, r)
            df.RA = [loss_r]

            @info "$(Dates.format(now(), "HH:MM:SS")) - $(i) After j=$("$(relax_jacobi)"[1]) v=$("$(after_vcycle)"[1]) loss_e $(loss_e) $(loss!(e, e_true)) $(loss!(e ./ norm(e), e_true ./ norm(e_true))) $(size(e)) loss_r $(loss_r) $(loss!(r_unet ./ norm(r_unet), r)) $(size(r_unet)) e_true $(norm(e_true)) $(size(e_true)) e_unet $(norm(e))"

            CSV.write("$(test_name) t_n=$(n) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) unet error.csv",
                                df, delim = ';',append=true)
        end
        i = i + 1
        return vec(e)
    end

    function M(r)
        r = reshape(r, n-1, n-1)
        e_vcycle = zeros(c_type,n-1,n-1)
        e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        end
        #log_error = false
        # if log_error == true && (i == 20 || i == 40 || i == 60)
        #     e_true = fgmres_v_cycle_helmholtz!(n, h, r, kappa, omega, gamma)
        #     loss_e = loss!(e_vcycle, e_true)
        #
        #     r_vcycle = reshape(A(e_vcycle), n-1, n-1)
        #     loss_r = loss!(r_vcycle, r)
        #
        #     @info "$(Dates.format(now(), "HH:MM:SS")) - $(i) vcycle loss_e $(loss_e) $(loss!(e_vcycle ./ norm(e_vcycle), e_true ./ norm(e_true))) loss_r $(loss_r) $(loss!(r_vcycle ./ norm(r_vcycle), r)) e_true $(norm(e_true)) e_vcycle $(norm(e_vcycle))"
        #     df_unet.E = [loss_e]
        #     df_unet.R = [loss_r]
        #     df_unet.Title = ["$(i) v=$("$(after_vcycle)"[1])"]
        #     CSV.write("$(test_name) t_n=$(n) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) vcycle error.csv", df_unet, delim = ';',append=true)
        # end
        # i = i + 1
        return vec(e_vcycle)
    end

    x_init = zeros(c_type,n-1,n-1)
    x3,flag3,err3,iter3,resvec3= KrylovMethods.fgmres(A, vec(r_vcycle), 3, tol=1e-30, maxIter=1,
                                                    M=M, x=vec(x_init), out=-1, flexible=true)
    i = 1
    x1,flag1,err1,iter1,resvec1 = KrylovMethods.fgmres(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                    M=M_Unet, x=vec(x3), out=-1, flexible=true)

    x_init = zeros(c_type,n-1,n-1)
    x3,flag3,err3,iter3,resvec3= KrylovMethods.fgmres(A, vec(r_vcycle), 3, tol=1e-30, maxIter=1,
                                                    M=M, x=vec(x_init), out=-1, flexible=true)
    i = 1
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

function check_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=10, level=3, smooth=false, threshold=50, axb=false, norm_input=false, before_jacobi=false)
    unet_results = zeros(m,3,restrt*max_iter)
    vcycle_results = zeros(m,2,restrt*max_iter)

    CSV.write("$(test_name) t_n=$(n) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) unet error.csv",
                        DataFrame(Title=[],EB=[],RB=[],EA=[],RA=[]), delim = ';')
    CSV.write("$(test_name) t_n=$(n) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) vcycle error.csv",
                        DataFrame(Title=[],E=[],R=[]), delim = ';')
    # CSV.write("training set.csv", DataFrame(RR=[], RI=[], KAPPA=[], ER=[], EI=[]), delim = ';',append=true)

    for i=1:m
        x_true = randn(c_type,n-1,n-1, 1, 1)
        kappa = generate_kappa!(n; type=kappa_type, smooth=smooth, threshold=threshold)
        resvec1, resvec2 = unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, true, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, false; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input, log_error=false, test_name=test_name, before_jacobi=before_jacobi)
        unet_results[i,1,:] = resvec1
        vcycle_results[i,1,:] = resvec2
        resvec1, resvec2 = unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, false, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, false; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input, log_error=false, test_name=test_name, before_jacobi=before_jacobi)
        unet_results[i,2,:] = resvec1
        vcycle_results[i,2,:] = resvec2
        resvec1, resvec2 = unet_vs_vcycle!(model, n, kappa, omega, gamma, x_true, false, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, true; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input, log_error=false, test_name=test_name, before_jacobi=before_jacobi)
        unet_results[i,3,:] = resvec1
    end
    unet_vs_vcycle_graph!("$(test_name) t_n=$(n) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) t_j=$("$(before_jacobi)"[1])",
                    mean(unet_results[:,1,:],dims=1)', mean(unet_results[:,2,:],dims=1)', mean(vcycle_results[:,1,:],dims=1)', mean(vcycle_results[:,2,:],dims=1)', mean(unet_results[:,3,:],dims=1)')

    CSV.write("test/unet/results/$(test_name) t_n=$(n) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) t_j=$("$(before_jacobi)"[1]) preconditioner test.csv", DataFrame(VU=vec(mean(unet_results[:,1,:],dims=1)'),
                                                                U=vec(mean(unet_results[:,2,:],dims=1)'),
                                                                VV=vec(mean(vcycle_results[:,1,:],dims=1)'),
                                                                V=vec(mean(vcycle_results[:,2,:],dims=1)'),
                                                                JU=vec(mean(unet_results[:,3,:],dims=1)')), delim = ';')
end

function check_full_solution_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=10, level=3, smooth=false, threshold=50, axb=false, norm_input=false)
    unet_results = zeros(m,restrt*max_iter-1)
    h = 1.0./n
    for i=1:m
        x_true = randn(c_type,n-1,n-1, 1, 1)
        kappa = generate_kappa!(n; type=kappa_type, smooth=smooth, threshold=threshold)
        s_matrix, h_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)

        if axb == true
            r = helmholtz_chain!(x_true, h_matrix; h=h)
        else
            r, _ = generate_r_vcycle!(n, kappa, omega, gamma, x_true)
        end

        e = zeros(c_type, n-1, n-1, 1, 1)
        ae = zeros(c_type, n-1, n-1, 1, 1)
        for j = 1:restrt*max_iter-1
            # e, = v_cycle_helmholtz!(n, h, reshape(e,n-1,n-1), reshape(r,n-1,n-1), kappa, omega, gamma; v2_iter = v2_iter, level = level)
            # e = reshape(e, n-1, n-1, 1, 1)
            # e = jacobi_helmholtz_method!(n, h, reshape(e,n-1,n-1), reshape(r,n-1,n-1), h_matrix)
            ae = helmholtz_chain!(e, h_matrix; h=h)|> cpu
            correct_r = r - ae

            unet_results[i,j] = norm(correct_r) / norm(r)
            @info "$(i) $(j) : $(unet_results[i,j]) r=$(norm(r)) e=$(norm(e)) ae=$(norm(ae)) correct_r=$(norm(correct_r))"

            #
            correct_e = model(cat(complex_grid_to_channels!(reshape(correct_r,n-1,n-1,1,1).* (h^2), n), kappa, gamma, dims=3)|> cgpu)|> cpu
            e = e + (reshape((correct_e[:,:,1,1] + im*correct_e[:,:,2,1]),n-1,n-1,1,1))
            @info "$(i) $(j) : e=$(norm(e)) correct_e=$(norm(correct_e))"
            # ae = helmholtz_chain!(e, h_matrix; h=h)|> cpu
            # e = jacobi_helmholtz_method!(n, h, e, r, h_matrix)

        end
    end
    CSV.write("test/unet/results/$(test_name) only unet.csv", DataFrame(U=vec(mean(unet_results[:,:],dims=1)')), delim = ';')
end

function check_gmres_alternatively_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=10, level=3, smooth=false, threshold=50, axb=false, norm_input=false, before_jacobi=false)
    vu_results = zeros(m,max_iter*5)
    vv_results = zeros(m,max_iter*5)
    v_results = zeros(m,max_iter*5)
    vuv_results = zeros(m,max_iter*5)
    vuj_results = zeros(m,max_iter*5)

    h = 1.0./n

    for i=1:m
        x_true = randn(c_type, n-1, n-1, 1, 1)
        kappa = generate_kappa!(n; type=kappa_type, smooth=smooth, threshold=threshold)
        s_matrix, h_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
        if axb == true
            b_true = helmholtz_chain!(x_true, h_matrix; h=h)
        else
            b_true, _ = generate_r_vcycle!(n, kappa, omega, gamma, x_true)
        end

        A(v) = vec(helmholtz_chain!(reshape(v, n-1, n-1, 1, 1), h_matrix; h=h))

        function M(r)
            r = reshape(r, n-1, n-1)
            e_vcycle = zeros(c_type, n-1, n-1)
            e_vcycle, = v_cycle_helmholtz!(n, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
            return vec(e_vcycle)
        end

        x_vcycle = zeros(c_type,n-1,n-1)
        for j = 1:max_iter
            x_vcycle,flag,err,iter,resvec = KrylovMethods.fgmres(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)

            r = b_true - helmholtz_chain!(reshape(x_vcycle,n-1,n-1,1,1), h_matrix; h=h) |> cpu
            x_unet = (model(cat(complex_grid_to_channels!(r, n), kappa, gamma, dims=3)|> cgpu) .* (h^2))|> cpu
            @info "VU $(i) $(j) x_unet $(norm(x_unet)) x_vcycle $(norm(x_vcycle)) r $(norm(r)) b_true $(norm(b_true))"
            x_vcycle = reshape(x_vcycle,n-1,n-1) + (x_unet[:,:,1,1] + im*x_unet[:,:,2,1])
            vu_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
            # @info "VU $(i) $(j) : $(vu_results[i,5*(j-1)+1])"
        end
        x_vcycle = zeros(c_type,n-1,n-1)
        for j = 1:max_iter
            x_vcycle,flag,err,iter,resvec = KrylovMethods.fgmres(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)

            v_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
            # @info "V $(i) $(j) : $(v_results[i,5*(j-1)+1])"
        end
        x_vcycle = zeros(c_type,n-1,n-1)
        for j = 1:max_iter
            x_vcycle,flag,err,iter,resvec = KrylovMethods.fgmres(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)
            x_vcycle, = v_cycle_helmholtz!(n, h, reshape(x_vcycle,n-1,n-1), reshape(b_true,n-1,n-1), kappa, omega, gamma; v2_iter = v2_iter, level = level)
            vv_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
            # @info "VV $(i) $(j) : $(vv_results[i,5*(j-1)+1])"
        end
        x_vcycle = zeros(c_type,n-1,n-1)
        for j = 1:max_iter

            x_vcycle,flag,err,iter,resvec = KrylovMethods.fgmres(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)
            r = b_true - helmholtz_chain!(reshape(x_vcycle,n-1,n-1,1,1), h_matrix; h=h) |> cpu
            x_unet = (model(cat(complex_grid_to_channels!(r, n), kappa, gamma, dims=3)|> cgpu) .* (h^2))|> cpu
            @info "VUV $(i) $(j) x_unet $(norm(x_unet)) x_vcycle $(norm(x_vcycle)) r $(norm(r)) b_true $(norm(b_true))"
            x_vcycle = reshape(x_vcycle,n-1,n-1) + (x_unet[:,:,1,1] + im*x_unet[:,:,2,1])
            x_vcycle, = v_cycle_helmholtz!(n, h, x_vcycle, reshape(b_true,n-1,n-1), kappa, omega, gamma; v2_iter = v2_iter, level = level)
            vuv_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
            # @info "VUV $(i) $(j) : $(vuv_results[i,5*(j-1)+1])"
        end
        x_vcycle = zeros(c_type,n-1,n-1)
        for j = 1:max_iter
            x_vcycle,flag,err,iter,resvec = KrylovMethods.fgmres(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)
            r = b_true - helmholtz_chain!(reshape(x_vcycle,n-1,n-1,1,1), h_matrix; h=h) |> cpu
            x_unet = (model(cat(complex_grid_to_channels!(r, n), kappa, gamma, dims=3)|> cgpu) .* (h^2))|> cpu
            @info "VUJ $(i) $(j) x_unet $(norm(x_unet)) x_vcycle $(norm(x_vcycle)) r $(norm(r)) b_true $(norm(b_true))"
            x_vcycle = reshape(x_vcycle,n-1,n-1) + (x_unet[:,:,1,1] + im*x_unet[:,:,2,1])
            x_vcycle = jacobi_helmholtz_method!(n, h, x_vcycle, reshape(b_true,n-1,n-1), h_matrix)
            vuj_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
            # @info "VUJ $(i) $(j) : $(vuj_results[i,5*(j-1)+1])"
        end
    end
    unet_outof_gmres_graph!("$(test_name) t_n=$(n) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) t_j=$("$(before_jacobi)"[1])",
                    mean(vu_results[:,:],dims=1)', mean(v_results[:,:],dims=1)', mean(vv_results[:,:],dims=1)', mean(vuv_results[:,:],dims=1)', mean(vuj_results[:,:],dims=1)')
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
    input = gamma_input == true ? cat(input, gamma, dims=3) : input

    e_unet = model(u_type.(input)|>cgpu)
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
