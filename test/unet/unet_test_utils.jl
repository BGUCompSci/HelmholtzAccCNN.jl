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

function unet_vs_vcycle_graph!(title, vc_unet_res, vc_unet_times, unet_res, unet_times, vc_res, vc_times, j_unet_res, j_unet_times; after_vcycle=false, e_vcycle_input=false)
    iterations = length(vc_unet_res)
    iter = range(1, length=iterations)

    # Unet
    factor = convergence_factor!(unet_res)
    factor_text = "u=$(factor)"
    p = plot(unet_times,unet_res,label="UNet(r) $(factor)")

    # Jacobi Unet
    factor = convergence_factor!(j_unet_res)
    factor_text = "$(factor_text) ju=$(factor)"
    plot!(j_unet_times,j_unet_res,label="J(UNet(r)) $(factor)")

    # Vcycle
    factor = convergence_factor!(vc_res)
    factor_text = "$(factor_text) v=$(factor)"
    plot!(vc_times,vc_res,label="V(r,0) $(factor)")

    # Vcycle Vcycle
    # factor = convergence_factor!(vc_vc_res)
    # factor_text = "$(factor_text) vv=$(factor)"
    # plot!(iter,vc_vc_res,label="V(r,V(r,0)) $(factor)")

    # Vcycle Unet
    factor = convergence_factor!(vc_unet_res)
    factor_text = "$(factor_text) vu=$(factor)"
    plot!(vc_unet_times,vc_unet_res,label="V(r,UNet(r)) $(factor)")

    yaxis!(L"\Vert b - Hx \Vert_2", :log10)
    xlabel!("milliseconds")

    savefig("test/unet/results/$(title) $(factor_text)")
    @info "$(Dates.format(now(), "HH:MM:SS.sss")) - Convergence Factors : $(factor_text)"
    return "$(title) $(factor_text)"
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
    @info "$(Dates.format(now(), "HH:MM:SS.sss")) - Convergence Factors : $(factor_text)"
end

function loss!(x, y)
    return norm(x .- y) / norm(y)
end

function unet_vs_vcycle!(model, n, m, kappa, omega, gamma, x_true, after_vcycle, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, relax_jacobi; v2_iter=10, level=3, axb=false, norm_input=false, log_error=true, test_name="", before_jacobi=false, unet_in_vcycle=false)

    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    h = r_type(2.0 ./ (n+m))
    if axb == true
        r_vcycle = zeros(c_type,n-1,m-1,1,1)
        r_vcycle[floor(Int32,n / 2.0),floor(Int32,m / 2.0),1,1] = 1.0 ./mean(h.^2);
        # r_vcycle = helmholtz_chain!(x_true, helmholtz_matrix; h=h)
    else
        r_vcycle, _ = generate_r_vcycle!(n, m, kappa, omega, gamma, x_true)
    end

    coefficient = h^2 # norm_input == true ? h^2 : 1.0
    i = 0
    df = DataFrame(Title=["Title"],EB=[1.0],RB=[1.0],EA=[1.0],RA=[1.0])
    df_unet = DataFrame(Title=["Title"],E=[1.0],R=[1.0])
    # dt_training = DataFrame(RR=vec(zeros(n-1,m-1)), RI=vec(zeros(n-1,m-1)), KAPPA=vec(kappa), ER=vec(zeros(n-1,m-1)), EI=vec(zeros(n-1,m-1)))

    A(v) = vec(helmholtz_chain!(reshape(v, n-1, m-1, 1, 1), helmholtz_matrix; h=h))

    function M_Unet(r)
        # dt_training.RR = real(r)
        # dt_training.RI = imag(r)
        @info "$(Dates.format(now(), "HH:MM:SS.sss")) - start unet"
        r = reshape(r, n-1, m-1)
        rj = reshape(r, n-1, m-1)
        e = zeros(c_type, n-1, m-1)
        ej = zeros(c_type, n-1, m-1)

        if after_vcycle != true # after_vcycle != true # after_vcycle != true #before_jacobi == true
            ej = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
            # ej = jacobi_helmholtz_method!(n, m, h, ej, r, helmholtz_matrix)
            rj = r - reshape(A(ej), n-1, m-1)
            log_error1 = false
            if log_error1 == true && i == 5
                @info "$(Dates.format(now(), "HH:MM:SS.sss")) - r = $(norm(rj)) $(size(rj)) e= $(norm(ej)) $(size(ej))"
            end
        end

        if e_vcycle_input == true
            e, = v_cycle_helmholtz!(n, m, h, e, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
            input = cat(complex_grid_to_channels!(reshape(e, n-1, m-1, 1, 1), n),
                                complex_grid_to_channels!(reshape(r, n-1, m-1, 1, 1), n), dims=3)
        else
            input = complex_grid_to_channels!(reshape(rj , n-1, m-1, 1, 1) , n)
        end

        input = kappa_input == true ? cat(input, reshape(kappa, n-1, m-1, 1, 1), dims=3) : input
        input = gamma_input == true ? cat(input, reshape(gamma, n-1, m-1, 1, 1), dims=3) : input

        e_unet = (model(u_type.(input)|>cgpu)|>pu)
        e_unet = reshape((e_unet[:,:,1,1] + im*e_unet[:,:,2,1]),n-1,m-1,1,1) .* coefficient
        e = reshape(e_unet, n-1,m-1)

        e = ej + e
        log_error1 = false
        if log_error1 == true && i == 5
            e_true = fgmres_v_cycle_helmholtz!(n, m, h, rj, kappa, omega, gamma)
            loss_e = loss!(e_unet, complex_grid_to_channels!(e_true, n))
            df.EB = [loss_e]

            r_unet = reshape(A(e), n-1, m-1)
            loss_r = loss!(r_unet, rj)
            df.RB = [loss_r]

            df.Title = ["$(i) j=$("$(relax_jacobi)"[1]) v=$("$(after_vcycle)"[1])"]
            @info "$(Dates.format(now(), "HH:MM:SS.sss")) - $(i) Before loss_e $(loss_e) $(loss!(e, e_true)) $(loss!(e ./ norm(e), e_true ./ norm(e_true))) $(size(e)) loss_r $(loss_r) $(loss!(r_unet ./ norm(r_unet), rj)) $(size(r_unet)) e_true $(norm(e_true)) $(size(e_true)) e $(norm(e)) r $(norm(rj)) $(size(r))"
        end

        if relax_jacobi == true
            e = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
            e = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
        end

        if after_vcycle == true
            e, = v_cycle_helmholtz!(n, m, h, e, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        end

        log_error1 = false
        if log_error1 == true && i == 5
            e_true = fgmres_v_cycle_helmholtz!(n, m, h, r, kappa, omega, gamma)
            # dt_training.ER = real(vec(e_true))
            # dt_training.EI = imag(vec(e_true))
            # CSV.write("training set.csv", dt_training, delim = ';',append=true)
            loss_e = loss!(complex_grid_to_channels!(e), complex_grid_to_channels!(e_true))
            df.EA = [loss_e]

            r_unet = reshape(A(e), n-1, m-1)
            loss_r = loss!(r_unet, r)
            df.RA = [loss_r]

            @info "$(Dates.format(now(), "HH:MM:SS.sss")) - $(i) After j=$("$(relax_jacobi)"[1]) v=$("$(after_vcycle)"[1]) loss_e $(loss_e) $(loss!(e, e_true)) $(loss!(e ./ norm(e), e_true ./ norm(e_true))) $(size(e)) loss_r $(loss_r) $(loss!(r_unet ./ norm(r_unet), r)) $(size(r_unet)) e_true $(norm(e_true)) $(size(e_true)) e_unet $(norm(e))"

            CSV.write("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) unet error.csv",
                                df, delim = ';',append=true)
        end
        i = i + 1
        @info "$(Dates.format(now(), "HH:MM:SS.sss")) - end unet"
        return vec(e)
    end

    function M_VU(r)
        r = reshape(r, n-1, m-1)
        e_vcycle = zeros(c_type,n-1,m-1)
        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz_unet!(model, n, m, h, e_vcycle, r, kappa, omega, gamma; alpha=0.05, v2_iter = 2, level = 2)
        else
            e_vcycle, = v_cycle_helmholtz_unet!(model, n, m, h, e_vcycle, r, kappa, omega, gamma; alpha=0.05, v2_iter = 1, level = 2)
        end
        return vec(e_vcycle)
    end

    function M(r)

        @info "$(Dates.format(now(), "HH:MM:SS.sss")) - start vcycle"
        r = reshape(r, n-1, m-1)
        e_vcycle = zeros(c_type,n-1,m-1)
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        @info "$(Dates.format(now(), "HH:MM:SS.sss")) - end vcycle"
        return vec(e_vcycle)
    end

    function SM(r)
        r = reshape(r, n-1, m-1)
        e_vcycle = zeros(c_type,n-1,m-1)
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        return vec(e_vcycle)
    end

    x_init = zeros(c_type,n-1,m-1)
    x3,flag3,err3,iter3,resvec3 = fgmres_func(A, vec(r_vcycle), 3, tol=1e-30, maxIter=1,
                                                    M=SM, x=vec(x_init), out=-1, flexible=true)

    i = 1
    if unet_in_vcycle == true
        x1,flag1,err1,iter1,resvec1 = fgmres_func(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                        M=M_VU, x=vec(x3), out=-1, flexible=true)
    else
        x1 = x3
        resvec1 = zeros(restrt*max_iter)
        for iteration in 1:max_iter
            x1,flag1,err1,iter1,resvec = fgmres_func(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=1,
                                                            M=M_Unet, x=vec(x1), out=-1, flexible=true)
            resvec1[restrt*(iteration-1)+1:restrt*iteration] = resvec
            if log_error == true
                e_true = fgmres_v_cycle_helmholtz!(n, m, h, r_vcycle, kappa, omega, gamma; restrt=30, maxIter=15)
                error_complex = norm_diff!(reshape(x1,n-1,m-1), e_true)
                df.EA = [sqrt(real(error_complex)^2 + imag(error_complex)^2)]

                residual_complex = norm_diff!(reshape(A(x1), n-1, m-1), r_vcycle)
                df.RA = [sqrt(real(residual_complex)^2 + imag(residual_complex)^2)]

                CSV.write("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) unet error.csv",
                                    df, delim = ';',append=true)
            end
        end
    end

    x_init = zeros(c_type,n-1,m-1)
    x3,flag3,err3,iter3,resvec3= fgmres_func(A, vec(r_vcycle), 3, tol=1e-30, maxIter=1,
                                                    M=SM, x=vec(x_init), out=-1, flexible=true)
    i = 1
    x2 = x3
    resvec2 = zeros(restrt*max_iter)
    for iteration in 1:max_iter
        x2,flag2,err2,iter2,resvec = fgmres_func(A, vec(r_vcycle), restrt, tol=1e-30, maxIter=1,
                                                    M=M, x=vec(x2), out=-1, flexible=true)
        resvec2[restrt*(iteration-1)+1:restrt*iteration] = resvec
        if log_error == true
            e_true = fgmres_v_cycle_helmholtz!(n, m, h, r_vcycle, kappa, omega, gamma; restrt=30, maxIter=15)
            error_complex = norm_diff!(reshape(x2,n-1,m-1), e_true)
            df.EA = [sqrt(real(error_complex)^2 + imag(error_complex)^2)]

            residual_complex = norm_diff!(reshape(A(x2), n-1, m-1), r_vcycle)
            df.RA = [sqrt(real(residual_complex)^2 + imag(residual_complex)^2)]

            CSV.write("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) vcycle error.csv",
                                    df, delim = ';',append=true)
        end
    end
    return cat([resvec3[end]],resvec1, dims=1), cat([resvec3[end]],resvec2,dims=1)
end

function load_model!(test_name, e_vcycle_input, kappa_input, gamma_input;kernel=(3,3), model_type=SUnet, k_type=NaN, resnet_type=SResidualBlock, k_chs=-1, indexes=3, σ=elu, arch=0)
    model = create_model!(e_vcycle_input, kappa_input, gamma_input; kernel=kernel, type=model_type, k_type=k_type, resnet_type=resnet_type, k_chs=k_chs, indexes=indexes, σ=σ, arch=arch)
    model = model|>pu
    @load test_name model #"../../models/$(test_name).bson" model
    @info "$(Dates.format(now(), "HH:MM:SS.sss")) - Load Model"
    model = model|>cgpu

    return model
end

function model_tuning!(model, test_name, kappa, omega, gamma, n, m, f, dataset_size, bs, iter, lr, threshold, axb, jac, k_kernel, restrt, kappa_type; residual_loss=false)
    name = "$(test_name) $(n) $(m) $(dataset_size) $(iter)"
    model, train_loss, test_loss = train_residual_unet!(model, name, n, m, f, kappa, omega, gamma,
                                                        dataset_size, 10, bs, iter, lr;
                                                        e_vcycle_input=false, v2_iter=10, level=3, data_augmentetion=false,
                                                        kappa_type=kappa_type, threshold=threshold, kappa_input=true, kappa_smooth=true, k_kernel=k_kernel,
                                                        gamma_input=true, kernel=3, smaller_lr=24, axb=axb, jac=jac, norm_input=norm_input, model_type=model_type, k_type=k_type, k_chs=k_chs, indexes=indexes,
                                                        data_path="", full_loss=false, residual_loss=residual_loss, gmres_restrt=restrt,σ=σ)
    return model, name
end

function check_model!(test_name, model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=10, level=3, smooth=false, k_kernel=3, threshold=50, axb=false, norm_input=false, before_jacobi=false, log_error=false, unet_in_vcycle=false, indexes=3, arch=1) # arch - Standart (0) SplitUNet (1) FeaturesUNet (2)
    unet_results = zeros(dataset_size,6,restrt*max_iter+1)
    vcycle_results = zeros(dataset_size,4,restrt*max_iter+1)

    # CSV.write("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) unet error.csv",
    #                      DataFrame(Title=[],EA=[],RA=[]), delim = ';')
    # CSV.write("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) vcycle error.csv",
    #                      DataFrame(Title=[],EA=[],RA=[]), delim = ';')
    # CSV.write("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) unet error.csv",
    #                     DataFrame(Title=[],EB=[],RB=[],EA=[],RA=[]), delim = ';')
    # CSV.write("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) vcycle error.csv",
    #                     DataFrame(Title=[],E=[],R=[]), delim = ';')
    # CSV.write("training set.csv", DataFrame(RR=[], RI=[], KAPPA=[], ER=[], EI=[]), delim = ';',append=true)

    # gamma = r_type.(gamma)
    # kappa_input = u_type.(kappa_input)
    for i=1:dataset_size
        x_true = randn(c_type,n-1,m-1, blocks, 1)|>pu
        kappa = r_type.(generate_kappa!(n, m; type=kappa_type, smooth=smooth, threshold=threshold, kernel=k_kernel)|>pu)
        # heatmap(kappa, color=:blues)
        # savefig("test/unet/results/kappa $(kappa_type) $(threshold) $(k_kernel) $(i)")
        # CSV.write("test/unet/results/kappa $(kappa_type) $(threshold) $(k_kernel) $(i).csv", DataFrame(K=vec(kappa)), delim = ';')
        kappa_features = NaN
        if arch != 0
            kappa_input = reshape(kappa, n-1, m-1, 1, 1)
            if indexes != 3
                kappa_input = cat(kappa_input, reshape(gamma, n-1, m-1, 1, 1), dims=3)
            end
            kappa_features = model.kappa_subnet(kappa_input|>cgpu)|>pu
            # @info "$(Dates.format(now(), "HH:MM:SS.sss")) - after kappa features $(size(kappa_features))"
        end
        resvec1, resvec2, times1, times2 = unet_vs_vcycle_split!(model, n, m, kappa, kappa_features, omega, gamma, x_true, true, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, false; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input, log_error=log_error, test_name=test_name, before_jacobi=before_jacobi, unet_in_vcycle=unet_in_vcycle, arch=arch)
        unet_results[i,1,:] = resvec1
        unet_results[i,2,:] = times1
        vcycle_results[i,1,:] = resvec2
        vcycle_results[i,2,:] = times2
        resvec1, resvec2, times1, times2 = unet_vs_vcycle_split!(model, n, m, kappa, kappa_features, omega, gamma, x_true, false, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, false; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input, log_error=log_error, test_name=test_name, before_jacobi=before_jacobi, unet_in_vcycle=unet_in_vcycle, arch=arch)
        unet_results[i,3,:] = resvec1
        unet_results[i,4,:] = times1
        vcycle_results[i,3,:] = resvec2
        vcycle_results[i,4,:] = times2
        resvec1, resvec2, times1, times2 = unet_vs_vcycle_split!(model, n, m, kappa, kappa_features, omega, gamma, x_true, false, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, true; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input, log_error=log_error, test_name=test_name, before_jacobi=before_jacobi, unet_in_vcycle=unet_in_vcycle, arch=arch)
        unet_results[i,5,:] = resvec1
        unet_results[i,6,:] = times1
    end
    # graph_title = unet_vs_vcycle_graph!("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1])", #" t_j=$("$(before_jacobi)"[1])",
    #                 mean(unet_results[:,1,:],dims=1)', mean(unet_results[:,2,:],dims=1)', mean(unet_results[:,3,:],dims=1)', mean(unet_results[:,4,:],dims=1)',
    #                 mean(vcycle_results[:,3,:],dims=1)', mean(vcycle_results[:,4,:],dims=1)', mean(unet_results[:,5,:],dims=1)', mean(unet_results[:,6,:],dims=1)')

    CSV.write("test/unet/results/$(test_name) t_n=$(n) t_m=$(m).csv", # t_j=$("$(before_jacobi)"[1])
                                                        DataFrame(VU=vec(mean(unet_results[:,1,:],dims=1)'),
                                                        VU_T=vec(mean(unet_results[:,2,:],dims=1)'),
                                                        U=vec(mean(unet_results[:,3,:],dims=1)'),
                                                        U_T=vec(mean(unet_results[:,4,:],dims=1)'),
                                                        V=vec(mean(vcycle_results[:,3,:],dims=1)'),
                                                        V_T=vec(mean(vcycle_results[:,4,:],dims=1)'),
                                                        JU=vec(mean(unet_results[:,5,:],dims=1)'),
                                                        JU_T=vec(mean(unet_results[:,6,:],dims=1)')), delim = ';')
end

function check_model_original!(test_name, model, n, m, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, dataset_size, restrt, max_iter; v2_iter=10, level=3, smooth=false, k_kernel=3, threshold=50, axb=false, norm_input=false, before_jacobi=false, log_error=false, unet_in_vcycle=false, indexes=3, arch=1) # arch - Standart (0) SplitUNet (1) FeaturesUNet (2)
    unet_results = zeros(dataset_size,3,restrt*max_iter+1)
    vcycle_results = zeros(dataset_size,2,restrt*max_iter+1)

    for i=1:dataset_size
        x_true = randn(c_type,n-1,m-1, blocks, 1)|>pu
        kappa = r_type.(generate_kappa!(n, m; type=kappa_type, smooth=smooth, threshold=threshold, kernel=k_kernel)|>pu)
        heatmap(kappa, color=:blues)
        savefig("test/unet/results/kappa $(kappa_type) $(threshold) $(k_kernel) $(i)1")
        CSV.write("test/unet/results/kappa $(kappa_type) $(threshold) $(k_kernel) $(i)1.csv", DataFrame(K=vec(kappa)), delim = ';')
        kappa_features = NaN
        if arch != 0
            kappa_input = reshape(kappa, n-1, m-1, 1, 1)
            if indexes != 3
                kappa_input = cat(kappa_input, reshape(gamma, n-1, m-1, 1, 1), dims=3)
            end
            kappa_features = model.kappa_subnet(kappa_input|>cgpu)|>pu
        end
        resvec1, resvec2 = unet_vs_vcycle_blocks_split!(model, n, m, kappa, kappa_features, omega, gamma, x_true, true, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, false; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input, log_error=log_error, test_name=test_name, before_jacobi=before_jacobi, unet_in_vcycle=unet_in_vcycle, arch=arch)
        unet_results[i,1,:] = resvec1
        vcycle_results[i,1,:] = resvec2
        resvec1, resvec2 = unet_vs_vcycle_blocks_split!(model, n, m, kappa, kappa_features, omega, gamma, x_true, false, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, false; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input, log_error=log_error, test_name=test_name, before_jacobi=before_jacobi, unet_in_vcycle=unet_in_vcycle, arch=arch)
        unet_results[i,2,:] = resvec1
        vcycle_results[i,2,:] = resvec2
        resvec1, resvec2 = unet_vs_vcycle_blocks_split!(model, n, m, kappa, kappa_features, omega, gamma, x_true, false, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, true; v2_iter=v2_iter, level=level, axb=axb, norm_input=norm_input, log_error=log_error, test_name=test_name, before_jacobi=before_jacobi, unet_in_vcycle=unet_in_vcycle, arch=arch)
        unet_results[i,3,:] = resvec1
    end
    # graph_title = unet_vs_vcycle_graph!("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1])", #" t_j=$("$(before_jacobi)"[1])",
    #                 mean(unet_results[:,1,:],dims=1)', mean(unet_results[:,2,:],dims=1)', mean(unet_results[:,3,:],dims=1)', mean(unet_results[:,4,:],dims=1)',
    #                 mean(vcycle_results[:,3,:],dims=1)', mean(vcycle_results[:,4,:],dims=1)', mean(unet_results[:,5,:],dims=1)', mean(unet_results[:,6,:],dims=1)')
    graph_title = "$(Dates.format(now(), "HH_MM_SS_sss"))_$(test_name)_t_n=$(n)_t_m=$(m)"
    @info "$(Dates.format(now(), "HH:MM:SS.sss")) - $(graph_title)"
    CSV.write("test/unet/results/$(graph_title).csv", DataFrame(VU=vec(mean(unet_results[:,1,:],dims=1)'),
                                                        U=vec(mean(unet_results[:,2,:],dims=1)'),
                                                        V=vec(mean(vcycle_results[:,2,:],dims=1)'),
                                                        JU=vec(mean(unet_results[:,3,:],dims=1)')), delim = ';')
end

function check_full_solution_model!(test_name, model, n, f, kappa, omega, gamma, e_vcycle_input, kappa_type, kappa_input, gamma_input, kernel, m, restrt, max_iter; v2_iter=10, level=3, smooth=false, threshold=50, axb=false, norm_input=false)
    unet_results = zeros(m,restrt*max_iter-1)
    h = r_type(2.0 ./ (n+m))
    for i=1:m
        x_true = randn(c_type,n-1,m-1, 1, 1)|>pu
        kappa = generate_kappa!(n; type=kappa_type, smooth=smooth, threshold=threshold)
        s_matrix, h_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)

        if axb == true
            r = helmholtz_chain!(x_true, h_matrix; h=h)
        else
            r, _ = generate_r_vcycle!(n, m, kappa, omega, gamma, x_true)
        end

        e = zeros(c_type, n-1, m-1, 1, 1)
        ae = zeros(c_type, n-1, m-1, 1, 1)
        for j = 1:restrt*max_iter-1
            # e, = v_cycle_helmholtz!(n, m, h, reshape(e,n-1,m-1), reshape(r,n-1,m-1), kappa, omega, gamma; v2_iter = v2_iter, level = level)
            # e = reshape(e, n-1, m-1, 1, 1)
            # e = jacobi_helmholtz_method!(n, m, h, reshape(e,n-1,m-1), reshape(r,n-1,m-1), h_matrix)
            ae = helmholtz_chain!(e, h_matrix; h=h)|> pu
            correct_r = r - ae

            unet_results[i,j] = norm(correct_r) / norm(r)
            @info "$(i) $(j) : $(unet_results[i,j]) r=$(norm(r)) e=$(norm(e)) ae=$(norm(ae)) correct_r=$(norm(correct_r))"

            #
            correct_e = model(cat(complex_grid_to_channels!(reshape(correct_r,n-1,m-1,1,1).* (h^2), n), kappa, gamma, dims=3)|> cgpu)|> pu
            e = e + (reshape((correct_e[:,:,1,1] + im*correct_e[:,:,2,1]),n-1,m-1,1,1))
            @info "$(i) $(j) : e=$(norm(e)) correct_e=$(norm(correct_e))"
            # ae = helmholtz_chain!(e, h_matrix; h=h)|> pu
            # e = jacobi_helmholtz_method!(n, m, h, e, r, h_matrix)

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

    h = r_type(2.0 ./ (n+m))

    for i=1:m
        x_true = randn(c_type, n-1, m-1, 1, 1)|>pu
        kappa = generate_kappa!(n; type=kappa_type, smooth=smooth, threshold=threshold)
        s_matrix, h_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
        if axb == true
            b_true = helmholtz_chain!(x_true, h_matrix; h=h)
        else
            b_true, _ = generate_r_vcycle!(n, m, kappa, omega, gamma, x_true)
        end

        A(v) = vec(helmholtz_chain!(reshape(v, n-1, m-1, 1, 1), h_matrix; h=h))

        function M(r)
            r = reshape(r, n-1, m-1)
            e_vcycle = zeros(c_type, n-1, m-1)
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
            return vec(e_vcycle)
        end

        x_vcycle = zeros(c_type,n-1,m-1)
        for j = 1:max_iter
            x_vcycle,flag,err,iter,resvec = fgmres_func(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)

            r = b_true - helmholtz_chain!(reshape(x_vcycle,n-1,m-1,1,1), h_matrix; h=h) |> pu
            x_unet = (model(cat(complex_grid_to_channels!(r, n), kappa, gamma, dims=3)|> cgpu) .* (h^2))|> pu
            @info "VU $(i) $(j) x_unet $(norm(x_unet)) x_vcycle $(norm(x_vcycle)) r $(norm(r)) b_true $(norm(b_true))"
            x_vcycle = reshape(x_vcycle,n-1,m-1) + (x_unet[:,:,1,1] + im*x_unet[:,:,2,1])
            vu_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
        end
        x_vcycle = zeros(c_type,n-1,m-1)
        for j = 1:max_iter
            x_vcycle,flag,err,iter,resvec = fgmres_func(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)

            v_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
        end
        x_vcycle = zeros(c_type,n-1,m-1)
        for j = 1:max_iter
            x_vcycle,flag,err,iter,resvec = fgmres_func(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)
            x_vcycle, = v_cycle_helmholtz!(n, m, h, reshape(x_vcycle,n-1,m-1), reshape(b_true,n-1,m-1), kappa, omega, gamma; v2_iter = v2_iter, level = level)
            vv_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
        end
        x_vcycle = zeros(c_type,n-1,m-1)
        for j = 1:max_iter

            x_vcycle,flag,err,iter,resvec = fgmres_func(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)
            r = b_true - helmholtz_chain!(reshape(x_vcycle,n-1,m-1,1,1), h_matrix; h=h) |> pu
            x_unet = (model(cat(complex_grid_to_channels!(r, n), kappa, gamma, dims=3)|> cgpu) .* (h^2))|> pu
            @info "VUV $(i) $(j) x_unet $(norm(x_unet)) x_vcycle $(norm(x_vcycle)) r $(norm(r)) b_true $(norm(b_true))"
            x_vcycle = reshape(x_vcycle,n-1,m-1) + (x_unet[:,:,1,1] + im*x_unet[:,:,2,1])
            x_vcycle, = v_cycle_helmholtz!(n, m, h, x_vcycle, reshape(b_true,n-1,m-1), kappa, omega, gamma; v2_iter = v2_iter, level = level)
            vuv_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
        end
        x_vcycle = zeros(c_type,n-1,m-1)
        for j = 1:max_iter
            x_vcycle,flag,err,iter,resvec = fgmres_func(A, vec(b_true), 5, tol=1e-30, maxIter=1,
                                                            M=M, x=vec(x_vcycle), out=-1, flexible=true)
            r = b_true - helmholtz_chain!(reshape(x_vcycle,n-1,m-1,1,1), h_matrix; h=h) |> pu
            x_unet = (model(cat(complex_grid_to_channels!(r, n), kappa, gamma, dims=3)|> cgpu) .* (h^2))|> pu
            @info "VUJ $(i) $(j) x_unet $(norm(x_unet)) x_vcycle $(norm(x_vcycle)) r $(norm(r)) b_true $(norm(b_true))"
            x_vcycle = reshape(x_vcycle,n-1,m-1) + (x_unet[:,:,1,1] + im*x_unet[:,:,2,1])
            x_vcycle = jacobi_helmholtz_method!(n, m, h, x_vcycle, reshape(b_true,n-1,m-1), h_matrix)
            vuj_results[i,5*(j-1)+1:5*(j-1)+5] = resvec
        end
    end
    unet_outof_gmres_graph!("$(test_name) t_n=$(n) t_m=$(m) t_axb=$("$(axb)"[1]) t_norm=$("$(norm_input)"[1]) t_j=$("$(before_jacobi)"[1])",
                    mean(vu_results[:,:],dims=1)', mean(v_results[:,:],dims=1)', mean(vv_results[:,:],dims=1)', mean(vuv_results[:,:],dims=1)', mean(vuj_results[:,:],dims=1)')
end

function check_point_source_problem!(test_name, model, n, m, kappa, omega, gamma, e_vcycle_input, kappa_input, gamma_input; v2_iter=10, level=3)
    b = zeros(c_type,n-1,m-1)|>pu
    h = 2.0 / (n+m)
    b[floor(Int32,n / 2.0),floor(Int32,m / 2.0)] = 1.0 ./ (h.^2);
    full_sol = fgmres_v_cycle_helmholtz!(n, m, h, b, kappa|>pu, omega, gamma|>pu; restrt=5, maxIter=5)
    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=0.5)
    coefficient = h^2

    if e_vcycle_input == true
        x = zeros(c_type,n-1,m-1)|>pu
        x, = v_cycle_helmholtz!(n, m, h, x, coefficient .* b, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        input = cat(complex_grid_to_channels!(reshape(x, n-1, m-1, 1, 1)),
                            complex_grid_to_channels!(reshape(coefficient .* b, n-1, m-1, 1, 1)), dims=3)
    else
        input = complex_grid_to_channels!(reshape(coefficient .* b, n-1, m-1, 1, 1))
    end
    input = kappa_input == true ? cat(input, reshape(kappa, n-1, m-1, 1, 1), dims=3) : input
    input = gamma_input == true ? cat(input, gamma, dims=3) : input

    e_unet = model(input)
    heatmap(e_unet[:,:,1,1]|>cpu, color=:blues)
    savefig("test/unet/results/$(test_name) $(n) e unet")
    CSV.write("test/unet/results/$(test_name) $(n) e unet.csv", DataFrame(E=vec(real(e_unet[:,:,1,1]))), delim = ';')

    approx = zeros(c_type,n-1,m-1)|>pu
    approx, = v_cycle_helmholtz!(n, m, h, approx, b, kappa, omega, gamma; v2_iter=v2_iter, level=level)
    heatmap(real(approx), color=:blues)
    savefig("test/unet/results/$(test_name) $(n) e vcycle")
    CSV.write("test/unet/results/$(test_name) $(n) e vcycle.csv", DataFrame(E=vec(real(approx))), delim = ';')
    ax = helmholtz_chain!(reshape(x,n-1,m-1,1,1), helmholtz_matrix; h=h)
    @info "Vcycle error=$(norm_diff!(x,full_sol)) residual=$(norm_diff!(ax,b))"

    approx, = v_cycle_helmholtz!(n, m, h, approx, b, kappa, omega, gamma; v2_iter=v2_iter, level=level)
    heatmap(real(approx), color=:blues)
    savefig("test/unet/results/$(test_name) $(n) e vcycle vcycle")
    CSV.write("test/unet/results/$(test_name) $(n) e vcycle vcycle.csv", DataFrame(E=vec(real(x))), delim = ';')
    ax = helmholtz_chain!(reshape(x,n-1,m-1,1,1), helmholtz_matrix; h=h)
    @info "Vcycle Vcycle error=$(norm_diff!(x,full_sol)) residual=$(norm_diff!(ax,b))"

    approx = e_unet[:,:,1,1]+im*e_unet[:,:,2,1]
    ax = helmholtz_chain!(reshape(x|>pu,n-1,m-1,1,1), helmholtz_matrix; h=h)
    @info "Unet error=$(norm_diff!(x|>pu,full_sol)) residual=$(norm_diff!(ax,b))"
    approx, = v_cycle_helmholtz!(n, m, h, approx|>pu, b, kappa, omega, gamma; v2_iter=v2_iter, level=level)
    heatmap(real(approx), color=:blues)
    savefig("test/unet/results/$(test_name) $(n) e vcycle unet")
    CSV.write("test/unet/results/$(test_name) $(n) e vcycle unet.csv", DataFrame(E=vec(real(x))), delim = ';')
    ax = helmholtz_chain!(reshape(x,n-1,m-1,1,1), helmholtz_matrix; h=h)
    @info "Vcycle Unet error=$(norm_diff!(x,full_sol)) residual=$(norm_diff!(ax,b))"

    approx = (e_unet[:,:,1,1]+im*e_unet[:,:,2,1])
    approx = jacobi_helmholtz_method!(n, m, h, approx|>pu, b, helmholtz_matrix)
    heatmap(real(approx), color=:blues)
    savefig("test/unet/results/$(test_name) $(n) e jacobi unet")
    CSV.write("test/unet/results/$(test_name) $(n) e jacobi unet.csv", DataFrame(E=vec(real(approx))), delim = ';')
    ax = helmholtz_chain!(reshape(x,n-1,m-1,1,1), helmholtz_matrix; h=h)
    @info "Jacobi Unet error=$(norm_diff!(x,full_sol)) residual=$(norm_diff!(ax,b))"
end


function unet_vs_vcycle_blocks_split!(model, n, m, kappa, kappa_features, omega, gamma, x_true, after_vcycle, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, relax_jacobi; v2_iter=10, level=3, axb=false, norm_input=false, log_error=true, test_name="", before_jacobi=false, unet_in_vcycle=false, arch=1)

    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    h = 2.0./(m+n)

    if axb == true
        r_vcycle = zeros(c_type,n-1,m-1,1,1)
        r_vcycle[floor(Int32,n / 2.0),floor(Int32,m / 2.0),1,1] = r_type(1.0 ./mean(h.^2))
    else
        x_true = randn(c_type,n-1,m-1, 1, 1)
        r_vcycle, _ = generate_r_vcycle!(n, m, kappa, omega, gamma, reshape(x_true,n-1,m-1,1,1))
        r_vcycle = vec(r_vcycle)
        for i = 2:blocks
            x_true = randn(c_type,n-1,m-1, 1, 1)
            r_vcycle1, _ = generate_r_vcycle!(n, m, kappa, omega, gamma, reshape(x_true,n-1,m-1,1,1))
            r_vcycle = cat(r_vcycle, vec(r_vcycle1), dims=2)
        end
    end

    coefficient = r_type(h^2)

    A(v) = vec(helmholtz_chain!(reshape(v, n-1, m-1, 1, 1), helmholtz_matrix; h=h))
    function As(v)
        res = vec(A(v[:,1]))
        for i = 2:blocks
            res = cat(res, vec(A(v[:,i])), dims=2)
        end

        return res
    end

    function M_Unet(r)
        r = reshape(r, n-1, m-1)
        rj = reshape(r, n-1, m-1)
        e = zeros(c_type, n-1, m-1)
        ej = zeros(c_type, n-1, m-1)

        if after_vcycle != true
            ej = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
            rj = r - reshape(A(ej), n-1, m-1)
        end

        input = complex_grid_to_channels!(reshape(rj , n-1, m-1, 1, 1))
        if arch == 1
            input = cat(input, reshape(kappa, n-1, m-1, 1, 1), reshape(gamma, n-1, m-1, 1, 1), kappa_features, dims=3)
            e_unet = (model.solve_subnet(u_type.(input)|>cgpu)|>cpu)
        elseif arch == 2
            input = cat(input, reshape(kappa, n-1, m-1, 1, 1), reshape(gamma, n-1, m-1, 1, 1), dims=3)
            e_unet = (model.solve_subnet(u_type.(input)|>cgpu, kappa_features|>cgpu)|>cpu)
        else
            input = cat(input, reshape(kappa, n-1, m-1, 1, 1), reshape(gamma, n-1, m-1, 1, 1), dims=3)
            e_unet = (model(u_type.(input)|>cgpu)|>cpu)
        end

        e_unet = reshape((e_unet[:,:,1,1] + im*e_unet[:,:,2,1]),n-1,m-1,1,1) .* coefficient
        e = reshape(e_unet, n-1,m-1)

        e = ej + e

        if relax_jacobi == true
            e = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
        end

        if after_vcycle == true
            e, = v_cycle_helmholtz!(n, m, h, e, r, kappa, omega, gamma; v2_iter = v2_iter, level = level)
        end
        return vec(e)
    end

    function M_Unets(rs)
        res = M_Unet(rs[:,1])
        for i = 2:blocks
            res = cat(res, M_Unet(rs[:,i]), dims=2)
        end

        return res
    end

    function M(r)
        e_vcycle = zeros(c_type,n-1,m-1)
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,1], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,1], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        end
        res = vec(e_vcycle)
        for i = 2:blocks
            e_vcycle = zeros(c_type,n-1,m-1)
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,i], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
            if after_vcycle == true
                e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,i], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
            end
            res = cat(res, vec(e_vcycle), dims=2)
        end

        return res
    end

    function SM(r)
        e_vcycle = zeros(c_type,n-1,m-1)
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,1], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
        res = vec(e_vcycle)
        for i = 2:blocks
            e_vcycle = zeros(c_type,n-1,m-1)
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[:,i], n-1, m-1), kappa, omega, gamma; v2_iter = v2_iter, level = 3)
            res = cat(res, vec(e_vcycle), dims=2)
        end

        return res
    end

    x_init = zeros(c_type,(n-1)*(m-1),blocks)
    x3,flag3,err3,iter3,resvec3= KrylovMethods.blockFGMRES(As, r_vcycle, 3, tol=1e-30, maxIter=1,
                                                    M=SM, X=x_init, out=-1,flexible=true)
    i = 1
    x1 = x3
    x1,flag1,err1,iter1,resvec1 = KrylovMethods.blockFGMRES(As, r_vcycle, restrt, tol=1e-30, maxIter=max_iter,
                                                            M=M_Unets, X =x1, out=-1,flexible=true)

    x_init = zeros(c_type,(n-1)*(m-1),blocks)
    x3,flag3,err3,iter3,resvec3= KrylovMethods.blockFGMRES(As, r_vcycle, 3, tol=1e-30, maxIter=1,
                                                    M=SM, X=x_init, out=-1,flexible=true)
    i = 1
    x2 = x3
    x2,flag2,err2,iter2,resvec2 = KrylovMethods.blockFGMRES(As, r_vcycle, restrt, tol=1e-30, maxIter=max_iter,
                                                    M=M, X=x2, out=-1,flexible=true)

    return cat([resvec3[end]],resvec1, dims=1), cat([resvec3[end]],resvec2,dims=1)
end

function unet_vs_vcycle_split!(model, n, m, kappa, kappa_features, omega, gamma, x_true, after_vcycle, e_vcycle_input, kappa_input, gamma_input, restrt, max_iter, relax_jacobi; v2_iter=10, level=3, axb=false, norm_input=false, log_error=true, test_name="", before_jacobi=false, unet_in_vcycle=false, arch=1)


    sl_matrix_level3, h_matrix_level3 = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    kappa_coarse = down(reshape(kappa, n-1, m-1, 1, 1)|>pu)[:,:,1,1]
    gamma_coarse = down(reshape(gamma, n-1, m-1, 1, 1)|>pu)[:,:,1,1]
    sl_matrix_level2, h_matrix_level2 = get_helmholtz_matrices!(kappa_coarse, omega, gamma_coarse; alpha=r_type(0.5))
    kappa_coarse = down(reshape(kappa_coarse, Int64((n/2)-1),  Int64((m/2)-1), 1, 1)|>pu)[:,:,1,1]
    gamma_coarse = down(reshape(gamma_coarse,  Int64((n/2)-1),  Int64((m/2)-1), 1, 1)|>pu)[:,:,1,1]
    sl_matrix_level1, h_matrix_level1 = get_helmholtz_matrices!(kappa_coarse, omega, gamma_coarse; alpha=r_type(0.5))

    e_zeros = zeros(c_type, n-1, m-1)|>pu
    shifted_laplacian_matrix, helmholtz_matrix = get_helmholtz_matrices!(kappa, omega, gamma; alpha=r_type(0.5))
    h = r_type(2.0./(n+m))

    if axb == true
        src = [64,64]

        r_vcycle = zeros(c_type,n-1,m-1,1,1)|>pu
        r_vcycle[floor(Int32,n / 2.0),floor(Int32,m / 2.0),1,1] = r_type(1.0 ./mean(h.^2))
    else
        x_true = randn(c_type,n-1,m-1, 1, 1)|>pu
        r_vcycle, _ = generate_r_vcycle!(n, m, kappa, omega, gamma, reshape(x_true,n-1,m-1,1,1))
        r_vcycle = vec(r_vcycle)
        for i = 2:blocks
            x_true = randn(c_type,n-1,m-1, 1, 1)|>pu
            r_vcycle1, _ = generate_r_vcycle!(n, m, kappa, omega, gamma, reshape(x_true,n-1,m-1,1,1))
            r_vcycle = cat(r_vcycle, vec(r_vcycle1), dims=2)
        end
    end

    coefficient = r_type(h^2)

    A(v::a_type) = vec(helmholtz_chain!(reshape(real(v), n-1, m-1, 1, 1), helmholtz_matrix; h=h) + im*helmholtz_chain!(reshape(imag(v), n-1, m-1, 1, 1), helmholtz_matrix; h=h))

    function As(v)
        res = vec(A(v[1:(n-1)*(m-1)]))
        for i = 2:blocks
            res = cat(res, vec(A(v[(i-1)*((n-1)*(m-1))+1:i*((n-1)*(m-1))])), dims=2)
        end

        return vec(res)
    end


    function M_Unet(r::a_type)
        r = reshape(r, n-1, m-1)
        rj = reshape(r, n-1, m-1)
        e = ej = e_zeros

        if after_vcycle != true
            ej = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
            rj = r - reshape(A(ej), n-1, m-1)
        end
        input = complex_grid_to_channels!(reshape(rj , n-1, m-1, 1, 1))
        if arch == 1
            input = cat(input, reshape(kappa, n-1, m-1, 1, 1), reshape(gamma, n-1, m-1, 1, 1), kappa_features, dims=3)
            e_unet = (model.solve_subnet(u_type.(input)|>cgpu)|>pu)
        elseif arch == 2
            input = cat(input, reshape(kappa, n-1, m-1, 1, 1), reshape(gamma, n-1, m-1, 1, 1), dims=3)
            e_unet = r_type.((model.solve_subnet(u_type.(input), kappa_features)))|>pu
        else
            input = cat(input, reshape(kappa, n-1, m-1, 1, 1), reshape(gamma, n-1, m-1, 1, 1), dims=3)
            e_unet = model(input)
        end

        e_unet = reshape((e_unet[:,:,1,1] + im*e_unet[:,:,2,1]),n-1,m-1,1,1) .* coefficient
        e = reshape(e_unet, n-1,m-1)

        e = ej + e

        if relax_jacobi == true
            e = jacobi_helmholtz_method!(n, m, h, e, r, helmholtz_matrix)
        end

        if after_vcycle == true
            e, = v_cycle_helmholtz!(n, m, h, e, r, h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3; v2_iter = v2_iter, level = level)
        end
        return vec(e)
    end

    function M_Unets(rs)
        res = M_Unet(rs[1:((n-1)*(m-1))])
        for i = 2:blocks
            res = cat(res, M_Unet(rs[(i-1)*((n-1)*(m-1))+1:i*((n-1)*(m-1))]), dims=2)
        end

        return vec(res)
    end

    function M(r::a_type)
        e_vcycle = e_zeros
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[1:((n-1)*(m-1))], n-1, m-1), h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3; v2_iter = v2_iter, level = 3)
        if after_vcycle == true
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[1:((n-1)*(m-1))], n-1, m-1), h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3; v2_iter = v2_iter, level = 3)
        end
        res = vec(e_vcycle)
        for i = 2:blocks
            e_vcycle = e_zeros
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[(i-1)*((n-1)*(m-1))+1:i*((n-1)*(m-1))], n-1, m-1), h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3; v2_iter = v2_iter, level = 3)
            if after_vcycle == true
                e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[(i-1)*((n-1)*(m-1))+1:i*((n-1)*(m-1))], n-1, m-1), h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3; v2_iter = v2_iter, level = 3)
            end
            res = cat(res, vec(e_vcycle), dims=2)
        end

        return vec(res)
    end

    function SM(r::a_type)
        e_vcycle = zeros(c_type,n-1,m-1)|>pu
        e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[1:((n-1)*(m-1))], n-1, m-1), h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3; v2_iter = v2_iter, level = 3)
        res = vec(e_vcycle)
        for i = 2:blocks
            e_vcycle = zeros(c_type,n-1,m-1)|>pu
            e_vcycle, = v_cycle_helmholtz!(n, m, h, e_vcycle, reshape(r[(i-1)*((n-1)*(m-1))+1:i*((n-1)*(m-1))], n-1, m-1), h_matrix_level1, sl_matrix_level1, h_matrix_level2, sl_matrix_level2, h_matrix_level3, sl_matrix_level3; v2_iter = v2_iter, level = 3)
            res = cat(res, vec(e_vcycle), dims=2)
        end

        return vec(res)
    end

    x_init = zeros(c_type,(n-1)*(m-1),blocks)|>pu
    x3,flag3,err3,iter3,resvec3,times3= fgmres_func(As, vec(r_vcycle), 3, tol=1e-30, maxIter=1,
                                                    M=SM, x=vec(x_init), out=-1,flexible=true)
    i = 1
    x1 = x3
    x1,flag1,err1,iter1,resvec1,times1 = fgmres_func(As, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                            M=M_Unets, x=vec(x1), out=-1,flexible=true)

    x_init = zeros(c_type,(n-1)*(m-1),blocks)|>pu
    x3,flag3,err3,iter3,resvec3,times3= fgmres_func(As, vec(r_vcycle), 3, tol=1e-30, maxIter=1,
                                                    M=SM, x=vec(x_init), out=-1,flexible=true)
    i = 1
    x2 = x3
    x2,flag2,err2,iter2,resvec2,times2 = fgmres_func(As, vec(r_vcycle), restrt, tol=1e-30, maxIter=max_iter,
                                                    M=M, x=vec(x2), out=-1,flexible=true)

    return cat([resvec3[end]],resvec1, dims=1), cat([resvec3[end]],resvec2,dims=1), cat([0],times1,dims=1), cat([0],times2,dims=1)
end
