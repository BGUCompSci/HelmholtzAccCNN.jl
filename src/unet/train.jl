using BSON: @save

# Loss types:

function norm_diff!(x,y)
    return sqrt(sum((x - y).^2) / sum(y.^2))
end

function error_loss!(model, input, output)
    model_result = model(input)
    return norm_diff!(model_result, output|>cgpu)
end

function get_matrices!(alpha, kappa, omega, gamma)
    inv = reshape([u_type(-1.0)],1,1,1,1)|> cgpu

    s_h_real = (kappa .* kappa .* omega .* omega)
    h_imag = kappa .* kappa .* omega .* gamma
    s_imag = (kappa .* kappa .* omega .* omega .* alpha) .+ (kappa .* kappa .* omega .* gamma)

    h_matrix = cat(cat(s_h_real, h_imag, dims=3),cat(inv .* h_imag, s_h_real, dims=3),dims=4) |> cgpu
    s_matrix = cat(cat(s_h_real, s_imag, dims=3),cat(inv .* s_imag, s_h_real, dims=3),dims=4) |> cgpu

    return h_matrix, s_matrix
end

function residual_loss!(model, n, m, f, input, output)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    h = u_type(2.0 ./ (n+m))
    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)

    r_unet = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu
    return norm_diff!(r_unet, r)
end

function error_residual_loss!(model, n, m, f, input, output)
    e0 = model(input)

    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    h = u_type(2.0 ./ (n+m))
    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu
    r_unet = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu

    e_loss = norm_diff!(e0, output|>cgpu)
    r_loss = norm_diff!(r_unet, r)

    return e_loss + 0.1 * r_loss
end

function error_residual_loss_details!(model, n, m, f, input, output)
    e0 = model(input)

    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    h = u_type(1.0 ./ n)
    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu
    r_unet = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu

    e_loss = norm_diff!(e0, output|>cgpu)
    r_loss = norm_diff!(r_unet, r)

    return [e_loss + 0.1 * r_loss, r_loss, e_loss]
end

function append_input!(tuple, extension)
    return (cat(tuple[1], extension, dims=3), tuple[2])
end

function convert_input!(tuple)
    return (u_type.(tuple[1]), u_type.(tuple[2]))
end

function batch_loss!(set, loss; errors_count=1, gamma_input=false, append_gamma=identity)
    set_size = size(set,1)
    batch_size = min(1000,set_size)
    batchs = floor(Int64,set_size/batch_size)
    loss_sum = zeros(errors_count)
    for batch_idx in 1:batchs
        batch_set = set[(batch_idx-1)*batch_size+1:batch_idx*batch_size]
        if gamma_input == true
            batch_set = append_gamma.(batch_set)
        end
        current_loss = loss.(batch_set|>cgpu)
        loss_sum = loss_sum .+ sum(hcat(current_loss...),dims=2)
    end
    return (loss_sum ./ set_size)
end


function jacobi_channels!(n, x, b, matrix; w=0.8)
    h = 2.0 ./ (n+m)

    y = helmholtz_chain_channels!(x, matrix; h=h)|> cgpu
        residual = b - y
    d = u_type(4.0 / h^2) .- sum(matrix, dims=4)
    alpha = u_type(w) ./ d
    step = alpha .* residual

    return x .+ step
end

function full_solution_loss!(model, input, output, n, m, f)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    e = output
    h = u_type(2.0 ./ (n+m))

    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)

    ae1 = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu

    return norm_diff!(e0, e|> cgpu) + norm_diff!(ae1, r)
end

function full_solution_loss_details!(model, input, output, n, m, f)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    e = output
    h = u_type(2.0 ./ (n+m))

    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)

    ae1 = (h^2) .* helmholtz_chain_channels!(e0, h_matrix; h=h)|> cgpu

    return [norm_diff!(e0, e|> cgpu) + norm_diff!(ae1, r), norm_diff!(e0, e|> cgpu), norm_diff!(ae1, r)]
end

function full_solution_loss1!(model, input, output, n, m, f)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    e = output
    h = u_type(2.0 ./ (n+m))

    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)

    e1 = jacobi_channels!(n, e0, r ./ (h^2), h_matrix)|> cgpu
    ae1 = (h^2) .* helmholtz_chain_channels!(e1, h_matrix; h=h)|> cgpu
    r1 = r - ae1

    e2 = model(cat(r1,kappa,gamma,dims=3)|> cgpu)
    e3 = e1 + e2

    e4 = jacobi_channels!(n, e3, r ./ (h^2), h_matrix)
    # ae4 = (h^2) .* helmholtz_chain_channels!(e4, h_matrix; h=h)|> cgpu
    # r2 = r - ae4

    # e5 = model(cat(r2,kappa,gamma,dims=3)|> cgpu)
    # e6 = e4 + e5

    # e7 = jacobi_channels!(n, e6, r ./ (h^2), h_matrix)

    return norm_diff!(e0, e|> cgpu) + norm_diff!(e3, e|> cgpu) + norm_diff!(e4, e|> cgpu)
    #return norm_diff!(e0, e|> cgpu) + norm_diff!(e1, e|> cgpu) + norm_diff!(e3, e|> cgpu) + norm_diff!(e4, e|> cgpu) + norm_diff!(e6, e|> cgpu) + norm_diff!(e7, e|> cgpu)
end

function full_solution_loss_details1!(model, input, output, n, m, f)
    r = reshape(input[:,:,1:2,1],n-1,m-1,2,1)|> cgpu
    kappa = reshape(input[:,:,3,1],n-1,m-1,1,1)|> cgpu
    gamma = reshape(input[:,:,4,1],n-1,m-1,1,1)|> cgpu
    omega = reshape([u_type(2.0*pi*f)],1,1,1,1)|> cgpu
    alpha = reshape([u_type(0.5)],1,1,1,1)|> cgpu
    e = output
    h = u_type(2.0 ./ (n+m))

    h_matrix, s_matrix = get_matrices!(alpha, kappa, omega, gamma)|> cgpu

    e0 = model(input)
    # @info "e0 $(size(e0)) $(typeof(e0)) h_matrix $(size(h_matrix)) $(typeof(h_matrix))"
    e1 = jacobi_channels!(n, e0, r ./ (h^2), h_matrix)|> cgpu
    ae1 = (h^2) .* helmholtz_chain_channels!(e1, h_matrix; h=h)|> cgpu
    r1 = r - ae1

    e2 = model(cat(r1,kappa,gamma,dims=3)|> cgpu)
    e3 = e1 + e2

    e4 = jacobi_channels!(n, e3, r ./ (h^2), h_matrix)
    # ae4 = (h^2) .* helmholtz_chain_channels!(e4, h_matrix; h=h)|> cgpu
    # r2 = r - ae4

    # e5 = model(cat(r2,kappa,gamma,dims=3)|> cgpu)
    # e6 = e4 + e5

    # e7 = jacobi_channels!(n, e6, r ./ (h^2), h_matrix)

    # if norm(r) > 70 && norm(r) < 70.005 # mod(print_index,1000) == 0
    #     @info "e $(norm(e)) r $(norm(r)) e0 $(norm(e0)) e1 $(norm(e1)) ae1 $(norm(ae1)) r1 $(norm(r1)) e2 $(norm(e2)) e3 $(norm(e3)) e4 $(norm(e4)) ae4 $(norm(ae4)) r2 $(norm(r2)) e5 $(norm(e5)) e6 $(norm(e6)) e7 $(norm(e7))"
    #     @info "e $(norm(e)) r $(norm(r)) e0 $(norm(e0)) e1 $(norm(e1)) ae1 $(norm(ae1)) r1 $(norm(r1)) e2 $(norm(e2)) e3 $(norm(e3)) e4 $(norm(e4)) ae4 $(norm(ae4)) r2 $(norm(r2)) e5 $(norm(e5)) e6 $(norm(e6)) e7 $(norm(e7))"
    #
    # end
    # print_index = print_index+1
    return [norm_diff!(e0, e|> cgpu) + norm_diff!(e3, e|> cgpu) + norm_diff!(e4, e|> cgpu),
            norm_diff!(e0, e|> cgpu), norm_diff!(e3, e|> cgpu), norm_diff!(e4, e|> cgpu)] #  + norm_diff!(e4, e|> cgpu) + norm_diff!(e6, e|> cgpu) + norm_diff!(e7, e|> cgpu)
end

function train_residual_unet!(model, test_name, n, m, f, kappa, omega, gamma,
                            train_size, test_size, batch_size, iterations, init_lr;
                            e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=true, kappa_type=1, threshold=50,
                            kappa_input=true, kappa_smooth=false, k_kernel=3, gamma_input=false, kernel=(3,3), smaller_lr=10, axb=false, jac=false, norm_input=false,
                            model_type=SUnet, k_type=NaN, k_chs=-1, indexes=3, data_path="", full_loss=false, residual_loss=false, error_details=false, gmres_restrt=1, σ=elu) #, model=NaN)

    @info "$(Dates.format(now(), "HH:MM:SS")) - Start Train $(test_name)"

    if data_path != ""
        train_set = get_csv_set!(data_path, n, train_size)
        test_set = get_csv_set!(data_path, n, test_size)
    else
        train_set = generate_random_data!(train_size, n, m, kappa, omega, gamma;
                                                e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion =data_augmentetion,
                                                kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, k_kernel=k_kernel, axb=axb, jac=jac, norm_input=norm_input, gmres_restrt=gmres_restrt)
        test_set = generate_random_data!(test_size, n, m, kappa, omega, gamma;
                                                e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level,
                                                kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, k_kernel=k_kernel, axb=axb, jac=jac, norm_input=norm_input, gmres_restrt=gmres_restrt)
    end
    @info "$(Dates.format(now(), "HH:MM:SS")) - Generated Data"
    mkpath("models")

    # if model == NaN
    #     model = create_model!(e_vcycle_input, kappa_input, gamma_input; kernel=kernel, type=model_type, k_type=k_type, k_chs=k_chs, indexes=indexes, σ=σ)|>cgpu
    # end
    #
    batchs = floor(Int64,train_size / batch_size) # (batch_size*10))
    test_loss = zeros(iterations)
    train_loss = zeros(iterations)
    CSV.write("$(test_name) loss.csv", DataFrame(Train=[], Test=[]), delim = ';')
    errors_count = 4
    if errors_count > 1 && full_loss == true
        # CSV.write("$(test_name) loss.csv", DataFrame(Train=[],Train_U1=[],Train_J1=[],Train_U2=[],Train_J2=[],Train_U3=[],Train_J3=[],Test=[]), delim = ';') # ,Test_U1=[],Test_J1=[],Test_U2=[],Test_J2=[])
        # CSV.write("$(test_name) loss.csv", DataFrame(Train=[],Train_E=[],Train_R=[],Test=[],Test_E=[],Test_R=[]), delim = ';') # ,Test_U1=[],Test_J1=[],Test_U2=[],Test_J2=[])
        CSV.write("test/unet/results/$(test_name) loss.csv", DataFrame(Train=[],Train_U1=[],Train_U2=[],Train_J2=[],Test=[]), delim = ';')

    end
    if residual_loss == true
        CSV.write("$(test_name) loss.csv", DataFrame(Train=[], Residual=[], Error=[], Test=[]), delim = ';')
    end
    loss!(x, y) = error_loss!(model, x, y)
    full_loss!(x, y) = full_solution_loss1!(model, x, y, n, m, f)
    r_loss!(x ,y) = error_residual_loss!(model, n, m, f, x, y)
    loss!(tuple) = loss!(tuple[1], tuple[2])
    full_loss_details!(tuple) = full_solution_loss_details1!(model, tuple[1], tuple[2], n, m, f)
    r_loss_details!(tuple) = error_residual_loss_details!(model, n, m, f, tuple[1], tuple[2])

    # Start model training
    append_gamma!(tuple) = append_input!(tuple,gamma)
    lr = init_lr
    opt = RADAM(lr)
    for iteration in 1:iterations
        if mod(iteration,smaller_lr) == 0
            lr = lr / 10
            opt = RADAM(lr)
            batch_size = min(batch_size * 2,512)
            batchs = floor(Int64,train_size / min((batch_size),train_size)) #*10
            smaller_lr = ceil(Int64,smaller_lr / 2)
            @info "$(Dates.format(now(), "HH:MM:SS")) - Update Learning Rate $(lr) Batch Size $(batch_size)"
        end
        idxs = randperm(train_size)
        for batch_idx in 1:batchs
            batch_set = train_set[idxs[(batch_idx-1)*batch_size+1:batch_idx*batch_size]]
            if gamma_input == true
                batch_set = append_gamma!.(batch_set)
            end
            batch_set = convert_input!.(batch_set) |>cgpu
            if full_loss == true
                Flux.train!(full_loss!, Flux.params(model), batch_set, RADAM(lr))
            elseif residual_loss == true
                Flux.train!(r_loss!, Flux.params(model), batch_set, RADAM(lr))
            else
                Flux.train!(loss!, Flux.params(model), batch_set, RADAM(lr))
            end
        end

        if full_loss == true
            test_res = batch_loss!(test_set, full_loss_details!;errors_count=errors_count,gamma_input=gamma_input,append_gamma=append_gamma!)
            train_res = batch_loss!(train_set, full_loss_details!;errors_count=errors_count,gamma_input=gamma_input,append_gamma=append_gamma!)
            test_loss[iteration] = test_res[1]
            train_loss[iteration] = train_res[1]
            CSV.write("$(test_name) loss.csv", DataFrame(Train=[train_res[1]],Train_U1=[train_res[2]],Train_U2=[train_res[3]],Train_J2=[train_res[4]],Test=[test_res[1]]), delim = ';', append=true)

            # CSV.write("$(test_name) loss.csv", DataFrame(Train=[train_res[1]],Train_U1=[train_res[2]],Train_J1=[train_res[3]],Train_U2=[train_res[4]],Train_J2=[train_res[5]],Train_U3=[train_res[6]],Train_J3=[train_res[7]],Test=[test_res[1]]), delim = ';', append=true)
            #CSV.write("$(test_name) loss.csv", DataFrame(Train=[train_res[1]],Train_E=[train_res[2]],Train_R=[train_res[3]],Test=[test_res[1]],Test_E=[test_res[2]],Test_R=[test_res[3]]), delim = ';', append=true)

        elseif residual_loss == true
            test_res = batch_loss!(test_set, r_loss_details!;gamma_input=gamma_input,append_gamma=append_gamma!)
            train_res = batch_loss!(train_set, r_loss_details!;gamma_input=gamma_input,append_gamma=append_gamma!)
            test_loss[iteration] = test_res[1]
            train_loss[iteration] = train_res[1]
            CSV.write("$(test_name) loss.csv", DataFrame(Train=[train_res[1]], Residual=[train_res[2]], Error=[train_res[3]], Test=[test_res[1]]), delim = ';',append=true)
        else
            test_loss[iteration] = batch_loss!(test_set, loss!;gamma_input=gamma_input,append_gamma=append_gamma!)[1]
            train_loss[iteration] = batch_loss!(train_set, loss!;gamma_input=gamma_input,append_gamma=append_gamma!)[1]
            CSV.write("$(test_name) loss.csv", DataFrame(Train=[train_loss[iteration]], Test=[test_loss[iteration]]), delim = ';',append=true)
        end

        @info "$(Dates.format(now(), "HH:MM:SS")) - $(iteration)) Train loss value = $(train_loss[iteration]) , Test loss value = $(test_loss[iteration])"

        if mod(iteration,30) == 0
            model = model|>cpu
            @save "models/$(test_name).bson" model
            @info "$(Dates.format(now(), "HH:MM:SS")) - Save Model $(test_name).bson"

            model = model|>cgpu
        end
    end

    model = model|>cpu
    @save "models/$(test_name).bson" model
    @info "$(Dates.format(now(), "HH:MM:SS")) - Save Model $(test_name).bson"

    model = model|>cgpu
    return model, train_loss, test_loss
end
