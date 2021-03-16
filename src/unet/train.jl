using BSON: @save

function norm_diff!(x,y)
    return sqrt(sum((x - y).^2) / sum(y.^2))
end

function error_loss!(model, input, output)
    model_result = model(input)
    return norm_diff!(model_result, output|>cgpu)
end

function error_residual_loss!(model, helmholtz_channels, n, input, output)
    model_result = model(input)
    e_loss = norm_diff!(model_result, output)
    r_unet = helmholtz_chain_channels!(model_result, helmholtz_channels, n)|>cgpu
    r_loss = norm_diff!(r_unet, input[:,:,end-1:end,:])
    return e_loss, r_loss
end

function append_input!(tuple, extension)
    return (cat(tuple[1], extension, dims=3), tuple[2])
end

function batch_loss!(set, loss; gamma_input=false, append_gamma=identity)
    set_size = size(set,1)
    batch_size = min(1000,set_size)
    batchs = floor(Int64,set_size/batch_size)
    loss_sum = 0.0
    for batch_idx in 1:batchs
        batch_set = set[(batch_idx-1)*batch_size+1:batch_idx*batch_size]
        if gamma_input == true
            batch_set = append_gamma.(batch_set)
        end
        current_loss = loss.(batch_set|>cgpu)
        loss_sum += sum(current_loss)
    end
    return (loss_sum / set_size)
end

function train_residual_unet!(test_name, n, kappa, omega, gamma,
                            train_size, test_size, batch_size, iterations, init_lr;
                            e_vcycle_input=true, v2_iter=10, level=3, data_augmentetion=true, kappa_type=1, threshold=50,
                            kappa_input=true, kappa_smooth=false, gamma_input=false, kernel=(3,3), smaller_lr=10, axb=false, norm_input=false, model_type=SUnet)

    @info "$(Dates.format(now(), "HH:MM:SS")) - Start Train $(test_name)"
    train_set = generate_random_data!(train_size, n, kappa, omega, gamma;
                                            e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level, data_augmentetion =data_augmentetion,
                                            kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, axb=axb, norm_input=norm_input)
    test_set = generate_random_data!(test_size, n, kappa, omega, gamma;
                                            e_vcycle_input=e_vcycle_input, v2_iter=v2_iter, level=level,
                                            kappa_type=kappa_type, threshold=threshold, kappa_input=kappa_input, kappa_smooth=kappa_smooth, axb=axb, norm_input=norm_input)
    @info "$(Dates.format(now(), "HH:MM:SS")) - Generated Data"

    model = create_model!(e_vcycle_input, kappa_input, gamma_input; kernel=kernel, type=model_type)|>cgpu

    batchs = floor(Int64,train_size / (batch_size * 10))
    test_loss = zeros(iterations)
    train_loss = zeros(iterations)
    CSV.write("$(test_name) loss.csv", DataFrame(Train=[], Test=[]), delim = ';')

    loss!(x, y) = error_loss!(model, x, y)
    function loss!(tuple)
        return error_loss!(model, tuple[1], tuple[2])
    end

    # Start model training
    gamma_channels = complex_grid_to_channels!(gamma, n)
    append_gamma!(tuple) = append_input!(tuple,gamma_channels)
    lr = init_lr
    opt = ADAM(lr)
    for iteration in 1:iterations
        if mod(iteration,smaller_lr) == 0
            lr = lr / 2
            opt = ADAM(lr)
            @info "$(Dates.format(now(), "HH:MM:SS")) - Update Learning Rate $(lr)"
        end
        idxs = randperm(train_size)
        for batch_idx in 1:batchs
            batch_set = train_set[idxs[(batch_idx-1)*batch_size+1:batch_idx*batch_size]]
            if gamma_input == true
                batch_set = append_gamma!.(batch_set)
            end
            batch_set = u_type.(batch_set) |>cgpu
            Flux.train!(loss!, Flux.params(model), batch_set, opt)
        end

        function loss!(tuple)
            return error_loss!(model, tuple[1], tuple[2])
        end

        test_loss[iteration] = batch_loss!(test_set, loss!;gamma_input=gamma_input,append_gamma=append_gamma!)
        train_loss[iteration] = batch_loss!(train_set, loss!;gamma_input=gamma_input,append_gamma=append_gamma!)

        CSV.write("$(test_name) loss.csv", DataFrame(Train=[train_loss[iteration]], Test=[test_loss[iteration]]), delim = ';',append=true)
        @info "$(Dates.format(now(), "HH:MM:SS")) - $(iteration)) Train loss value = $(train_loss[iteration]) , Test loss value = $(test_loss[iteration])"
    end

    model = model|>cpu
    @save "$(test_name).bson" model
    @info "$(Dates.format(now(), "HH:MM:SS")) - Save Model $(test_name).bson"

    model = model|>cgpu
    return model, train_loss, test_loss
end
